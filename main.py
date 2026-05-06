"""
video-intel-backend  -  Fly.io-hosted transcription + summary service.

Pipeline:
  1. POST /transcribe { id, url }
  2. yt-dlp tries native captions first (free, fast)
  3. Falls back to audio download + ElevenLabs Scribe STT if no captions
  4. Anthropic Claude Sonnet summarizes -> summary, key_points, CTA, skill_candidate
  5. Writes the full result back to vi_links via Supabase REST
  6. Returns 200 OK once started; result lands via DB update + frontend realtime sub

Env vars (set on Fly via `fly secrets set ...`):
  ELEVENLABS_API_KEY        ElevenLabs Scribe transcription
  ANTHROPIC_API_KEY         Claude summarization
  SUPABASE_URL              https://vaerkevjrupxdbgrxfkk.supabase.co
  SUPABASE_SERVICE_KEY      service_role JWT (bypasses RLS for backend writes)
  ALLOWED_ORIGIN            https://video-intel.ai-loren.com
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from anthropic import Anthropic
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ELEVENLABS_KEY = os.environ.get("ELEVENLABS_API_KEY")
ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_API_KEY")
SUPABASE_URL   = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY   = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

if not ELEVENLABS_KEY:
    print("WARN: ELEVENLABS_API_KEY not set - transcription fallback will fail", file=sys.stderr)
if not ANTHROPIC_KEY:
    print("WARN: ANTHROPIC_API_KEY not set - summarization will fail", file=sys.stderr)
if not SUPABASE_URL or not SUPABASE_KEY:
    print("WARN: SUPABASE_URL / SUPABASE_SERVICE_KEY not set - DB writes will fail", file=sys.stderr)

claude = Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="video-intel-backend", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class TranscribeRequest(BaseModel):
    id: str
    url: str


# ---------------------------------------------------------------------------
# Supabase REST helpers
# ---------------------------------------------------------------------------
def supabase_update(row_id: str, patch: dict) -> None:
    url = f"{SUPABASE_URL}/rest/v1/vi_links?id=eq.{row_id}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    r = httpx.patch(url, headers=headers, json=patch, timeout=30)
    r.raise_for_status()


# ---------------------------------------------------------------------------
# Platform detection + URL normalization
# ---------------------------------------------------------------------------
def detect_platform(url: str) -> tuple[str, str]:
    host = urlparse(url).netloc.lower().lstrip("www.")
    if "tiktok.com" in host:
        m = re.search(r"/video/(\d+)", url)
        return ("tiktok", m.group(1) if m else url)
    if "instagram.com" in host:
        m = re.search(r"/(?:reel|p|tv)/([A-Za-z0-9_-]+)", url)
        return ("instagram", m.group(1) if m else url)
    if "youtube.com" in host or "youtu.be" in host:
        m = re.search(r"(?:v=|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})", url)
        return ("youtube", m.group(1) if m else url)
    return ("other", url)


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        m = re.search(r"v=([A-Za-z0-9_-]{11})", parsed.query)
        if m:
            return f"https://www.youtube.com/watch?v={m.group(1)}"
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


# ---------------------------------------------------------------------------
# yt-dlp wrappers
# ---------------------------------------------------------------------------
def vtt_to_text(vtt: str) -> str:
    out, last = [], None
    for line in vtt.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        if "-->" in line or re.match(r"^\d+$", line):
            continue
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line and line != last:
            out.append(line); last = line
    return " ".join(out)


def get_metadata(url: str) -> dict:
    r = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--dump-json", "--no-warnings", "--skip-download", url],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return json.loads(r.stdout)


def try_native_captions(url: str, workdir: Path) -> str | None:
    sub_dir = workdir / "subs"
    sub_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--skip-download", "--write-auto-sub", "--write-sub",
        "--sub-lang", "en,en-US,en-GB", "--sub-format", "vtt",
        "--no-warnings", "-o", str(sub_dir / "%(id)s.%(ext)s"), url,
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    vtts = list(sub_dir.glob("*.vtt"))
    if not vtts:
        return None
    return vtt_to_text(vtts[0].read_text(encoding="utf-8", errors="ignore"))


def download_audio(url: str, workdir: Path) -> Path | None:
    out = str(workdir / "audio.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x", "--audio-format", "mp3", "--audio-quality", "5",
        "--no-warnings", "-o", out, url,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
    except subprocess.CalledProcessError:
        return None
    cands = list(workdir.glob("audio.mp3"))
    return cands[0] if cands else None


# ---------------------------------------------------------------------------
# ElevenLabs Scribe transcription
# ---------------------------------------------------------------------------
def elevenlabs_transcribe(audio: Path) -> str:
    """POST audio file to ElevenLabs Scribe STT, return transcript text."""
    if not ELEVENLABS_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": ELEVENLABS_KEY}
    with open(audio, "rb") as f:
        files = {"file": (audio.name, f, "audio/mpeg")}
        data = {"model_id": "scribe_v1"}
        r = httpx.post(url, headers=headers, files=files, data=data, timeout=180)
    r.raise_for_status()
    body = r.json()
    # API returns { "text": "...", "language_code": "...", "words": [...] }
    return body.get("text", "")


# ---------------------------------------------------------------------------
# Anthropic Claude summarization
# ---------------------------------------------------------------------------
SUMMARIZE_PROMPT = """You are an analyst working for Master Makabi (Michael Makabi), COO of BRiX Technologies, founder of Loren AI, 1PM AI, MTIP CRE, and BroadBridge Fund.

You receive a transcript from a short-form video Master Makabi captured. Produce a JSON object with these fields:

- "summary": 3-6 sentences in operator voice. Direct, no fluff. State the LESSON the video taught, not "the speaker said...". This is what Master Makabi will read in 6 months when he wants to remember why he saved this clip.
- "key_points": array of 3-7 strings. Concrete techniques, numbers, names, tools, scripts, frameworks. NO filler.
- "verification": array of objects with shape {"claim": "...", "status": "verified"|"unverified"|"contradicted", "note": "..."}. Only include claims worth verifying (numbers, attributed quotes, technical assertions). Skip if it's pure opinion. Empty array is fine.
- "call_to_action": ONE sentence. A concrete next step Master Makabi could take in the NEXT 24 HOURS that ties this video's lesson to BRiX, Loren AI, 1PM AI, MTIP, BroadBridge, or his team (Julio, Nicolas, Sanjeev, Shalu, Nataly, Loren). Not "consider doing X" - "do X." If you genuinely cannot tie it, say "No clear application - saved for reference."
- "skill_candidate": boolean. TRUE only if the video describes a REPEATABLE WORKFLOW with clear inputs/outputs that's worth codifying as a skill in his Agent OS. False for pure motivation, opinion, news, one-off product reviews.
- "skill_description": one sentence describing what the skill would do. Empty string if skill_candidate=false.

Voice rules:
- Operator. Direct. No corporate hedging.
- Tie everything to his actual ventures by name.
- Avoid generic advice. If you can't make it specific, say so.

Output: JSON ONLY. No markdown, no preamble. Begin with { and end with }."""


def summarize(transcript: str, title: str, author: str, platform: str) -> dict:
    if not claude:
        return {"summary":"(summarization unavailable - ANTHROPIC_API_KEY not set)","key_points":[],"verification":[],"call_to_action":"","skill_candidate":False,"skill_description":""}
    user = f"Platform: {platform}\nAuthor: {author}\nTitle: {title}\n\nTRANSCRIPT:\n{transcript}"
    msg = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=SUMMARIZE_PROMPT,
        messages=[{"role":"user","content":user}],
    )
    text = msg.content[0].text if msg.content else "{}"
    # Strip any accidental markdown fences
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"summary": text[:500],"key_points":[],"verification":[],"call_to_action":"","skill_candidate":False,"skill_description":""}


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------
def run_pipeline(row_id: str, url: str) -> None:
    try:
        url = normalize_url(url)
        platform, vid = detect_platform(url)

        try:
            supabase_update(row_id, {"status":"processing","platform":platform,"video_id":vid})
        except Exception:
            traceback.print_exc()

        try:
            meta = get_metadata(url)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or str(e))[-800:]
            supabase_update(row_id, {"status":"failed","error":f"yt-dlp metadata failed: {err}"})
            return

        title = meta.get("title","")
        author = meta.get("uploader") or meta.get("channel") or ""
        duration = meta.get("duration", 0) or 0

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            # Stage 1: native captions
            transcript = try_native_captions(url, workdir)
            transcript_source = "native_captions" if transcript and len(transcript) > 50 else None

            # Stage 2: ElevenLabs Scribe fallback
            if not transcript_source:
                audio = download_audio(url, workdir)
                if audio:
                    try:
                        transcript = elevenlabs_transcribe(audio)
                        transcript_source = "elevenlabs_scribe"
                    except Exception as e:
                        traceback.print_exc()
                        if not transcript:
                            supabase_update(row_id, {
                                "status":"failed",
                                "error":f"ElevenLabs transcription failed: {e}",
                                "title":title,"author":author,"duration_sec":duration,
                            })
                            return
                else:
                    supabase_update(row_id, {
                        "status":"failed",
                        "error":"Could not extract audio (private/geoblocked/dead link?)",
                        "title":title,"author":author,"duration_sec":duration,
                    })
                    return

        # Stage 3: summarize with Claude
        try:
            summary = summarize(transcript, title, author, platform)
        except Exception as e:
            traceback.print_exc()
            summary = {"summary":f"(summarization failed: {e})","key_points":[],"verification":[],"call_to_action":"","skill_candidate":False,"skill_description":""}

        # Final write
        supabase_update(row_id, {
            "title": title,
            "author": author,
            "duration_sec": duration,
            "transcript": transcript,
            "transcript_source": transcript_source,
            "summary": summary.get("summary",""),
            "key_points": summary.get("key_points",[]),
            "verification": summary.get("verification",[]),
            "call_to_action": summary.get("call_to_action",""),
            "skill_candidate": summary.get("skill_candidate", False),
            "skill_description": summary.get("skill_description",""),
            "status": "done",
            "error": None,
        })
    except Exception as e:
        traceback.print_exc()
        try:
            supabase_update(row_id, {"status":"failed","error":str(e)[:1000]})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"service":"video-intel-backend","version":"2.0","ok":True}


@app.get("/health")
def health():
    return {
        "ok": True,
        "elevenlabs": bool(ELEVENLABS_KEY),
        "anthropic": bool(ANTHROPIC_KEY),
        "supabase": bool(SUPABASE_URL and SUPABASE_KEY),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/transcribe")
def transcribe(req: TranscribeRequest, background: BackgroundTasks):
    if not req.url or not req.id:
        raise HTTPException(400, "id and url required")
    background.add_task(run_pipeline, req.id, req.url)
    return {"ok": True, "queued": True, "id": req.id}
