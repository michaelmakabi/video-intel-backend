"""
video-intel-backend — Render-hosted transcription + summary service.

Flow:
  1. POST /transcribe { id, url }  (id is a vi_links row UUID; url is the social URL)
  2. yt-dlp tries native captions first (free, fast)
  3. Falls back to audio download + OpenAI Whisper if no captions
  4. Sends transcript to GPT-4o-mini → summary, key_points, CTA, skill_candidate
  5. Writes the full result back to vi_links row via Supabase REST
  6. Returns 200 OK once started; result lands via DB update

Env vars (set on Render):
  OPENAI_API_KEY
  SUPABASE_URL          (e.g. https://vaerkevjrupxdbgrxfkk.supabase.co)
  SUPABASE_SERVICE_KEY  (service role — bypasses RLS for updates)
  ALLOWED_ORIGIN        (CORS — e.g. https://video-intel.ai-loren.com)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

if not OPENAI_KEY:
    print("WARN: OPENAI_API_KEY not set — Whisper + summarization will fail", file=sys.stderr)
if not SUPABASE_URL or not SUPABASE_KEY:
    print("WARN: SUPABASE_URL / SUPABASE_SERVICE_KEY not set — DB writes will fail", file=sys.stderr)

openai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="video-intel-backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=2)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class TranscribeRequest(BaseModel):
    id: str
    url: str


# ---------------------------------------------------------------------------
# Supabase REST helpers (no SDK to keep image small)
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
# yt-dlp + Whisper pipeline
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


def whisper_transcribe(audio: Path) -> str:
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY not set")
    with open(audio, "rb") as f:
        result = openai_client.audio.transcriptions.create(
            model="whisper-1", file=f, response_format="text",
        )
    return result if isinstance(result, str) else getattr(result, "text", "")


# ---------------------------------------------------------------------------
# Summarization (GPT-4o-mini)
# ---------------------------------------------------------------------------
SUMMARIZE_PROMPT = """You are an analyst working for Master Makabi (Michael Makabi), COO of BRiX Technologies, founder of Loren AI, 1PM AI, MTIP CRE, and BroadBridge Fund.

You are given a transcript from a short-form video Master Makabi captured. Produce a JSON object with these fields:

- "summary": 3-6 sentences in operator voice, direct, no fluff. State the LESSON the video taught, not "the speaker said...". This is what Master Makabi will read in 6 months when he wants to remember why he saved this.
- "key_points": array of 3-7 strings. Concrete techniques, numbers, names, tools, scripts. NO filler.
- "verification": array of objects with shape {"claim": "...", "status": "verified"|"unverified"|"contradicted", "note": "..."}. Only include claims worth verifying (numbers, attributed quotes, technical assertions). Skip if it's pure opinion. Empty array is fine.
- "call_to_action": ONE sentence. A concrete next step Master Makabi could take in the NEXT 24 HOURS that ties this video's lesson to BRiX, Loren AI, 1PM AI, MTIP, BroadBridge, or his team (Julio, Nicolas, Sanjeev, Shalu, Nataly, Loren). Not "consider doing X" — "do X." If you genuinely cannot tie it, say "No clear application — saved for reference."
- "skill_candidate": boolean. TRUE only if the video describes a REPEATABLE WORKFLOW with clear inputs/outputs that's worth codifying as a skill in his Agent OS. False for pure motivation, opinion, news, one-off product reviews.
- "skill_description": one sentence describing what the skill would do. Empty if skill_candidate=false.

Voice rules:
- Operator. Direct. No corporate hedging.
- Tie everything to his actual ventures by name.
- Avoid generic advice. If you can't make it specific, say so.

Output: JSON only. No markdown, no preamble."""


def summarize(transcript: str, title: str, author: str, platform: str) -> dict:
    if not openai_client:
        return {"summary":"(summarization unavailable)","key_points":[],"verification":[],"call_to_action":"","skill_candidate":False,"skill_description":""}
    user = f"Platform: {platform}\nAuthor: {author}\nTitle: {title}\n\nTRANSCRIPT:\n{transcript}"
    r = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":SUMMARIZE_PROMPT},
            {"role":"user","content":user},
        ],
        response_format={"type":"json_object"},
        temperature=0.4,
    )
    return json.loads(r.choices[0].message.content)


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------
def run_pipeline(row_id: str, url: str) -> None:
    try:
        url = normalize_url(url)
        platform, vid = detect_platform(url)

        # Mark processing
        try:
            supabase_update(row_id, {"status":"processing","platform":platform,"video_id":vid})
        except Exception:
            traceback.print_exc()

        # Metadata
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

            # Stage 2: Whisper
            if not transcript_source:
                audio = download_audio(url, workdir)
                if audio:
                    try:
                        transcript = whisper_transcribe(audio)
                        transcript_source = "whisper"
                    except Exception as e:
                        traceback.print_exc()
                        transcript = transcript or ""
                        if not transcript:
                            supabase_update(row_id, {
                                "status":"failed",
                                "error":f"Whisper failed: {e}",
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

        # Stage 3: summarize
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
    return {"service":"video-intel-backend","version":"1.0","ok":True}


@app.get("/health")
def health():
    return {
        "ok": True,
        "openai": bool(OPENAI_KEY),
        "supabase": bool(SUPABASE_URL and SUPABASE_KEY),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/transcribe")
def transcribe(req: TranscribeRequest, background: BackgroundTasks):
    if not req.url or not req.id:
        raise HTTPException(400, "id and url required")
    background.add_task(run_pipeline, req.id, req.url)
    return {"ok": True, "queued": True, "id": req.id}
