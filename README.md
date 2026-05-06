# video-intel-backend

FastAPI service powering [video-intel.ai-loren.com](https://video-intel.ai-loren.com).

## Pipeline

1. POST `/transcribe` with `{id, url}` (id = Supabase `vi_links` row UUID, url = TikTok/IG/YouTube URL)
2. yt-dlp tries native captions (free, fast — covers most YouTube)
3. Falls back to audio download + OpenAI Whisper if no captions
4. GPT-4o-mini summarizes → summary, key points, verification, CTA, skill candidacy
5. Result PATCHed to Supabase `vi_links` row → status="done"
6. Frontend listens via Supabase realtime; updates the moment the row lands

## Deploy to Render

1. New → Web Service → connect GitHub repo `michaelmakabi/video-intel-backend`
2. Render auto-detects `render.yaml`. Plan: starter ($7/mo, no cold start).
3. Set the two `sync: false` env vars in Render dashboard:
   - `OPENAI_API_KEY` — your OpenAI key
   - `SUPABASE_SERVICE_KEY` — Loren AI Supabase service role key (Supabase dashboard → Project Settings → API → service_role)
4. Deploy. Once live, copy the `*.onrender.com` URL.
5. In the frontend (`video-intel.ai-loren.com`) → Settings → Backend URL → paste the Render URL → Save.

## Cost (per video)

- Native captions path: $0 (free, fast)
- Whisper path: ~$0.006 per minute (Whisper) + ~$0.0002 (GPT-4o-mini summary) ≈ $0.006/min total
- Render starter: $7/mo flat
