# video-intel-backend

FastAPI service powering [video-intel.ai-loren.com](https://video-intel.ai-loren.com).

## Pipeline

1. POST `/transcribe` with `{id, url}` (id = Supabase `vi_links` row UUID, url = TikTok/IG/YouTube URL)
2. yt-dlp tries native captions (free, fast — covers most YouTube)
3. Falls back to audio download + OpenAI Whisper if no captions
4. GPT-4o-mini summarizes → summary, key points, verification, CTA, skill candidacy
5. Result PATCHed to Supabase `vi_links` row → status="done"
6. Frontend listens via Supabase realtime; updates the moment the row lands

## Deploy to Fly.io (recommended — already on free tier, ~1s cold start)

One-time setup (PowerShell, ~2 min):

```powershell
# Install flyctl
iwr https://fly.io/install.ps1 -useb | iex

# Login (opens browser)
fly auth login

# Clone and cd
git clone https://github.com/michaelmakabi/video-intel-backend
cd video-intel-backend

# Launch — uses fly.toml in the repo. Confirm app name when prompted.
fly launch --no-deploy --copy-config

# Set secrets (these stay encrypted, never in source)
fly secrets set OPENAI_API_KEY="sk-..."
fly secrets set SUPABASE_URL="https://vaerkevjrupxdbgrxfkk.supabase.co"
fly secrets set SUPABASE_SERVICE_KEY="eyJ..."
fly secrets set ALLOWED_ORIGIN="https://video-intel.ai-loren.com"

# Deploy
fly deploy
```

After deploy, your backend is live at `https://loren-video-intel.fly.dev` (or whatever app name you picked). Paste that URL into the frontend Settings panel.

Get the SUPABASE_SERVICE_KEY from: https://supabase.com/dashboard/project/vaerkevjrupxdbgrxfkk/settings/api → `service_role` (NOT anon).

## Deploy to Render (alternative)

`render.yaml` is in the repo — connect repo on render.com → free plan auto-detected → set the same secrets in Render dashboard.

## Cost

- yt-dlp + native captions path: $0
- Whisper API path: ~$0.006/min of audio (only fires for IG Reels + silent TikToks)
- GPT-4o-mini summary: ~$0.0002 per video
- Fly.io free tier: $0/mo within the free allowance (small footprint, auto-stops when idle)

So unless you transcribe Instagram Reels constantly, this runs ~$0/mo all-in.
