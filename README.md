# Shotty

Tennis swing analysis app. Upload a video, get back an annotated video with biomechanics analysis.

## Architecture

```
shotty/
├── frontend/        # Next.js 16 + React 19 + Tailwind
├── flask-api/       # Flask + MediaPipe pose detection
└── supabase/        # Supabase CLI config (root level)
```

**How it works:**
1. User uploads video via frontend → stored in Supabase Storage
2. Frontend calls Flask API (`POST /api/process`)
3. Flask downloads video, runs MediaPipe pose detection, analyzes swing phases/biomechanics, creates annotated video
4. Annotated video uploaded back to Supabase, signed URL returned to frontend

---

## Prerequisites

- Python 3.12+
- Node.js 18+
- [Supabase CLI](https://supabase.com/docs/guides/cli/getting-started)
- Docker (required by Supabase CLI for local dev)

---

## 1. Supabase (local)

The Supabase config lives in `frontend/supabase/`.

```bash
cd frontend
supabase start
```

This starts:
- API: http://127.0.0.1:54321
- Studio: http://127.0.0.1:54323
- DB: port 54322

On first run, `supabase start` prints your local `anon key` and `service_role key`. The `seed.sql` will automatically create the `videos` storage bucket.

To stop:
```bash
supabase stop
```

---

## 2. Flask API

```bash
cd flask-api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `flask-api/.env.local`:
```env
PORT=5001
FLASK_DEBUG=true
FLASK_SECRET_KEY=dev-secret-key

SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_SERVICE_ROLE_KEY=<service_role key from supabase start output>
SUPABASE_STORAGE_BUCKET=videos
```

Run:
```bash
python app.py
# → http://localhost:5001
```

> **Note:** On first use, MediaPipe will download the `pose_landmarker_heavy.task` model (~200MB).

---

## 3. Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:
```env
NEXT_PUBLIC_SUPABASE_URL=http://127.0.0.1:54321
NEXT_PUBLIC_SUPABASE_ANON_KEY=<anon key from supabase start output>
```

Run:
```bash
npm run dev
# → http://localhost:3000
```

---

## Running everything

Open three terminals:

```bash
# Terminal 1 — Supabase
cd frontend && supabase start

# Terminal 2 — Flask API
cd flask-api && source venv/bin/activate && python app.py

# Terminal 3 — Frontend
cd frontend && npm run dev
```

Then open http://localhost:3000.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Status check |
| GET | `/health` | Health + Supabase config status |
| POST | `/api/process` | Process a video |

**POST /api/process**

Headers:
```
x-secret: <FLASK_SECRET_KEY>
Content-Type: application/json
```

Body:
```json
{ "video_path": "video/123_file.mp4" }
```

Returns a signed URL to the annotated output video.

---

## Tests

```bash
cd flask-api
source venv/bin/activate
pytest
```
