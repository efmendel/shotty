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
- Node.js 18+ with npm (Supabase CLI runs via `npx`, installed as a dev dependency)
- Docker (required by Supabase CLI for local dev)

---

## 1. Supabase (local)

The Supabase config lives in `frontend/supabase/`. Since `supabase` is a dev dependency, run `npm install` first so `npx` can find it.

```bash
cd frontend
npm install
npx supabase start
```

When it finishes, find the **Authentication Keys** section in the output and copy both values:

```
│ Publishable │ sb_publishable_...   ← this is your anon key
│ Secret      │ sb_secret_...        ← this is your service role key
```

The `seed.sql` will automatically create the `videos` storage bucket.

To stop:
```bash
npx supabase stop
```

---

## 2. Flask API

```bash
cd flask-api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `flask-api/.env.local`. Paste the **Secret** (`sb_secret_...`) from the `npx supabase start` output above. `FLASK_SECRET_KEY` is a value you choose — it just needs to match whatever you send in the `x-secret` header when calling the API.

```env
PORT=5001
FLASK_DEBUG=true
FLASK_SECRET_KEY=any-string-you-choose

SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_SERVICE_ROLE_KEY=<paste service_role key here>
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

Create `frontend/.env.local` using the keys from the `npx supabase start` output:

```env
NEXT_PUBLIC_SUPABASE_URL=http://127.0.0.1:54321
NEXT_PUBLIC_SUPABASE_ANON_KEY=<paste Publishable key here>
SUPABASE_SERVICE_ROLE_KEY=<paste Secret key here>
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
# Terminal 1 — Supabase (npm install first so npx can find the supabase CLI)
cd frontend && npm install && npx supabase start

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
