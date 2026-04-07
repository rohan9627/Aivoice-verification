# MeYou Voice Service

Standalone Python microservice for partner voice verification.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Environment

Optional:

```bash
VOICE_VERIFICATION_SERVICE_TOKEN=your-shared-secret
VOICE_FEMALE_PITCH_THRESHOLD_HZ=165
```

## API

`POST /verify`

Multipart form fields:

- `audio` or `file`
- `primaryLanguage`
- `expectedText` (optional)
- `expectedGender` (optional)
- `minDurationSeconds` (optional)
- `minTranscriptSimilarity` (optional)

Health check:

`GET /health`
