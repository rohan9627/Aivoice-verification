# MeYou Voice Service

Standalone Python microservice for partner voice verification.

## Current AI verification mode

The service currently uses OpenAI speech-to-text for verification.

How it works:

1. [app.py](./app.py) receives the uploaded audio on `POST /verify`
2. [verify_partner_voice.py](./verify_partner_voice.py) sends the audio to OpenAI transcription
3. the returned transcript is compared against the expected verification sentence using `rapidfuzz`
4. verification passes when the transcript similarity reaches the configured threshold

Current model:

- `gpt-4o-mini-transcribe` by default

Current behavior:

- transcript verification is active
- gender verification is temporarily disabled in this mode
- the previous local AI stack is preserved in [verify_partner_voice_legacy.py](./verify_partner_voice_legacy.py)
- the old heavy dependencies are commented out in [requirements.txt](./requirements.txt)

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
OPENAI_API_KEY=your-openai-api-key
OPENAI_TRANSCRIBE_MODEL=gpt-4o-mini-transcribe
VOICE_VERIFICATION_SERVICE_TOKEN=your-shared-secret
VOICE_FEMALE_PITCH_THRESHOLD_HZ=165
```

`VOICE_FEMALE_PITCH_THRESHOLD_HZ` is only used by the legacy local-model verifier.

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

## What was used before

The earlier local verification stack used:

- Whisper for transcription
- RapidFuzz for transcript similarity
- SpeechBrain + Torch/Torchaudio for female voice checks

That implementation is preserved in [verify_partner_voice_legacy.py](./verify_partner_voice_legacy.py) so it can be restored later if needed.
