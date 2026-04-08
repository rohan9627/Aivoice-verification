# MeYou Voice Service

Standalone Python microservice for partner voice verification.

## Current AI verification mode

The service currently uses Gemini 2.5 Flash for transcript-based verification.

How it works:

1. [app.py](./app.py) receives the uploaded audio on `POST /verify`
2. [verify_partner_voice.py](./verify_partner_voice.py) sends the audio to Gemini 2.5 Flash
3. Gemini returns both a transcript and a lightweight perceived voice gender classification
4. the transcript is compared against the expected verification sentence using `rapidfuzz`
5. verification passes only when both the sentence similarity and expected gender check pass

Current model:

- `gemini-2.5-flash` by default

Current behavior:

- transcript verification is active
- lightweight Gemini-based perceived gender blocking is active
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
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash
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
