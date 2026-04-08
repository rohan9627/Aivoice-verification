import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile


app = FastAPI(title="MeYou Voice Verification Service")
BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = BASE_DIR / "verify_partner_voice.py"


def require_auth(authorization: str | None):
    expected_token = (os.getenv("VOICE_VERIFICATION_SERVICE_TOKEN") or "").strip()
    if not expected_token:
        return

    if authorization != f"Bearer {expected_token}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/verify")
async def verify(
    authorization: str | None = Header(default=None),
    audio: UploadFile | None = File(default=None),
    file: UploadFile | None = File(default=None),
    primaryLanguage: str = Form(...),
    expectedText: str = Form(default=""),
    expectedGender: str = Form(default="FEMALE"),
    minDurationSeconds: str = Form(default="3"),
    minTranscriptSimilarity: str = Form(default="85"),
):
    del minDurationSeconds

    require_auth(authorization)

    uploaded_file = audio or file
    if uploaded_file is None:
        raise HTTPException(status_code=400, detail="Audio file is required")

    temp_dir = tempfile.mkdtemp(prefix="meyou-voice-service-")
    try:
        extension = Path(uploaded_file.filename or "").suffix or ".m4a"
        audio_path = Path(temp_dir) / f"voice{extension}"
        with audio_path.open("wb") as destination:
            shutil.copyfileobj(uploaded_file.file, destination)

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--audio",
                str(audio_path),
                "--language",
                primaryLanguage,
                "--expected-text",
                expectedText,
                "--expected-gender",
                expectedGender,
                "--threshold",
                str(minTranscriptSimilarity),
            ],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
            },
        )

        if completed.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=(completed.stderr or completed.stdout or "Voice verification failed").strip(),
            )

        try:
            return json.loads(completed.stdout.strip())
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=500, detail=f"Invalid verifier output: {error}") from error
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
