#!/usr/bin/env python3
import argparse
import json
import os
import sys
import mimetypes
import time

from google import genai
from google.genai import types
from rapidfuzz import fuzz


EXPECTED_TEXT_BY_LANGUAGE = {
    "Hindi": "नमस्ते दोस्ती बहुत खास होती है अच्छे दोस्त हमेशा साथ देते हैं दोस्त खुशी बढ़ाते हैं और दुख कम करते हैं उनके बिना सब अधूरा है धन्यवाद",
    "Telugu": "నమస్కారం స్నేహం చాలా ప్రత్యేకం మంచి స్నేహితులు ఎప్పుడూ తోడుగా ఉంటారు స్నేహితులు ఆనందాన్ని పెంచి బాధను తగ్గిస్తారు వాళ్లు లేకపోతే జీవితం అసంపూర్ణంగా అనిపిస్తుంది ధన్యవాదాలు",
    "Bangla": "নমস্কার বন্ধুত্ব খুবই বিশেষ ভালো বন্ধু সবসময় পাশে থাকে বন্ধুরা আনন্দ বাড়ায় এবং দুঃখ কমায় তাদের ছাড়া জীবন অসম্পূর্ণ লাগে ধন্যবাদ",
    "Tamil": "வணக்கம் நட்பு மிகவும் சிறப்பு வாய்ந்தது நல்ல நண்பர்கள் எப்போதும் துணையாக இருப்பார்கள் நண்பர்கள் மகிழ்ச்சியை அதிகரித்து துக்கத்தை குறைப்பார்கள் அவர்கள் இல்லாமல் வாழ்க்கை முழுமையற்றதாக தோன்றும் நன்றி",
    "Kannada": "ನಮಸ್ಕಾರ ಸ್ನೇಹವು ತುಂಬಾ ವಿಶೇಷ ಒಳ್ಳೆಯ ಸ್ನೇಹಿತರು ಯಾವಾಗಲೂ ಜೊತೆಗಿರುತ್ತಾರೆ ಸ್ನೇಹಿತರು ಸಂತೋಷವನ್ನು ಹೆಚ್ಚಿಸಿ ದುಃಖವನ್ನು ಕಡಿಮೆ ಮಾಡುತ್ತಾರೆ ಅವರಿಲ್ಲದೆ ಜೀವನ ಅಪೂರ್ಣವೆನಿಸುತ್ತದೆ ಧನ್ಯವಾದಗಳು",
    "Malayalam": "നമസ്കാരം സൗഹൃദം വളരെ പ്രത്യേകമാണ് നല്ല സുഹൃത്തുക്കൾ എപ്പോഴും കൂടെയുണ്ടാകും സുഹൃത്തുകൾ സന്തോഷം കൂട്ടുകയും ദുഃഖം കുറയ്ക്കുകയും ചെയ്യും അവരില്ലാതെ ജീവിതം അപൂർണ്ണമായി തോന്നും നന്ദി",
    "Marathi": "नमस्कार मैत्री खूप खास असते चांगले मित्र नेहमी साथ देतात मित्र आनंद वाढवतात आणि दुःख कमी करतात त्यांच्याशिवाय जीवन अपूर्ण वाटते धन्यवाद",
    "Punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਦੋਸਤੀ ਬਹੁਤ ਖਾਸ ਹੁੰਦੀ ਹੈ ਚੰਗੇ ਦੋਸਤ ਹਮੇਸ਼ਾਂ ਨਾਲ ਖੜ੍ਹੇ ਰਹਿੰਦੇ ਹਨ ਦੋਸਤ ਖੁਸ਼ੀ ਵਧਾਉਂਦੇ ਹਨ ਅਤੇ ਦੁੱਖ ਘਟਾਉਂਦੇ ਹਨ ਉਨ੍ਹਾਂ ਤੋਂ ਬਿਨਾਂ ਜ਼ਿੰਦਗੀ ਅਧੂਰੀ ਲੱਗਦੀ ਹੈ ਧੰਨਵਾਦ",
}

LANGUAGE_CODE_BY_LANGUAGE = {
    "Hindi": "hi",
    "Telugu": "te",
    "Bangla": "bn",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Punjabi": "pa",
}


def normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def get_expected_text(primary_language: str, expected_text: str):
    cleaned = (expected_text or "").strip()
    if cleaned:
        return cleaned
    return EXPECTED_TEXT_BY_LANGUAGE.get(primary_language, EXPECTED_TEXT_BY_LANGUAGE["Hindi"])


def should_retry_gemini_error(error: Exception):
    message = str(error).lower()
    transient_markers = [
        "503",
        "unavailable",
        "high demand",
        "try again later",
        "resource exhausted",
        "temporarily unavailable",
    ]
    return any(marker in message for marker in transient_markers)


def transcribe_audio(audio_path: str, primary_language: str, expected_text: str):
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured")

    client = genai.Client(api_key=api_key)
    expected = get_expected_text(primary_language, expected_text)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    mime_type = mimetypes.guess_type(audio_path)[0] or "audio/mp4"
    prompt = (
        "Generate a transcript of the speech only. "
        "Return only the spoken transcript text with no explanation. "
        f"Primary language: {primary_language}. "
        f"Expected verification sentence for reference: {expected}"
    )
    max_retries = max(0, int(os.getenv("GEMINI_MAX_RETRIES", "3")))
    retry_delay_seconds = max(1.0, float(os.getenv("GEMINI_RETRY_DELAY_SECONDS", "2")))

    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        response = None
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        prompt,
                        types.Part.from_bytes(
                            data=audio_bytes,
                            mime_type=mime_type,
                        ),
                    ],
                )
                break
            except Exception as error:
                last_error = error
                if attempt >= max_retries or not should_retry_gemini_error(error):
                    raise
                time.sleep(retry_delay_seconds * (attempt + 1))

        if response is None and last_error is not None:
            raise last_error

    transcript = (getattr(response, "text", None) or "").strip()
    usage = getattr(response, "usage_metadata", None)
    duration_seconds = 0.0
    if transcript == "":
        raise RuntimeError("Gemini returned an empty transcription")

    return transcript, expected, duration_seconds, model


def verify_sentence(audio_path: str, primary_language: str, expected_text: str, threshold: int):
    transcript, expected, duration_seconds, model = transcribe_audio(
        audio_path, primary_language, expected_text
    )
    score = fuzz.token_set_ratio(normalize_text(transcript), normalize_text(expected))
    sentence_passed = score >= threshold

    return {
        "verified": sentence_passed,
        "reason": (
            "Speech verification passed"
            if sentence_passed
            else "Spoken sentence did not match the expected script"
        ),
        "metrics": {
            "durationSeconds": duration_seconds,
            "transcript": transcript,
            "transcriptSimilarity": float(score),
            "predictedGender": "UNKNOWN",
            "modelPredictedGender": "UNKNOWN",
            "modelPredictedGenderLabel": None,
            "modelPredictedGenderConfidence": None,
            "modelError": "Gender verification disabled while Gemini transcription mode is active",
            "pitchPredictedGender": "UNKNOWN",
            "pitchHz": None,
            "speechbrainEmbeddingAvailable": False,
            "speechbrainEmbeddingNorm": None,
            "transcriptionModel": model,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Partner voice verification")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--expected-text", required=False, default="")
    parser.add_argument("--threshold", required=False, default="85")
    args = parser.parse_args()

    threshold = int(float(args.threshold))
    output = verify_sentence(args.audio, args.language, args.expected_text, threshold)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Verifier error: {exc}", file=sys.stderr)
        sys.exit(1)
