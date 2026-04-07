#!/usr/bin/env python3
import argparse
import json
import os
import sys
from statistics import median

import torch
import torchaudio
from rapidfuzz import fuzz
import whisper

# Compatibility shim:
# Some newer torchaudio versions removed list_audio_backends(), while
# current speechbrain still calls it during import.
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends_fallback():
        return []

    torchaudio.list_audio_backends = _list_audio_backends_fallback  # type: ignore[attr-defined]

from speechbrain.inference.classifiers import EncoderClassifier


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


def normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def verify_sentence(audio_path: str, primary_language: str, threshold: int):
    model = whisper.load_model("small")
    transcript = model.transcribe(audio_path).get("text", "").strip()
    expected = EXPECTED_TEXT_BY_LANGUAGE.get(primary_language, EXPECTED_TEXT_BY_LANGUAGE["Hindi"])
    score = fuzz.token_set_ratio(normalize_text(transcript), normalize_text(expected))
    return {
        "sentencePassed": score >= threshold,
        "sentenceScore": float(score),
        "sentenceThreshold": float(threshold),
        "transcript": transcript,
        "expectedText": expected,
    }


def verify_gender_female(audio_path: str):
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join("/tmp", "sb_spkrec_ecapa_voxceleb"),
    )

    signal, sample_rate = torchaudio.load(audio_path)
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    if sample_rate != 16000:
        signal = torchaudio.functional.resample(signal, sample_rate, 16000)
        sample_rate = 16000

    _ = classifier.encode_batch(signal)

    pitch = torchaudio.functional.detect_pitch_frequency(signal, sample_rate=sample_rate)
    pitch_values = pitch[pitch > 0].flatten().tolist()
    if not pitch_values:
        return {
            "genderPassed": False,
            "genderConfidence": 0.0,
            "detectedGender": "UNKNOWN",
            "reasons": ["Could not detect clear voice pitch from audio."],
        }

    med_pitch = float(median(pitch_values))
    female_threshold_hz = float(os.getenv("VOICE_FEMALE_PITCH_THRESHOLD_HZ", "165"))
    confidence = min(abs(med_pitch - female_threshold_hz) / 120.0, 1.0)
    is_female = med_pitch >= female_threshold_hz

    return {
        "genderPassed": bool(is_female),
        "genderConfidence": float(confidence),
        "detectedGender": "FEMALE" if is_female else "MALE",
        "reasons": []
        if is_female
        else [f"Detected voice pitch suggests non-female voice (median {med_pitch:.1f}Hz)."],
    }


def main():
    parser = argparse.ArgumentParser(description="Partner voice verification")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--threshold", required=False, default="85")
    args = parser.parse_args()

    threshold = int(float(args.threshold))

    sentence_result = verify_sentence(args.audio, args.language, threshold)
    gender_result = verify_gender_female(args.audio)

    output = {
        **sentence_result,
        **gender_result,
    }
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Verifier error: {exc}", file=sys.stderr)
        sys.exit(1)
