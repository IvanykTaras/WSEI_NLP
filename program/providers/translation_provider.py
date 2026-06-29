from typing import Optional


SUPPORTED_TRANSLATION_LANGUAGES = ("en", "pl", "de", "fr", "es")
TRANSLATION_MODEL = "facebook/m2m100_418M"
MAX_TRANSLATION_TEXT_LENGTH = 2000


class TranslationProvider:
    def __init__(self, model_name: str = TRANSLATION_MODEL):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    def detect_language(self, text: str) -> dict:
        text = self._validate_text(text, max_length=3000)
        try:
            from langdetect import DetectorFactory, detect_langs
            from langdetect.lang_detect_exception import LangDetectException
        except ImportError as exc:
            raise ImportError(
                "Brak langdetect. Uruchom `python3 -m pip install -r program/requirements.txt`."
            ) from exc

        DetectorFactory.seed = 0
        try:
            predictions = detect_langs(text)
        except LangDetectException as exc:
            raise ValueError("Tekst jest zbyt krótki lub niejednoznaczny do wykrycia języka.") from exc
        best = predictions[0]
        return {
            "text": text,
            "language": best.lang,
            "confidence": round(float(best.prob), 4),
            "alternatives": [
                {"language": prediction.lang, "confidence": round(float(prediction.prob), 4)}
                for prediction in predictions[:3]
            ],
        }

    def translate(self, text: str, target_language: str, source_language: Optional[str] = None) -> dict:
        text = self._validate_text(text, max_length=MAX_TRANSLATION_TEXT_LENGTH)
        target_language = self._validate_supported_language(target_language)
        detection = None
        if source_language:
            source_language = self._validate_supported_language(source_language)
        else:
            detection = self.detect_language(text)
            source_language = detection["language"]
            if source_language not in SUPPORTED_TRANSLATION_LANGUAGES:
                raise ValueError(
                    f"Wykryto nieobsługiwany język `{source_language}`. Dostępne: "
                    f"{', '.join(SUPPORTED_TRANSLATION_LANGUAGES)}."
                )

        if source_language == target_language:
            raise ValueError("Język źródłowy i docelowy muszą być różne.")
        self._load_local_model()

        self._tokenizer.src_lang = source_language
        encoded = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        generated = self._model.generate(
            **encoded,
            forced_bos_token_id=self._tokenizer.get_lang_id(target_language),
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
        )
        translated = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return {
            "source_language": source_language,
            "target_language": target_language,
            "source_text": text,
            "translation": translated,
            "model": self.model_name,
            "detection_confidence": detection["confidence"] if detection else None,
        }

    def _load_local_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        try:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        except ImportError as exc:
            raise ImportError(
                "Brak Transformers. Uruchom `python3 -m pip install -r program/requirements.txt`."
            ) from exc
        try:
            self._tokenizer = M2M100Tokenizer.from_pretrained(
                self.model_name, local_files_only=True
            )
            self._model = M2M100ForConditionalGeneration.from_pretrained(
                self.model_name, local_files_only=True
            )
            self._model.eval()
        except (OSError, ValueError) as exc:
            self._tokenizer = None
            self._model = None
            raise FileNotFoundError(
                f"Brak lokalnego modelu `{self.model_name}`. Pobierz go jednorazowo komendą "
                "podaną w sekcji Lab4 README."
            ) from exc

    @staticmethod
    def _validate_text(text: str, max_length: int) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("Tekst nie może być pusty.")
        if len(text) > max_length:
            raise ValueError(f"Tekst może mieć maksymalnie {max_length} znaków.")
        return text

    @staticmethod
    def _validate_supported_language(language: str) -> str:
        language = (language or "").strip().lower()
        if language not in SUPPORTED_TRANSLATION_LANGUAGES:
            raise ValueError(
                f"Język musi być jednym z: {', '.join(SUPPORTED_TRANSLATION_LANGUAGES)}."
            )
        return language
