import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from providers.dataset_provider import CUSTOM_SENTIMENT_FILE, DatasetProvider
from providers.sequence_model_provider import SEQUENCE_MODELS, SequenceModelProvider

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB3_RESULTS_FILE = os.path.join(BASE_DIR, "lab3results.csv")

SUPPORTED_SENTIMENT_METHODS = (
    "rule",
    "nb",
    "rf",
    "transformer",
    "textblob",
    "stanza",
    "simplernn",
    "lstm",
    "gru",
)
VALID_SENTIMENT_LABELS = ("pozytywny", "neutralny", "negatywny")
SENTIMENT_SAMPLE_FRACTIONS = {
    "custom": 1.0,
    "imdb": 0.1,
    "amazon": 1 / 360,
}


class SentimentProvider:
    def __init__(
        self,
        dataset_provider: Optional[DatasetProvider] = None,
        sequence_provider: Optional[SequenceModelProvider] = None,
    ):
        self.dataset_provider = dataset_provider or DatasetProvider()
        self.sequence_provider = sequence_provider or SequenceModelProvider()
        self._transformer_pipeline = None
        self._stanza_pipeline = None
        self._classic_pipeline_cache = {}

    def predict(self, method: str, text: str, dataset_name: str = "custom") -> dict:
        method = self._validate_method(method)
        text = self._validate_text(text)
        dataset_name = dataset_name.lower()

        if method == "rule":
            return self._rule_predict(text)
        if method == "textblob":
            return self._textblob_predict(text)
        if method == "transformer":
            return self._transformer_predict(text)
        if method == "stanza":
            return self._stanza_predict(text)
        if method in ("nb", "rf"):
            return self._classic_ml_predict(method, dataset_name, text)
        if method in SEQUENCE_MODELS:
            return self.sequence_provider.predict(method, dataset_name, text)
        raise ValueError(f"Nieznana metoda `{method}`.")

    def compare(self, dataset_name: str, methods: list, sample_fraction: float = 0.2, seed: int = 42) -> dict:
        dataset_name = dataset_name.lower()
        methods = [self._validate_method(method.strip().lower()) for method in methods if method.strip()]
        if not methods:
            raise ValueError("Parametr methods musi zawierac co najmniej jedna metode.")

        X, y, _ = self.load_sentiment_dataset(dataset_name, sample_fraction=sample_fraction, seed=seed)
        X_train, X_test, y_train, y_test = self._split_for_compare(X, y, seed)
        results = []
        predictions_by_method = {}

        for method in methods:
            y_pred = self._predict_many_for_compare(method, dataset_name, X_train, y_train, X_test)
            metrics = {
                "dataset": dataset_name,
                "method": method,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "model_path": self._model_path_for(method, dataset_name),
            }
            results.append(metrics)
            predictions_by_method[method] = y_pred

        self._append_results(results)
        return {
            "results": results,
            "X": X,
            "y": y,
            "y_test": y_test,
            "predictions": predictions_by_method,
        }

    def add_custom_example(self, text: str, label: str) -> int:
        text = self._validate_text(text)
        label = label.strip().lower()
        if label not in VALID_SENTIMENT_LABELS:
            raise ValueError(f"Niepoprawna etykieta `{label}`. Dostepne: {', '.join(VALID_SENTIMENT_LABELS)}.")

        os.makedirs(os.path.dirname(CUSTOM_SENTIMENT_FILE), exist_ok=True)
        exists = os.path.exists(CUSTOM_SENTIMENT_FILE)
        row = pd.DataFrame([{"text": text, "label": label}])
        row.to_csv(CUSTOM_SENTIMENT_FILE, mode="a" if exists else "w", header=not exists, index=False)
        self._classic_pipeline_cache.clear()
        return len(pd.read_csv(CUSTOM_SENTIMENT_FILE))

    def load_sentiment_dataset(self, dataset_name: str, sample_fraction: float = 1.0, seed: int = 42):
        X, y, target_names = self.dataset_provider.get_dataset(
            dataset_name, sample_fraction=sample_fraction, seed=seed
        )
        labels = [self._normalize_sentiment_label(label) for label in y]
        target_names = ["negatywny", "neutralny", "pozytywny"]
        return X, labels, target_names

    def _predict_many_for_compare(self, method: str, dataset_name: str, X_train: list, y_train: list, X_test: list) -> list:
        if method == "rule":
            return [self._rule_predict(text)["label"] for text in X_test]
        if method == "textblob":
            return [self._textblob_predict(text)["label"] for text in X_test]
        if method == "transformer":
            return [self._transformer_predict(text)["label"] for text in X_test]
        if method == "stanza":
            return [self._stanza_predict(text)["label"] for text in X_test]
        if method in ("nb", "rf"):
            return self._fit_classic_pipeline(method, X_train, y_train).predict(X_test).tolist()
        if method in SEQUENCE_MODELS:
            return self.sequence_provider.predict_many(method, dataset_name, X_test)
        raise ValueError(f"Nieznana metoda `{method}`.")

    def _classic_ml_predict(self, method: str, dataset_name: str, text: str) -> dict:
        pipeline = self._get_classic_pipeline(method, dataset_name)
        label = str(pipeline.predict([text])[0])
        score = None
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            probabilities = pipeline.predict_proba([text])[0]
            score = float(np.max(probabilities))
        representation = "TF-IDF" if method == "nb" else "CountVectorizer"
        return {"label": label, "score": score, "method": method, "model_path": f"cache pipeline {representation} {dataset_name}"}

    def _get_classic_pipeline(self, method: str, dataset_name: str):
        key = (method, dataset_name)
        if key not in self._classic_pipeline_cache:
            sample_fraction = SENTIMENT_SAMPLE_FRACTIONS[dataset_name]
            X, y, _ = self.load_sentiment_dataset(dataset_name, sample_fraction=sample_fraction, seed=42)
            self._classic_pipeline_cache[key] = self._fit_classic_pipeline(method, X, y)
        return self._classic_pipeline_cache[key]

    def _fit_classic_pipeline(self, method: str, X: list, y: list):
        self._validate_minimum_class_examples(y, minimum=2)
        classifier = MultinomialNB() if method == "nb" else RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        )
        vectorizer = (
            TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
            if method == "nb"
            else CountVectorizer(max_features=5000, ngram_range=(1, 2), binary=True)
        )
        pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ])
        pipeline.fit(X, y)
        return pipeline

    def _rule_predict(self, text: str) -> dict:
        lowered = text.lower()
        positive = {
            "dobry", "dobra", "dobre", "swietny", "świetny", "najlepszy", "uwielbiam",
            "zadowolony", "polecam", "super", "wspanialy", "wspaniały", "pomocna",
            "piekny", "piękny", "wzruszajacy", "wzruszający", "zachwycajace", "zachwycające",
            "udany", "solidny", "pyszny", "pyszne", "mila", "miła", "przyjazna", "intuicyjny",
            "stabilna", "wygodna", "przydatne", "sprawnie", "uczciwie", "punktualny",
            "profesjonalnie", "terminowo", "jasne", "praktyczny", "ciekawy",
        }
        negative = {
            "zly", "zły", "fatalny", "najgorszy", "uszkodzony", "rozczarowany", "slaby",
            "słaby", "nie polecam", "zimne", "niesmaczne", "zawiesza", "traci",
            "nudny", "chaotyczny", "sztucznie", "bez sensu", "marnowal", "marnował",
            "zawiodla", "zawiodła", "niemily", "niemiły", "ignorowal", "ignorował",
            "przesolone", "suche", "frustrujacy", "frustrujący", "niestabilnie",
            "zepsula", "zepsuła", "opryskliwa", "odrzucona", "spoznila", "spóźniła",
            "pognieciona", "nieaktualne", "bledow", "błędów", "niedbale",
        }
        neutral = {
            "poprawny", "poprawna", "poprawne", "przecietny", "przeciętny", "standardowo",
            "zwyczajny", "akceptowalna", "akceptowalny", "podstawowe", "typowy",
            "bez wiekszych", "bez większych", "zgodny z opisem", "ani dobry ani zly",
            "ani dobry, ani zly", "ani dobry, ani zły",
        }
        pos_score = sum(1 for phrase in positive if phrase in lowered)
        neg_score = sum(1 for phrase in negative if phrase in lowered)
        neutral_score = sum(1 for phrase in neutral if phrase in lowered)
        if pos_score > neg_score:
            label = "pozytywny"
        elif neg_score > pos_score:
            label = "negatywny"
        elif neutral_score > 0:
            label = "neutralny"
        else:
            label = "neutralny"
        confidence = max(abs(pos_score - neg_score), neutral_score) / max(1, pos_score + neg_score + neutral_score)
        return {"label": label, "score": float(confidence), "method": "rule", "model_path": "reguly slow kluczowych"}

    def _textblob_predict(self, text: str) -> dict:
        try:
            from textblob import TextBlob
        except ImportError as exc:
            raise ImportError("Brak TextBlob. Zainstaluj zaleznosci z program/requirements.txt.") from exc

        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            label = "pozytywny"
        elif polarity < -0.1:
            label = "negatywny"
        else:
            label = "neutralny"
        return {"label": label, "score": float(abs(polarity)), "method": "textblob", "model_path": "TextBlob polarity"}

    def _transformer_predict(self, text: str) -> dict:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("Brak transformers. Zainstaluj zaleznosci z program/requirements.txt.") from exc

        if self._transformer_pipeline is None:
            self._transformer_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                framework="pt",
            )
        result = self._transformer_pipeline(text)[0]
        raw_label = result["label"].lower()
        score = float(result["score"])
        stars = int(raw_label[0]) if raw_label and raw_label[0].isdigit() else 3
        if stars >= 4:
            label = "pozytywny"
        elif stars <= 2:
            label = "negatywny"
        else:
            label = "neutralny"
        return {
            "label": label,
            "score": score,
            "method": "transformer",
            "model_path": "nlptown/bert-base-multilingual-uncased-sentiment",
        }

    def _stanza_predict(self, text: str) -> dict:
        try:
            import stanza
        except ImportError as exc:
            raise ImportError("Brak Stanza. Zainstaluj zaleznosci z program/requirements.txt.") from exc

        try:
            if self._stanza_pipeline is None:
                self._stanza_pipeline = stanza.Pipeline(
                    lang="en",
                    processors="tokenize,sentiment",
                    download_method=stanza.DownloadMethod.NONE,
                    verbose=False,
                )
            doc = self._stanza_pipeline(text)
        except Exception as exc:
            raise RuntimeError(
                "Nie udało się uruchomić angielskiego modelu Stanza. Zatrzymaj bota i pobierz wymagane pliki: "
                "python3 -c \"import stanza; stanza.download('en', processors='tokenize,sentiment')\". "
                "Następnie uruchom bota ponownie."
            ) from exc

        sentiments = [sentence.sentiment for sentence in doc.sentences]
        mean_sentiment = float(np.mean(sentiments)) if sentiments else 1.0
        if mean_sentiment > 1.2:
            label = "pozytywny"
        elif mean_sentiment < 0.8:
            label = "negatywny"
        else:
            label = "neutralny"
        return {"label": label, "score": abs(mean_sentiment - 1.0), "method": "stanza", "model_path": "stanza:en sentiment"}

    def _split_for_compare(self, X: list, y: list, seed: int):
        counts = pd.Series(y).value_counts()
        stratify = y if not counts.empty and counts.min() >= 2 else None
        return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=stratify)

    def _append_results(self, results: list):
        df = pd.DataFrame(results)
        exists = os.path.exists(LAB3_RESULTS_FILE)
        df.to_csv(LAB3_RESULTS_FILE, mode="a" if exists else "w", header=not exists, index=False)

    def _model_path_for(self, method: str, dataset_name: str) -> str:
        if method in SEQUENCE_MODELS:
            return self.sequence_provider.paths_for(method, dataset_name)["model"]
        if method in ("nb", "rf"):
            return "pipeline trenowany w pamieci"
        return method

    def _normalize_sentiment_label(self, label) -> str:
        if isinstance(label, (int, np.integer)):
            return "negatywny" if int(label) == 0 else "pozytywny"
        value = str(label).strip().lower()
        if value in ("0", "negative", "neg", "negatywny"):
            return "negatywny"
        if value in ("1", "positive", "pos", "pozytywny"):
            return "pozytywny"
        if value in ("2", "neutral", "neu", "neutralny"):
            return "neutralny"
        return value

    def _validate_minimum_class_examples(self, y: list, minimum: int = 2):
        counts = pd.Series(y).value_counts()
        if counts.empty or counts.min() < minimum:
            raise ValueError(
                f"Dataset musi miec co najmniej {minimum} przyklady w kazdej klasie. "
                f"Aktualne liczności: {counts.to_dict()}."
            )

    def _validate_method(self, method: str) -> str:
        method = method.lower()
        if method not in SUPPORTED_SENTIMENT_METHODS:
            raise ValueError(f"Nieznana metoda `{method}`. Dostepne: {', '.join(SUPPORTED_SENTIMENT_METHODS)}.")
        return method

    def _validate_text(self, text: str) -> str:
        text = str(text or "").strip()
        if not text:
            raise ValueError("Tekst nie moze byc pusty.")
        return text
