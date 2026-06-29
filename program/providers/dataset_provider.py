from sklearn.datasets import fetch_20newsgroups
import numpy as np
import os
import pandas as pd
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_SENTIMENT_FILE = os.path.join(BASE_DIR, "sentiment_dataset.csv")

SUPPORTED_DATASETS = ("20news_group", "imdb", "amazon", "ag_news", "custom")
SENTIMENT_DATASETS = ("imdb", "amazon", "custom")
SKLEARN_DATA_HOME = os.getenv(
    "SKLEARN_DATA",
    os.path.join(tempfile.gettempdir(), "wsei_nlp_sklearn_data")
)


class DatasetProvider:
    """
    Warstwa Danych (Data Layer).
    Odpowiada za pobieranie, ładowanie i wstępne formatowanie zbiorów danych.
    """

    def get_dataset(self, name: str, sample_fraction: float = 1.0, seed: int = 42):
        """
        Zwraca zbiór danych podzielony na teksty (X), etykiety (y) i nazwy klas.
        sample_fraction pozwala na pobranie tylko części danych do szybkich testów.
        """
        name = name.lower()

        if name == "20news_group":
            return self._load_20news_group(sample_fraction, seed)
        elif name == "imdb":
            return self._load_imdb(sample_fraction, seed)
        elif name == "amazon":
            return self._load_amazon(sample_fraction, seed)
        elif name == "ag_news":
            return self._load_ag_news(sample_fraction, seed)
        elif name == "custom":
            return self._load_custom_sentiment(sample_fraction, seed)
        else:
            raise ValueError(
                f"Nieznany zbiór danych: '{name}'. "
                f"Dostępne: {', '.join(SUPPORTED_DATASETS)}"
            )

    def _sample(self, X, y, sample_fraction: float, seed: int):
        if not 0 < sample_fraction <= 1.0:
            raise ValueError("sample_fraction musi być z zakresu (0, 1].")
        if sample_fraction >= 1.0:
            return X, y

        sample_size = max(1, int(len(X) * sample_fraction))
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(X), sample_size, replace=False)
        return X[indices], y[indices]

    def _sample_huggingface(self, dataset, sample_fraction: float, seed: int):
        if not 0 < sample_fraction <= 1.0:
            raise ValueError("sample_fraction musi być z zakresu (0, 1].")
        if sample_fraction >= 1.0:
            return dataset

        sample_size = max(1, int(len(dataset) * sample_fraction))
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(dataset), sample_size, replace=False)
        return dataset.select(indices.tolist())

    # ------------------------------------------------------------------
    # 20 Newsgroups
    # ------------------------------------------------------------------
    def _load_20news_group(self, sample_fraction: float, seed: int):
        print("Pobieranie/ładowanie datasetu 20news_group...")
        dataset = fetch_20newsgroups(
            subset='all',
            remove=('headers', 'footers', 'quotes'),
            data_home=SKLEARN_DATA_HOME
        )

        X = np.array(dataset.data)
        y = np.array(dataset.target)
        target_names = dataset.target_names
        X, y = self._sample(X, y, sample_fraction, seed)

        print(f"Załadowano {len(X)} próbek z 20news_group.")
        return X.tolist(), y.tolist(), list(target_names)

    # ------------------------------------------------------------------
    # IMDB  (HuggingFace datasets)
    # ------------------------------------------------------------------
    def _load_imdb(self, sample_fraction: float, seed: int):
        print("Ładowanie datasetu IMDB (HuggingFace)...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Brak biblioteki 'datasets'. Zainstaluj: pip install datasets"
            )

        ds = load_dataset("imdb", split="train+test")
        ds = self._sample_huggingface(ds, sample_fraction, seed)
        X = np.array(ds["text"])
        y = np.array(ds["label"])
        target_names = ["negative", "positive"]

        print(f"Załadowano {len(X)} próbek z IMDB.")
        return X.tolist(), y.tolist(), target_names

    # ------------------------------------------------------------------
    # Amazon Reviews  (HuggingFace datasets — amazon_polarity)
    # ------------------------------------------------------------------
    def _load_amazon(self, sample_fraction: float, seed: int):
        print("Ładowanie datasetu Amazon Reviews (HuggingFace)...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Brak biblioteki 'datasets'. Zainstaluj: pip install datasets"
            )

        # amazon_polarity: label 0=negative, 1=positive; kolumna tekstu = 'content'
        ds = load_dataset("amazon_polarity", split="train")
        ds = self._sample_huggingface(ds, sample_fraction, seed)
        X = np.array(ds["content"])
        y = np.array(ds["label"])
        target_names = ["negative", "positive"]

        print(f"Załadowano {len(X)} próbek z Amazon Reviews.")
        return X.tolist(), y.tolist(), target_names

    # ------------------------------------------------------------------
    # AG News  (HuggingFace datasets)
    # ------------------------------------------------------------------
    def _load_ag_news(self, sample_fraction: float, seed: int):
        print("Ładowanie datasetu AG News (HuggingFace)...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Brak biblioteki 'datasets'. Zainstaluj: pip install datasets"
            )

        ds = load_dataset("ag_news", split="train+test")
        # Łączymy tytuł i treść artykułu
        texts = [f"{row['text']}" for row in ds]
        labels = list(ds["label"])
        target_names = ds.features["label"].names  # ['World', 'Sports', 'Business', 'Sci/Tech']

        X = np.array(texts)
        y = np.array(labels)

        X, y = self._sample(X, y, sample_fraction, seed)

        print(f"Załadowano {len(X)} próbek z AG News.")
        return X.tolist(), y.tolist(), list(target_names)

    # ------------------------------------------------------------------
    # Custom sentiment dataset
    # ------------------------------------------------------------------
    def _load_custom_sentiment(self, sample_fraction: float, seed: int):
        if not os.path.exists(CUSTOM_SENTIMENT_FILE):
            raise FileNotFoundError(
                f"Brak pliku {CUSTOM_SENTIMENT_FILE}. Dodaj dane komendą /add_sentiment."
            )

        df = pd.read_csv(CUSTOM_SENTIMENT_FILE)
        required_columns = {"text", "label"}
        if not required_columns.issubset(df.columns):
            raise ValueError("Custom dataset musi mieć kolumny: text,label.")

        df = df.dropna(subset=["text", "label"])
        df["text"] = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df = df[(df["text"] != "") & (df["label"] != "")]
        if df.empty:
            raise ValueError("Custom dataset jest pusty.")

        X = df["text"].to_numpy()
        y = df["label"].to_numpy()
        X, y = self._sample(X, y, sample_fraction, seed)
        target_names = ["negatywny", "neutralny", "pozytywny"]

        print(f"Załadowano {len(X)} próbek z custom sentiment dataset.")
        return X.tolist(), y.tolist(), target_names
