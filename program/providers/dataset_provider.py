from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np


class DatasetProvider:
    """
    Warstwa Danych (Data Layer).
    Odpowiada za pobieranie, ładowanie i wstępne formatowanie zbiorów danych.
    """

    def get_dataset(self, name: str, sample_fraction: float = 1.0):
        """
        Zwraca zbiór danych podzielony na teksty (X), etykiety (y) i nazwy klas.
        sample_fraction pozwala na pobranie tylko części danych do szybkich testów.
        """
        name = name.lower()

        if name == "20news_group":
            return self._load_20news_group(sample_fraction)
        elif name == "imdb":
            return self._load_imdb(sample_fraction)
        elif name == "amazon":
            return self._load_amazon(sample_fraction)
        elif name == "ag_news":
            return self._load_ag_news(sample_fraction)
        else:
            raise ValueError(
                f"Nieznany zbiór danych: '{name}'. "
                f"Dostępne: 20news_group, imdb, amazon, ag_news"
            )

    # ------------------------------------------------------------------
    # 20 Newsgroups
    # ------------------------------------------------------------------
    def _load_20news_group(self, sample_fraction: float):
        print("Pobieranie/ładowanie datasetu 20news_group...")
        dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

        X = np.array(dataset.data)
        y = np.array(dataset.target)
        target_names = dataset.target_names

        if sample_fraction < 1.0:
            sample_size = int(len(X) * sample_fraction)
            indices = np.random.choice(len(X), sample_size, replace=False)
            X, y = X[indices], y[indices]

        print(f"Załadowano {len(X)} próbek z 20news_group.")
        return X.tolist(), y.tolist(), list(target_names)

    # ------------------------------------------------------------------
    # IMDB  (HuggingFace datasets)
    # ------------------------------------------------------------------
    def _load_imdb(self, sample_fraction: float):
        print("Ładowanie datasetu IMDB (HuggingFace)...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Brak biblioteki 'datasets'. Zainstaluj: pip install datasets"
            )

        ds = load_dataset("imdb", split="train+test")
        X = np.array(ds["text"])
        y = np.array(ds["label"])
        target_names = ["negative", "positive"]

        if sample_fraction < 1.0:
            sample_size = int(len(X) * sample_fraction)
            indices = np.random.choice(len(X), sample_size, replace=False)
            X, y = X[indices], y[indices]

        print(f"Załadowano {len(X)} próbek z IMDB.")
        return X.tolist(), y.tolist(), target_names

    # ------------------------------------------------------------------
    # Amazon Reviews  (HuggingFace datasets — amazon_polarity)
    # ------------------------------------------------------------------
    def _load_amazon(self, sample_fraction: float):
        print("Ładowanie datasetu Amazon Reviews (HuggingFace)...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Brak biblioteki 'datasets'. Zainstaluj: pip install datasets"
            )

        # amazon_polarity: label 0=negative, 1=positive; kolumna tekstu = 'content'
        ds = load_dataset("amazon_polarity", split="train")
        X = np.array(ds["content"])
        y = np.array(ds["label"])
        target_names = ["negative", "positive"]

        if sample_fraction < 1.0:
            sample_size = int(len(X) * sample_fraction)
            indices = np.random.choice(len(X), sample_size, replace=False)
            X, y = X[indices], y[indices]

        print(f"Załadowano {len(X)} próbek z Amazon Reviews.")
        return X.tolist(), y.tolist(), target_names

    # ------------------------------------------------------------------
    # AG News  (HuggingFace datasets)
    # ------------------------------------------------------------------
    def _load_ag_news(self, sample_fraction: float):
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

        if sample_fraction < 1.0:
            sample_size = int(len(X) * sample_fraction)
            indices = np.random.choice(len(X), sample_size, replace=False)
            X, y = X[indices], y[indices]

        print(f"Załadowano {len(X)} próbek z AG News.")
        return X.tolist(), y.tolist(), list(target_names)