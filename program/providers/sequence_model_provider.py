import os
import pickle
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SEQUENCE_MODELS = ("simplernn", "lstm", "gru")


@dataclass
class SequenceTrainResult:
    model_name: str
    dataset_name: str
    model_path: str
    tokenizer_path: str
    label_encoder_path: str
    history: object
    test_accuracy: float
    epochs_run: int
    duration_seconds: float


class SequenceModelProvider:
    def __init__(
        self,
        models_dir: str = None,
        max_words: int = 5000,
        max_len: int = 100,
        embedding_dim: int = 64,
        epochs: int = 5,
        batch_size: int = 16,
        seed: int = 42,
    ):
        self.models_dir = models_dir or MODELS_DIR
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self._loaded_model_cache = {}
        os.makedirs(self.models_dir, exist_ok=True)

    def paths_for(self, model_name: str, dataset_name: str) -> dict:
        prefix = f"{model_name}_{dataset_name}"
        return {
            "model": os.path.join(self.models_dir, f"{prefix}.h5"),
            "tokenizer": os.path.join(self.models_dir, f"{prefix}_tokenizer.pkl"),
            "label_encoder": os.path.join(self.models_dir, f"{prefix}_label_encoder.pkl"),
        }

    def is_model_ready(self, model_name: str, dataset_name: str) -> bool:
        paths = self.paths_for(model_name, dataset_name)
        return all(os.path.exists(path) for path in paths.values())

    def list_models(self) -> list:
        rows = []
        if not os.path.exists(self.models_dir):
            return rows

        for filename in sorted(os.listdir(self.models_dir)):
            if not filename.endswith(".h5"):
                continue
            stem = filename[:-3]
            parts = stem.split("_", 1)
            if len(parts) != 2:
                continue
            model_name, dataset_name = parts
            paths = self.paths_for(model_name, dataset_name)
            rows.append({
                "model": model_name,
                "dataset": dataset_name,
                "model_path": paths["model"],
                "has_tokenizer": os.path.exists(paths["tokenizer"]),
                "has_label_encoder": os.path.exists(paths["label_encoder"]),
            })
        return rows

    def train(self, model_name: str, dataset_name: str, X: list, y: list) -> SequenceTrainResult:
        model_name = self._validate_model_name(model_name)
        tf, Tokenizer, pad_sequences, Sequential, Embedding, Dense, Dropout, SimpleRNN, LSTM, GRU, EarlyStopping = (
            self._tensorflow_parts()
        )
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(X)
        X_seq = tokenizer.texts_to_sequences(X)
        X_pad = pad_sequences(X_seq, maxlen=self.max_len, padding="pre", truncating="pre")

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)
        if num_classes < 2:
            raise ValueError("Do treningu potrzeba co najmniej dwoch klas.")
        self._validate_minimum_class_examples(y_encoded, minimum=5)

        stratify = y_encoded if min(np.bincount(y_encoded)) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_pad, y_encoded, test_size=0.2, random_state=self.seed, stratify=stratify
        )

        model = Sequential()
        model.add(Embedding(input_dim=self.max_words, output_dim=self.embedding_dim))
        if model_name == "simplernn":
            model.add(SimpleRNN(96))
        elif model_name == "lstm":
            model.add(LSTM(96))
        elif model_name == "gru":
            model.add(GRU(96))
        model.add(Dropout(0.1))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        training_batch_size = 32 if len(X) >= 1000 else self.batch_size
        start = time.time()
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=training_batch_size,
            verbose=1,
            callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        )
        duration = time.time() - start
        _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        paths = self.paths_for(model_name, dataset_name)
        model.save(paths["model"])
        with open(paths["tokenizer"], "wb") as handle:
            pickle.dump(tokenizer, handle)
        with open(paths["label_encoder"], "wb") as handle:
            pickle.dump(label_encoder, handle)

        return SequenceTrainResult(
            model_name=model_name,
            dataset_name=dataset_name,
            model_path=paths["model"],
            tokenizer_path=paths["tokenizer"],
            label_encoder_path=paths["label_encoder"],
            history=history,
            test_accuracy=float(test_accuracy),
            epochs_run=len(history.history.get("loss", [])),
            duration_seconds=duration,
        )

    def predict(self, model_name: str, dataset_name: str, text: str) -> dict:
        model_name = self._validate_model_name(model_name)
        probabilities, label_encoder, paths = self._predict_probabilities(model_name, dataset_name, [text])
        probabilities = probabilities[0]
        index = int(np.argmax(probabilities))
        return {
            "label": str(label_encoder.inverse_transform([index])[0]),
            "score": float(probabilities[index]),
            "method": model_name,
            "model_path": paths["model"],
        }

    def predict_many(self, model_name: str, dataset_name: str, texts: list) -> list:
        model_name = self._validate_model_name(model_name)
        probabilities, label_encoder, _ = self._predict_probabilities(model_name, dataset_name, texts)
        indices = np.argmax(probabilities, axis=1)
        return [str(label) for label in label_encoder.inverse_transform(indices)]

    def _predict_probabilities(self, model_name: str, dataset_name: str, texts: list):
        model, tokenizer, label_encoder, paths = self._load_model_bundle(model_name, dataset_name)
        _, _, pad_sequences, *_ = self._tensorflow_parts()
        X_seq = tokenizer.texts_to_sequences(texts)
        X_pad = pad_sequences(X_seq, maxlen=self.max_len, padding="pre", truncating="pre")
        probabilities = model.predict(X_pad, verbose=0)
        return probabilities, label_encoder, paths

    def _load_model_bundle(self, model_name: str, dataset_name: str):
        key = (model_name, dataset_name)
        if key in self._loaded_model_cache:
            return self._loaded_model_cache[key]

        paths = self.paths_for(model_name, dataset_name)
        missing = [name for name, path in paths.items() if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                f"Brak zapisanego modelu dla {model_name}/{dataset_name}. "
                f"Najpierw uruchom /train model={model_name} dataset={dataset_name}."
            )

        tf, *_ = self._tensorflow_parts()
        model = tf.keras.models.load_model(paths["model"])
        with open(paths["tokenizer"], "rb") as handle:
            tokenizer = pickle.load(handle)
        with open(paths["label_encoder"], "rb") as handle:
            label_encoder = pickle.load(handle)
        self._loaded_model_cache[key] = (model, tokenizer, label_encoder, paths)
        return self._loaded_model_cache[key]

    def _validate_model_name(self, model_name: str) -> str:
        model_name = model_name.lower()
        if model_name not in SEQUENCE_MODELS:
            raise ValueError(f"Nieznany model sekwencyjny `{model_name}`. Dostepne: {', '.join(SEQUENCE_MODELS)}.")
        return model_name

    def _validate_minimum_class_examples(self, y_encoded, minimum: int):
        counts = np.bincount(y_encoded)
        if len(counts) < 2 or int(counts.min()) < minimum:
            raise ValueError(
                f"Do treningu modelu sekwencyjnego potrzeba co najmniej {minimum} przykladow "
                f"w kazdej klasie. Aktualne liczności: {counts.tolist()}."
            )

    def _tensorflow_parts(self):
        try:
            import tensorflow as tf
            from tensorflow.keras.callbacks import EarlyStopping
            from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Embedding, SimpleRNN
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            from tensorflow.keras.preprocessing.text import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "Brak TensorFlow/Keras. Zainstaluj zaleznosci: python3 -m pip install -r program/requirements.txt"
            ) from exc

        return tf, Tokenizer, pad_sequences, Sequential, Embedding, Dense, Dropout, SimpleRNN, LSTM, GRU, EarlyStopping
