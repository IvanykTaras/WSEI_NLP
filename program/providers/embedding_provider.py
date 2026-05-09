from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from gensim.models import Word2Vec
import gensim.downloader as gensim_api

class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Wrapper łączący modele wektorowe (Word2Vec / GloVe) ze standardem Scikit-Learn.
    Dziedziczenie po TransformerMixin dodaje automatycznie metodę fit_transform.
    """
    def __init__(self, word_vectors, dim: int):
        self.word_vectors = word_vectors
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = []
        for text in X:
            words = str(text).lower().split()
            vectors = [self.word_vectors[w] for w in words if w in self.word_vectors]
            if vectors:
                X_transformed.append(np.mean(vectors, axis=0))
            else:
                X_transformed.append(np.zeros(self.dim))
        return np.array(X_transformed)

    def get_feature_names_out(self):
        return np.array([f"emb_dim_{i}" for i in range(self.dim)])


class EmbeddingProvider:
    def __init__(self):
        self.w2v_model = None
        self.glove_model = None

    def get_vectorizer(self, method: str, texts: list = None):
        method = method.lower()

        if method == "bow":
            return CountVectorizer(max_features=5000)

        elif method == "tfidf":
            return TfidfVectorizer(max_features=5000)

        elif method == "word2vec":
            if texts is None:
                raise ValueError("Word2Vec wymaga tekstów do wytrenowania.")
            print("Trenowanie modelu Word2Vec...")
            tokenized_texts = [str(text).lower().split() for text in texts]
            self.w2v_model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=100,
                window=5,
                min_count=2,
                workers=4
            )
            self._save_similar_words(self.w2v_model.wv, label="WORD2VEC")
            return MeanEmbeddingVectorizer(self.w2v_model.wv, dim=100)

        elif method == "glove":
            print("Ładowanie pretrenowanego modelu GloVe (glove-wiki-gigaword-100)...")
            print("Pierwsze uruchomienie może pobrać ~130 MB danych – proszę czekać...")
            self.glove_model = gensim_api.load("glove-wiki-gigaword-100")
            self._save_similar_words(self.glove_model, label="GLOVE")
            return MeanEmbeddingVectorizer(self.glove_model, dim=100)

        else:
            raise ValueError(f"Nieznana metoda embedingu: '{method}'. Dostępne: bow, tfidf, word2vec, glove")

    def _save_similar_words(self, word_vectors, label: str = "WORD2VEC"):
        """Zapisuje podobne słowa dla przykładowych zapytań do pliku lab2_similar_words.txt."""
        queries = ["space", "computer", "science", "music", "car"]
        filename = "lab2_similar_words.txt"

        # Dołączamy do pliku (append) lub tworzymy nowy
        mode = "a" if label == "GLOVE" else "w"
        with open(filename, mode, encoding="utf-8") as f:
            f.write(f"--- PODOBNE SŁOWA ({label}) ---\n\n")
            for query in queries:
                f.write(f"Zapytanie: '{query}'\n")
                if query in word_vectors:
                    similar = word_vectors.most_similar(query, topn=5)
                    for word, score in similar:
                        f.write(f"  -> {word} (Pewność: {score:.4f})\n")
                else:
                    f.write("  -> [Brak słowa w słowniku]\n")
                f.write("\n")
            f.write("\n")