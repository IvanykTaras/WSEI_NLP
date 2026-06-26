import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "wsei_nlp_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import re
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VisualizationProvider:
    """
    Warstwa Wizualizacji.
    Odpowiada za generowanie wykresów, WordCloud oraz redukcję wymiarowości (PCA/t-SNE/SVD).
    """

    def __init__(self, plots_dir: Optional[str] = None):
        self.plots_dir = plots_dir or os.path.join(BASE_DIR, "lab2plots")
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

    def _safe_filename_part(self, value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]+", "", str(value).replace(" ", "_").replace("/", "_"))

    # ------------------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, y_true, y_pred, dataset_name: str, model_name: str, embedding_name: str) -> str:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap='Blues')
        plt.title(f'Macierz Pomyłek: {dataset_name} | {model_name.upper()} | {embedding_name.upper()}')
        plt.ylabel('Rzeczywista klasa')
        plt.xlabel('Przewidziana klasa')
        filename = os.path.join(self.plots_dir, f"confusion_{embedding_name}_{model_name}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        return filename

    # ------------------------------------------------------------------
    # Word Cloud — corpus i per klasa
    # ------------------------------------------------------------------
    def plot_wordcloud(self, texts: list, title_suffix: str = "corpus") -> str:
        """Generuje chmurę słów dla przekazanego zbioru tekstów."""
        text_data = " ".join([str(t) for t in texts])
        wordcloud = WordCloud(
            width=800, height=400, background_color='white', max_words=150
        ).generate(text_data)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud — {title_suffix}')

        filename = os.path.join(self.plots_dir, f"wordcloud_{title_suffix}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        return filename

    def plot_wordcloud_per_class(self, texts: list, y: list, target_names: list) -> list:
        """
        Generuje osobną chmurę słów dla każdej klasy.
        Zwraca listę ścieżek do plików.
        """
        saved = []
        texts_arr = np.array(texts)
        y_arr = np.array(y)

        for class_idx, class_name in enumerate(target_names):
            class_texts = texts_arr[y_arr == class_idx]
            if len(class_texts) == 0:
                continue

            safe_name = self._safe_filename_part(class_name)
            path = self.plot_wordcloud(class_texts.tolist(), title_suffix=f"class_{safe_name}")
            saved.append(path)

        return saved

    # ------------------------------------------------------------------
    # Dimensionality Reduction — PCA / t-SNE / SVD dla dokumentów
    # ------------------------------------------------------------------
    def plot_dimensionality_reduction(
        self, X_vectors, y, target_names, dataset_name, model_name, embedding_name, method="svd"
    ) -> str:
        """Rzutuje wysokowymiarowe wektory na przestrzeń 2D i zapisuje wykres."""
        plt.figure(figsize=(12, 8))

        # Konwersja do dense jeśli sparse
        if hasattr(X_vectors, 'toarray'):
            X_dense = X_vectors.toarray()
        else:
            X_dense = np.array(X_vectors)

        if method == "pca":
            reducer = PCA(n_components=2)
            X_reduced = reducer.fit_transform(X_dense)
        elif method == "tsne":
            # t-SNE jest wolny na dużych zbiorach — ograniczamy próbkę
            max_samples = 3000
            if len(X_dense) > max_samples:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_dense), max_samples, replace=False)
                X_dense = X_dense[idx]
                y = np.array(y)[idx]
            perplexity = min(30, max(1, len(X_dense) - 1))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            X_reduced = reducer.fit_transform(X_dense)
        elif method == "svd":
            reducer = TruncatedSVD(n_components=2)
            X_reduced = reducer.fit_transform(X_vectors)
        else:
            raise ValueError(f"Nieznana metoda redukcji: '{method}'. Dostępne: pca, tsne, svd")

        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='tab20', alpha=0.7)
        plt.title(f'Wizualizacja 2D ({method.upper()}) — {dataset_name} | {model_name.upper()} | {embedding_name.upper()}')

        handles, _ = scatter.legend_elements()
        num_classes = len(handles)
        plt.legend(
            handles, target_names[:num_classes],
            title="Klasy", bbox_to_anchor=(1.05, 1), loc='upper left'
        )
        plt.tight_layout()

        filename = os.path.join(
            self.plots_dir,
            f"{dataset_name}_{model_name}_{embedding_name}_{method}_embedding.png"
        )
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        return filename

    # ------------------------------------------------------------------
    # Word Embeddings Visualization (Word2Vec / GloVe)
    # ------------------------------------------------------------------
    def plot_word_embeddings(self, word_vectors, words_to_visualize: list) -> list:
        """
        Tworzy wykres relacji między wybranymi słowami używając PCA oraz t-SNE.
        Akceptuje zarówno KeyedVectors (w2v) jak i obiekty gensim API (GloVe).
        """
        valid_words = [w for w in words_to_visualize if w in word_vectors]
        if not valid_words:
            return []

        extra_words = list(word_vectors.index_to_key[:100])
        all_words = list(dict.fromkeys(valid_words + extra_words))
        vectors = np.array([word_vectors[w] for w in all_words])

        saved_files = []
        for method in ["pca", "tsne"]:
            plt.figure(figsize=(10, 8))

            if method == "pca":
                reducer = PCA(n_components=2)
            else:
                reducer = TSNE(
                    n_components=2, random_state=42,
                    perplexity=min(30, len(all_words) - 1)
                )

            reduced = reducer.fit_transform(vectors)
            plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.2)

            for i, word in enumerate(all_words):
                if word in valid_words:
                    plt.annotate(word, (reduced[i, 0], reduced[i, 1]),
                                 color='red', weight='bold', fontsize=12)
                elif i % 5 == 0:
                    plt.annotate(word, (reduced[i, 0], reduced[i, 1]),
                                 alpha=0.4, fontsize=8)

            plt.title(f'Wizualizacja osadzenia słów ({method.upper()})')
            filename = os.path.join(self.plots_dir, f"word_embedding_{method}.png")
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            plt.close()
            saved_files.append(filename)

        return saved_files

    # ------------------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------------------
    def save_feature_importance(
        self, classifier, vectorizer, target_names, dataset_name: str, model_name: str, top_n: int = 10
    ) -> str:
        try:
            feature_names = vectorizer.get_feature_names_out()
        except AttributeError:
            return ""

        records = []
        if hasattr(classifier, 'coef_'):
            coefs = classifier.coef_
            for class_idx, class_coefs in enumerate(coefs):
                class_name = target_names[class_idx]
                top_indices = np.argsort(class_coefs)[-top_n:]
                for idx in reversed(top_indices):
                    records.append({
                        "class": class_name,
                        "feature": feature_names[idx],
                        "importance": class_coefs[idx]
                    })
        elif hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            top_indices = np.argsort(importances)[-top_n:]
            for idx in reversed(top_indices):
                records.append({
                    "class": "ALL_CLASSES",
                    "feature": feature_names[idx],
                    "importance": importances[idx]
                })
        else:
            return ""

        if records:
            df = pd.DataFrame(records)
            filename = os.path.join(self.plots_dir, f"{dataset_name}_{model_name}_feature_importance.csv")
            df.to_csv(filename, index=False)
            return filename
        return ""
