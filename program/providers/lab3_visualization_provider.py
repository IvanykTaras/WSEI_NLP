import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "wsei_nlp_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Lab3VisualizationProvider:
    def __init__(self, plots_dir: str = None):
        self.plots_dir = plots_dir or os.path.join(BASE_DIR, "lab3plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_training_history(self, history, model_name: str, dataset_name: str) -> str:
        hist = history.history if hasattr(history, "history") else history
        filename = os.path.join(self.plots_dir, f"train_history_{model_name}_{dataset_name}.png")

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(hist.get("accuracy", []), label="accuracy")
        plt.plot(hist.get("val_accuracy", []), label="val_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoka")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(hist.get("loss", []), label="loss")
        plt.plot(hist.get("val_loss", []), label="val_loss")
        plt.title("Loss")
        plt.xlabel("Epoka")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=120, bbox_inches="tight")
        plt.close()
        return filename

    def plot_confusion_matrix(self, y_true, y_pred, method: str, dataset_name: str) -> str:
        labels = sorted(set(list(y_true) + list(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        filename = os.path.join(self.plots_dir, f"confusion_{method}_{dataset_name}.png")

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(f"Macierz pomylek: {method} | {dataset_name}")
        plt.ylabel("Rzeczywista klasa")
        plt.xlabel("Predykcja")
        plt.tight_layout()
        plt.savefig(filename, dpi=120, bbox_inches="tight")
        plt.close()
        return filename

    def plot_compare_methods(self, results: list, dataset_name: str) -> str:
        filename = os.path.join(self.plots_dir, f"compare_methods_{dataset_name}.png")
        df = pd.DataFrame(results)
        if df.empty:
            return ""

        plt.figure(figsize=(10, 5))
        sns.barplot(data=df, x="method", y="macro_f1", color="#4C78A8")
        plt.ylim(0, 1)
        plt.title(f"Porownanie metod - {dataset_name}")
        plt.ylabel("Macro F1")
        plt.xlabel("Metoda")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(filename, dpi=120, bbox_inches="tight")
        plt.close()
        return filename

    def plot_wordcloud_per_label(self, texts: list, labels: list, dataset_name: str) -> list:
        saved = []
        for label in sorted(set(labels)):
            label_texts = [text for text, y in zip(texts, labels) if y == label]
            if not label_texts:
                continue
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
                " ".join(str(text) for text in label_texts)
            )
            filename = os.path.join(self.plots_dir, f"wordcloud_{label}_{dataset_name}.png")
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"WordCloud - {label}")
            plt.tight_layout()
            plt.savefig(filename, dpi=120, bbox_inches="tight")
            plt.close()
            saved.append(filename)
        return saved

    def plot_class_distribution(self, labels: list, dataset_name: str) -> str:
        filename = os.path.join(self.plots_dir, f"class_distribution_{dataset_name}.png")
        counts = pd.Series(labels).value_counts().sort_index()
        plt.figure(figsize=(7, 4))
        sns.barplot(x=counts.index, y=counts.values, color="#59A14F")
        plt.title(f"Rozklad klas - {dataset_name}")
        plt.ylabel("Liczba przykladow")
        plt.xlabel("Klasa")
        plt.tight_layout()
        plt.savefig(filename, dpi=120, bbox_inches="tight")
        plt.close()
        return filename
