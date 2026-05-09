import logging
import os
import numpy as np
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from providers.dataset_provider import DatasetProvider
from providers.embedding_provider import EmbeddingProvider
from providers.classification_provider import ClassificationProvider, ALL_MODELS
from providers.visualization_provider import VisualizationProvider

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

BOT_TOKEN = '8622639294:AAGSsDW82owPsvm5vZVJ3zIk7_NnKbnzhLI'

dataset_provider = DatasetProvider()
classification_provider = ClassificationProvider()
visualization_provider = VisualizationProvider()

# Seedy używane dla kolejnych uruchomień (run=1 → seed[0], run=2 → seed[0]+seed[1], itd.)
SEEDS = [42, 1337, 2024]

CSV_FILE = "lab2results.csv"
PLOTS_DIR = "lab2plots"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_params(args: list) -> dict:
    params = {}
    for part in args:
        if '=' in part:
            key, value = part.split('=', 1)
            params[key.lower()] = value.lower()
    return params


def _save_result(embedding: str, model: str, accuracy: float, macro_f1: float, seed: int):
    row = pd.DataFrame([{
        "embedding": embedding,
        "model": model,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "seed": seed,
    }])
    if os.path.exists(CSV_FILE):
        row.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        row.to_csv(CSV_FILE, mode='w', header=True, index=False)


async def _send_photo_safe(update: Update, path: str, caption: str = "", parse_mode: str = None):
    """Wysyła zdjęcie z obsługą błędów (np. zbyt duży plik)."""
    try:
        with open(path, 'rb') as photo:
            kwargs = dict(photo=photo, caption=caption, read_timeout=60, write_timeout=60)
            if parse_mode:
                kwargs['parse_mode'] = parse_mode
            await update.message.reply_photo(**kwargs)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Nie udało się wysłać wykresu `{os.path.basename(path)}`: {e}")


# ---------------------------------------------------------------------------
# /start  /help
# ---------------------------------------------------------------------------

async def send_welcome(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🤖 *Bot NLP — Laboratorium 2*\n\n"
        "Dostępne komendy:\n"
        "`/classify dataset=<nazwa> method=<model> gridsearch=<true/false> run=<n> embedding=<typ>`\n\n"
        "*Datasety:* `20news_group`, `imdb`, `amazon`, `ag_news`\n"
        "*Modele:* `nb`, `rf`, `mlp`, `logreg`, `all`\n"
        "*Embeddingi:* `bow`, `tfidf`, `word2vec`, `glove`\n\n"
        "*Przykłady:*\n"
        "`/classify dataset=20news_group method=all gridsearch=false run=1`\n"
        "`/classify dataset=imdb method=logreg gridsearch=true run=2 embedding=tfidf`\n"
        "`/classify dataset=ag_news method=nb gridsearch=false run=3 embedding=glove`"
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')


# ---------------------------------------------------------------------------
# /classify — główna logika
# ---------------------------------------------------------------------------

async def handle_classify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text(
                "❌ Brak parametrów.\nUżyj: `/classify dataset=... method=... gridsearch=... run=... embedding=...`",
                parse_mode='Markdown'
            )
            return

        params = _parse_params(context.args)
        dataset_name = params.get('dataset')
        method = params.get('method')
        use_gridsearch = params.get('gridsearch', 'false') == 'true'
        run_count = int(params.get('run', '1'))          # liczba uruchomień (1–3)
        embedding_name = params.get('embedding', 'tfidf')

        if not dataset_name or not method:
            await update.message.reply_text("❌ Wymagane parametry: `dataset` i `method`.", parse_mode='Markdown')
            return

        if run_count < 1 or run_count > len(SEEDS):
            await update.message.reply_text(f"❌ Parametr `run` musi być od 1 do {len(SEEDS)}.", parse_mode='Markdown')
            return

        await update.message.reply_text(
            f"⏳ Rozpoczynam eksperyment...\n"
            f"• Dataset: `{dataset_name}`\n"
            f"• Embedding: `{embedding_name}`\n"
            f"• Metoda: `{method}`\n"
            f"• GridSearch: `{'tak' if use_gridsearch else 'nie'}`\n"
            f"• Liczba uruchomień: `{run_count}`\n\n"
            f"To może zająć kilka minut! ☕",
            parse_mode='Markdown'
        )

        # --- Ładowanie danych ---
        X, y, target_names = dataset_provider.get_dataset(dataset_name, sample_fraction=0.05)

        # -----------------------------------------------------------------------
        # PRZYPADEK: method=all  →  uruchamia wszystkie modele dla jednego seeda
        # -----------------------------------------------------------------------
        if method == "all":
            seeds_to_use = SEEDS[:run_count]
            all_run_results = []  # zbieramy wyniki z wszystkich run-ów

            for run_idx, seed in enumerate(seeds_to_use):
                await update.message.reply_text(
                    f"🔄 Run {run_idx + 1}/{run_count} (seed={seed}) — uruchamiam wszystkie modele..."
                )

                def make_vectorizer():
                    ep = EmbeddingProvider()
                    return ep.get_vectorizer(embedding_name, texts=X)

                run_results = classification_provider.run_all_models(
                    X=X, y=y,
                    vectorizer_factory=make_vectorizer,
                    seed=seed,
                    use_gridsearch=use_gridsearch
                )
                all_run_results.append((seed, run_results))

                # Zapisujemy wyniki każdego run-u do CSV
                for r in run_results:
                    _save_result(embedding_name, r["model_name"], r["acc"], r["f1"], seed)

            # Uśredniamy wyniki po run-ach per model
            await _send_all_models_summary(
                update, all_run_results, dataset_name, embedding_name,
                X, y, target_names, use_gridsearch
            )
            return

        # -----------------------------------------------------------------------
        # PRZYPADEK: pojedynczy model z N uruchomieniami (uśrednianie)
        # -----------------------------------------------------------------------
        seeds_to_use = SEEDS[:run_count]
        run_accuracies = []
        run_f1s = []
        last_y_test = None
        last_y_pred = None
        last_pipeline = None
        last_vectorizer = None

        for run_idx, seed in enumerate(seeds_to_use):
            await update.message.reply_text(
                f"🔄 Run {run_idx + 1}/{run_count} (seed={seed})..."
            )
            embedding_provider = EmbeddingProvider()
            vectorizer = embedding_provider.get_vectorizer(embedding_name, texts=X)

            acc, macro_f1, y_test, y_pred, pipeline = classification_provider.run_experiment(
                X=X, y=y, vectorizer=vectorizer,
                model_name=method, seed=seed, use_gridsearch=use_gridsearch
            )

            run_accuracies.append(acc)
            run_f1s.append(macro_f1)
            _save_result(embedding_name, method, acc, macro_f1, seed)

            last_y_test = y_test
            last_y_pred = y_pred
            last_pipeline = pipeline
            last_vectorizer = vectorizer

            # Zapisujemy podobne słowa jeśli używamy word2vec lub glove
            if embedding_name in ("word2vec", "glove"):
                word_model = embedding_provider.w2v_model.wv if embedding_name == "word2vec" \
                    else embedding_provider.glove_model
                if word_model is not None:
                    word_plots = visualization_provider.plot_word_embeddings(
                        word_model,
                        words_to_visualize=["space", "computer", "science", "music", "car", "god"]
                    )
                    for w_plot in word_plots:
                        await _send_photo_safe(update, w_plot, caption="Wizualizacja osadzenia słów (W2V/GloVe)")
                    await update.message.reply_text(
                        f"📄 Podobne słowa zapisane w `lab2_similar_words.txt`",
                        parse_mode='Markdown'
                    )

        mean_acc = float(np.mean(run_accuracies))
        mean_f1 = float(np.mean(run_f1s))

        # --- Wykresy (na podstawie ostatniego run-u) ---
        await update.message.reply_text("🎨 Generuję wykresy...")

        # 1. Macierz pomyłek
        cm_path = visualization_provider.plot_confusion_matrix(
            last_y_test, last_y_pred, dataset_name, method, embedding_name
        )

        # 2. Redukcja wymiarowości: PCA, t-SNE, SVD
        actual_vectorizer = (
            last_pipeline.best_estimator_.named_steps['vectorizer']
            if use_gridsearch else
            last_pipeline.named_steps['vectorizer']
        )
        X_vectors = actual_vectorizer.transform(X)

        dim_paths = []
        for dim_method in ["pca", "tsne", "svd"]:
            try:
                path = visualization_provider.plot_dimensionality_reduction(
                    X_vectors, y, target_names, dataset_name, method, embedding_name, dim_method
                )
                dim_paths.append((dim_method, path))
            except Exception as e:
                print(f"Błąd redukcji {dim_method}: {e}")

        # 3. WordCloud corpus
        wc_corpus_path = visualization_provider.plot_wordcloud(X, "corpus")

        # 4. WordCloud per klasa
        wc_class_paths = visualization_provider.plot_wordcloud_per_class(X, y, target_names)

        # 5. Feature importance
        actual_classifier = (
            last_pipeline.best_estimator_.named_steps['classifier']
            if use_gridsearch else
            last_pipeline.named_steps['classifier']
        )
        feat_path = visualization_provider.save_feature_importance(
            classifier=actual_classifier,
            vectorizer=actual_vectorizer,
            target_names=target_names,
            dataset_name=dataset_name,
            model_name=method,
            top_n=10
        )

        # --- Odpowiedź tekstowa ---
        runs_info = " | ".join([f"seed={s}: {a:.4f}" for s, a in zip(seeds_to_use, run_accuracies)])
        feat_info = f"\n📄 Feature importance → `{feat_path}`" if feat_path else ""

        response = (
            f"✅ *EKSPERYMENT ZAKOŃCZONY*\n\n"
            f"• Model: `{method.upper()}`\n"
            f"• Embedding: `{embedding_name.upper()}`\n"
            f"• GridSearch: `{'Tak' if use_gridsearch else 'Nie'}`\n"
            f"• Uruchomienia: `{run_count}` ({runs_info})\n\n"
            f"📊 *Uśrednione wyniki:*\n"
            f"• Accuracy: `{mean_acc:.4f}`\n"
            f"• Macro F1: `{mean_f1:.4f}`\n\n"
            f"💾 Wyniki zapisano w `{CSV_FILE}`"
            f"{feat_info}"
        )

        # Wysyłamy macierz pomyłek z głównym raportem
        await _send_photo_safe(update, cm_path, caption=response, parse_mode='Markdown')

        # Wykresy redukcji wymiarowości
        for dim_method, path in dim_paths:
            await _send_photo_safe(update, path, caption=f"Redukcja wymiarowości: {dim_method.upper()}")

        # WordCloud corpus
        await _send_photo_safe(update, wc_corpus_path, caption="☁️ Word Cloud — cały korpus")

        # WordCloud per klasa (tylko kilka pierwszych, by nie zasypać chata)
        if wc_class_paths:
            await update.message.reply_text(
                f"☁️ Generuję Word Cloud dla {len(wc_class_paths)} klas — wysyłam pierwsze 3..."
            )
            for path in wc_class_paths[:3]:
                class_name = os.path.basename(path).replace("wordcloud_class_", "").replace(".png", "")
                await _send_photo_safe(update, path, caption=f"☁️ Word Cloud — klasa: {class_name}")
            if len(wc_class_paths) > 3:
                await update.message.reply_text(
                    f"ℹ️ Pozostałe {len(wc_class_paths) - 3} wordclouds per klasa zapisano w `{PLOTS_DIR}/`",
                    parse_mode='Markdown'
                )

    except NotImplementedError as e:
        await update.message.reply_text(f"❌ Błąd implementacji: {e}")
    except ValueError as e:
        await update.message.reply_text(f"❌ Błąd parametrów: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        await update.message.reply_text(f"❌ Nieoczekiwany błąd: {e}")


# ---------------------------------------------------------------------------
# Pomocnik dla method=all
# ---------------------------------------------------------------------------

async def _send_all_models_summary(
    update, all_run_results, dataset_name, embedding_name,
    X, y, target_names, use_gridsearch
):
    """Wysyła podsumowanie eksperymentu dla method=all."""
    # Zbieramy uśrednione wyniki per model
    model_scores: dict[str, list] = {}
    for seed, run_results in all_run_results:
        for r in run_results:
            model_scores.setdefault(r["model_name"], {"acc": [], "f1": []})
            model_scores[r["model_name"]]["acc"].append(r["acc"])
            model_scores[r["model_name"]]["f1"].append(r["f1"])

    lines = [f"✅ *EKSPERYMENT ZAKOŃCZONY — method=all*\n",
             f"• Dataset: `{dataset_name}`\n• Embedding: `{embedding_name.upper()}`\n\n",
             "📊 *Uśrednione wyniki (po seedach):*\n"]
    for model_name, scores in model_scores.items():
        mean_acc = np.mean(scores["acc"])
        mean_f1 = np.mean(scores["f1"])
        lines.append(f"• `{model_name.upper()}` — Acc: `{mean_acc:.4f}` | F1: `{mean_f1:.4f}`\n")

    await update.message.reply_text("".join(lines), parse_mode='Markdown')

    # Wysyłamy macierze pomyłek z ostatniego run-u dla każdego modelu
    if all_run_results:
        last_seed, last_run_results = all_run_results[-1]
        for r in last_run_results:
            cm_path = visualization_provider.plot_confusion_matrix(
                r["y_test"], r["y_pred"], dataset_name, r["model_name"], embedding_name
            )
            await _send_photo_safe(update, cm_path, caption=f"Macierz pomyłek: {r['model_name'].upper()}")

        # WordCloud corpus — raz
        wc_path = visualization_provider.plot_wordcloud(X, "corpus")
        await _send_photo_safe(update, wc_path, caption="☁️ Word Cloud — korpus")

        # WordCloud per klasa
        wc_class_paths = visualization_provider.plot_wordcloud_per_class(X, y, target_names)
        for path in wc_class_paths[:3]:
            class_name = os.path.basename(path).replace("wordcloud_class_", "").replace(".png", "")
            await _send_photo_safe(update, path, caption=f"☁️ Word Cloud — klasa: {class_name}")


# ---------------------------------------------------------------------------
# Uruchomienie bota
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Uruchamianie bota NLP (Lab 2)...")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", send_welcome))
    app.add_handler(CommandHandler("help", send_welcome))
    app.add_handler(CommandHandler("classify", handle_classify))
    app.run_polling()