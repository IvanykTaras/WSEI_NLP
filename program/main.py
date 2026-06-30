import asyncio
import logging
import os
import shlex
import numpy as np
import pandas as pd
import requests
from typing import Sequence
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from providers.dataset_provider import DatasetProvider, SENTIMENT_DATASETS, SUPPORTED_DATASETS
from providers.embedding_provider import EmbeddingProvider, SUPPORTED_EMBEDDINGS
from providers.classification_provider import ClassificationProvider, ALL_MODELS
from providers.visualization_provider import VisualizationProvider
from providers.lab3_visualization_provider import Lab3VisualizationProvider
from providers.sentiment_provider import (
    LAB3_RESULTS_FILE,
    SENTIMENT_SAMPLE_FRACTIONS,
    SUPPORTED_SENTIMENT_METHODS,
    VALID_SENTIMENT_LABELS,
    SentimentProvider,
)
from providers.sequence_model_provider import SEQUENCE_MODELS, SequenceModelProvider
from providers.entity_provider import EntityProvider, SUPPORTED_NER_METHODS
from providers.lab4_artifact_provider import Lab4ArtifactProvider
from providers.summarization_provider import (
    SUPPORTED_SUMMARY_LENGTHS,
    SUPPORTED_SUMMARY_TYPES,
    SummarizationProvider,
)
from providers.translation_provider import (
    SUPPORTED_TRANSLATION_LANGUAGES,
    TranslationProvider,
)
from providers.lab5_tools_provider import Lab5ToolsProvider
from providers.tool_calling_provider import ToolCallingProvider
from providers.moderation_provider import ModerationProvider, VALID_ACTIONS

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOT_TOKEN = os.getenv("BOT_TOKEN")

dataset_provider = DatasetProvider()
classification_provider = ClassificationProvider()
visualization_provider = VisualizationProvider()
sequence_provider = SequenceModelProvider()
sentiment_provider = SentimentProvider(dataset_provider=dataset_provider, sequence_provider=sequence_provider)
lab3_visualization_provider = Lab3VisualizationProvider()
entity_provider = EntityProvider()
translation_provider = TranslationProvider()
summarization_provider = SummarizationProvider()
lab4_artifact_provider = Lab4ArtifactProvider()
lab5_tools_provider = Lab5ToolsProvider()
tool_calling_provider = ToolCallingProvider(tools_provider=lab5_tools_provider)
moderation_provider = ModerationProvider(
    sentiment_provider=sentiment_provider,
    entity_provider=entity_provider,
)

# Seedy używane dla kolejnych uruchomień (run=1 → seed[0], run=2 → seed[0]+seed[1], itd.)
SEEDS = [42, 1337, 2024]
DEFAULT_SAMPLE_FRACTION = 0.05

CSV_FILE = os.path.join(BASE_DIR, "lab2results.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "lab2plots")
LAB3_PLOTS_DIR = os.path.join(BASE_DIR, "lab3plots")


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


def _parse_params_preserve_case(args: list) -> dict:
    try:
        parts = shlex.split(" ".join(args))
    except ValueError as exc:
        raise ValueError("Niepoprawne cudzyslowy w komendzie.") from exc

    params = {}
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            params[key.lower()] = value
    return params


def _parse_two_quoted_args(args: list) -> tuple[str, str]:
    try:
        parts = shlex.split(" ".join(args))
    except ValueError as exc:
        raise ValueError("Niepoprawne cudzyslowy w komendzie.") from exc
    if len(parts) != 2:
        raise ValueError('Uzyj formatu: /add_sentiment "tekst" "etykieta".')
    return parts[0], parts[1]


def _parse_quoted_args(args: list, expected: int, usage: str) -> list:
    try:
        parts = shlex.split(" ".join(args))
    except ValueError as exc:
        raise ValueError("Niepoprawne cudzysłowy w komendzie.") from exc
    if len(parts) != expected:
        raise ValueError(f"Użyj: {usage}")
    return parts


def _format_supported(values: Sequence) -> str:
    return ", ".join(f"`{value}`" for value in values)


def _validate_params(params: dict) -> tuple[str, str, bool, int, str]:
    dataset_name = params.get('dataset')
    method = params.get('method')
    gridsearch_value = params.get('gridsearch', 'false')
    run_value = params.get('run', '1')
    embedding_name = params.get('embedding', 'tfidf')

    dataset_name = dataset_name.lower() if dataset_name else dataset_name
    method = method.lower() if method else method
    gridsearch_value = gridsearch_value.lower()
    embedding_name = embedding_name.lower()

    if not dataset_name or not method:
        raise ValueError("Wymagane parametry: `dataset` i `method`.")
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Nieznany dataset `{dataset_name}`. Dostępne: {_format_supported(SUPPORTED_DATASETS)}."
        )
    if method not in ALL_MODELS + ["all"]:
        raise ValueError(
            f"Nieznany model `{method}`. Dostępne: {_format_supported(ALL_MODELS + ['all'])}."
        )
    if embedding_name not in SUPPORTED_EMBEDDINGS:
        raise ValueError(
            f"Nieznany embedding `{embedding_name}`. Dostępne: {_format_supported(SUPPORTED_EMBEDDINGS)}."
        )
    if gridsearch_value not in ("true", "false"):
        raise ValueError("Parametr `gridsearch` musi mieć wartość `true` albo `false`.")

    try:
        run_count = int(run_value)
    except ValueError as exc:
        raise ValueError("Parametr `run` musi być liczbą: `1`, `2` albo `3`.") from exc

    if run_count < 1 or run_count > len(SEEDS):
        raise ValueError(f"Parametr `run` musi być od 1 do {len(SEEDS)}.")

    if method != "all":
        classification_provider.validate_model_embedding(method, embedding_name)

    return dataset_name, method, gridsearch_value == "true", run_count, embedding_name


def _save_result(embedding: str, model: str, accuracy: float, macro_f1: float, seed: int):
    row = pd.DataFrame([{
        "embedding": embedding,
        "model": model,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "seed": seed,
    }])
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
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


async def _reply_long_text(update: Update, text: str, chunk_size: int = 3900):
    """Wysyła długą odpowiedź w bezpiecznych fragmentach dla Telegrama."""
    remaining = text.strip()
    while remaining:
        if len(remaining) <= chunk_size:
            chunk, remaining = remaining, ""
        else:
            split_at = remaining.rfind("\n", 0, chunk_size)
            if split_at < chunk_size // 2:
                split_at = remaining.rfind(" ", 0, chunk_size)
            if split_at < chunk_size // 2:
                split_at = chunk_size
            chunk, remaining = remaining[:split_at], remaining[split_at:].lstrip()
        await update.message.reply_text(chunk)


def _artifact_relative_path(path: str) -> str:
    return os.path.relpath(path, BASE_DIR)


# ---------------------------------------------------------------------------
# /start  /help
# ---------------------------------------------------------------------------

async def send_welcome(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🤖 *Bot NLP — Laboratorium 2 + 3 + 4 + 5 + 6*\n\n"
        "Dostępne komendy:\n"
        "`/classify dataset=<nazwa> method=<model> gridsearch=<true/false> run=<n> embedding=<typ>`\n"
        "`/sentiment method=<metoda> text=\"tekst\"`\n"
        "`/train model=<simplernn|lstm|gru> dataset=<amazon|imdb|custom>`\n"
        "`/compare dataset=<amazon|imdb|custom> methods=<lista>`\n"
        "`/add_sentiment \"tekst\" \"etykieta\"`\n"
        "`/models`\n\n"
        "*Komendy Lab4:*\n"
        "`/ner method=<spacy|stanza> text=\"tekst\"`\n"
        "`/nel text=\"encja\" language=<pl|en>`\n"
        "`/ned entity=\"encja\" context=\"kontekst\"`\n"
        "`/translate text=\"tekst\" target_lang=<en|pl|de|fr|es>`\n"
        "`/summarize text=\"tekst\" summary_type=<typ> length=<długość>`\n"
        "`/analyze_entities text=\"tekst\" link=<true|false>`\n"
        "`/language_detect text=\"tekst\"`\n\n"
        "*Komenda Lab5 — automatyczny wybór narzędzi:*\n"
        "`/agent <pytanie>`\n"
        "Możesz też wysłać zdjęcie z podpisem `/agent <pytanie>`.\n\n"
        "*Komendy Lab6 — moderacja treści:*\n"
        "`/moderate \"tekst\"`\n"
        "`/mod_policy_check \"tekst\"`\n"
        "`/mod_status <content_id>`  `/mod_history <user_id>`\n"
        "`/mod_analytics`  `/mod_watchlist`\n"
        "`/mod_add_feedback <content_id> \"komentarz\" \"decyzja\"`\n"
        "`/mod_train_on_feedback`  `/mod_help`\n\n"
        "*Datasety Lab2:* `20news_group`, `imdb`, `amazon`, `ag_news`\n"
        "*Datasety Lab3:* `amazon`, `imdb`, `custom`\n"
        "*Modele:* `nb`, `rf`, `mlp`, `logreg`, `all`\n"
        "*Embeddingi:* `bow`, `tfidf`, `word2vec`, `glove`\n\n"
        "*Przykłady:*\n"
        "`/classify dataset=20news_group method=all gridsearch=false run=1`\n"
        "`/sentiment method=rule text=\"To byl swietny film\"`\n"
        "`/train model=simplernn dataset=custom`\n"
        "`/compare dataset=custom methods=rule,nb,rf,textblob`\n"
        "`/ner method=spacy text=\"Steve Jobs założył Apple\"`\n"
        "`/translate text=\"Good morning\" target_lang=pl`\n"
        "`/agent Porównaj pogodę w Warszawie i Paryżu`"
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
        dataset_name, method, use_gridsearch, run_count, embedding_name = _validate_params(params)

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
        X, y, target_names = dataset_provider.get_dataset(
            dataset_name,
            sample_fraction=DEFAULT_SAMPLE_FRACTION,
            seed=SEEDS[0]
        )

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
                    use_gridsearch=use_gridsearch,
                    embedding_name=embedding_name
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

        for run_idx, seed in enumerate(seeds_to_use):
            await update.message.reply_text(
                f"🔄 Run {run_idx + 1}/{run_count} (seed={seed})..."
            )
            embedding_provider = EmbeddingProvider()
            vectorizer = embedding_provider.get_vectorizer(embedding_name, texts=X)

            acc, macro_f1, y_test, y_pred, pipeline = classification_provider.run_experiment(
                X=X, y=y, vectorizer=vectorizer,
                model_name=method, seed=seed, use_gridsearch=use_gridsearch,
                embedding_name=embedding_name
            )

            run_accuracies.append(acc)
            run_f1s.append(macro_f1)
            _save_result(embedding_name, method, acc, macro_f1, seed)

            last_y_test = y_test
            last_y_pred = y_pred
            last_pipeline = pipeline
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
        feat_info = f"\n📄 Feature importance → `lab2plots/{os.path.basename(feat_path)}`" if feat_path else ""

        response = (
            f"✅ *EKSPERYMENT ZAKOŃCZONY*\n\n"
            f"• Model: `{method.upper()}`\n"
            f"• Embedding: `{embedding_name.upper()}`\n"
            f"• GridSearch: `{'Tak' if use_gridsearch else 'Nie'}`\n"
            f"• Uruchomienia: `{run_count}` ({runs_info})\n\n"
            f"📊 *Uśrednione wyniki:*\n"
            f"• Accuracy: `{mean_acc:.4f}`\n"
            f"• Macro F1: `{mean_f1:.4f}`\n\n"
            f"💾 Wyniki zapisano w `{os.path.basename(CSV_FILE)}`"
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
                    f"ℹ️ Pozostałe {len(wc_class_paths) - 3} wordclouds per klasa zapisano w `lab2plots/`",
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

    compatible_models = set(classification_provider.get_compatible_models(embedding_name))
    skipped_models = set(ALL_MODELS) - compatible_models
    if skipped_models:
        skipped = ", ".join(sorted(skipped_models))
        lines.append(f"\nℹ️ Pominięto modele niezgodne z embeddingiem `{embedding_name}`: `{skipped}`.\n")

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
# Lab 3: sentyment, trening modeli sekwencyjnych, porównania
# ---------------------------------------------------------------------------

async def handle_sentiment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        params = _parse_params_preserve_case(context.args)
        method = params.get("method", "").lower()
        text = params.get("text", "")
        dataset_name = params.get("dataset", "custom").lower()

        if not method or not text:
            await update.message.reply_text(
                '❌ Uzyj: `/sentiment method=<metoda> text="tekst"`',
                parse_mode='Markdown'
            )
            return
        if dataset_name not in SENTIMENT_DATASETS:
            raise ValueError(f"Dataset dla Lab3 musi byc jednym z: {_format_supported(SENTIMENT_DATASETS)}.")

        if method == "stanza":
            await update.message.reply_text(
                "⏳ Ładuję angielski model Stanza. Pierwsza analiza może potrwać kilkadziesiąt sekund."
            )
        result = await asyncio.to_thread(
            sentiment_provider.predict,
            method,
            text,
            dataset_name,
        )
        score = result.get("score")
        score_line = f"\n• Pewnosc/score: `{score:.4f}`" if isinstance(score, (int, float)) else ""
        await update.message.reply_text(
            f"✅ *Analiza sentymentu*\n\n"
            f"• Metoda: `{result.get('method', method)}`\n"
            f"• Dataset/model: `{dataset_name}`\n"
            f"• Predykcja: `{result['label']}`"
            f"{score_line}\n"
            f"• Model: `{result.get('model_path', '-')}`",
            parse_mode='Markdown'
        )
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as e:
        await update.message.reply_text(f"❌ {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        await update.message.reply_text(f"❌ Nieoczekiwany błąd: {e}")


async def handle_train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        params = _parse_params_preserve_case(context.args)
        model_name = params.get("model", "").lower()
        dataset_name = params.get("dataset", "").lower()

        if model_name not in SEQUENCE_MODELS:
            raise ValueError(f"Model musi byc jednym z: {_format_supported(SEQUENCE_MODELS)}.")
        if dataset_name not in SENTIMENT_DATASETS:
            raise ValueError(f"Dataset musi byc jednym z: {_format_supported(SENTIMENT_DATASETS)}.")

        await update.message.reply_text(
            f"⏳ Rozpoczynam trening `{model_name}` na `{dataset_name}`. To moze potrwac kilka minut.",
            parse_mode='Markdown'
        )

        sample_fraction = SENTIMENT_SAMPLE_FRACTIONS[dataset_name]
        X, y, _ = await asyncio.to_thread(
            sentiment_provider.load_sentiment_dataset,
            dataset_name,
            sample_fraction,
            SEEDS[0],
        )
        await update.message.reply_text(
            f"📚 Załadowano `{len(X)}` przykładów. Rozpoczynam trening...",
            parse_mode='Markdown',
        )
        result = await asyncio.to_thread(
            sequence_provider.train,
            model_name,
            dataset_name,
            X,
            y,
        )
        history_path = lab3_visualization_provider.plot_training_history(
            result.history, model_name, dataset_name
        )
        lab3_visualization_provider.plot_wordcloud_per_label(X, y, dataset_name)
        lab3_visualization_provider.plot_class_distribution(y, dataset_name)

        response = (
            f"✅ *Trening zakonczony*\n\n"
            f"• Model: `{model_name}`\n"
            f"• Dataset: `{dataset_name}`\n"
            f"• Accuracy testowe: `{result.test_accuracy:.4f}`\n"
            f"• Epoki: `{result.epochs_run}`\n"
            f"• Czas: `{result.duration_seconds:.1f}s`\n\n"
            f"💾 Model: `{os.path.relpath(result.model_path, BASE_DIR)}`\n"
            f"💾 Tokenizer: `{os.path.relpath(result.tokenizer_path, BASE_DIR)}`\n"
            f"💾 Encoder: `{os.path.relpath(result.label_encoder_path, BASE_DIR)}`"
        )
        await _send_photo_safe(update, history_path, caption=response, parse_mode='Markdown')
    except (ValueError, ImportError) as e:
        await update.message.reply_text(f"❌ {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        await update.message.reply_text(f"❌ Nieoczekiwany błąd: {e}")


async def handle_compare(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        params = _parse_params_preserve_case(context.args)
        dataset_name = params.get("dataset", "").lower()
        methods_value = params.get("methods", "")
        if dataset_name not in SENTIMENT_DATASETS:
            raise ValueError(f"Dataset musi byc jednym z: {_format_supported(SENTIMENT_DATASETS)}.")
        if not methods_value:
            raise ValueError("Podaj methods, np. methods=rule,nb,rf,textblob.")

        methods = [method.strip().lower() for method in methods_value.split(",")]
        await update.message.reply_text(
            f"⏳ Porownuje metody `{', '.join(methods)}` na `{dataset_name}`...",
            parse_mode='Markdown'
        )

        sample_fraction = SENTIMENT_SAMPLE_FRACTIONS[dataset_name]
        comparison = sentiment_provider.compare(
            dataset_name=dataset_name,
            methods=methods,
            sample_fraction=sample_fraction,
            seed=SEEDS[0],
        )
        results = comparison["results"]
        compare_path = lab3_visualization_provider.plot_compare_methods(results, dataset_name)
        for method, y_pred in comparison["predictions"].items():
            lab3_visualization_provider.plot_confusion_matrix(
                comparison["y_test"], y_pred, method, dataset_name
            )
        lab3_visualization_provider.plot_wordcloud_per_label(comparison["X"], comparison["y"], dataset_name)

        lines = [
            "✅ *Porownanie zakonczone*\n\n",
            f"💾 Wyniki zapisano w `{os.path.relpath(LAB3_RESULTS_FILE, BASE_DIR)}`\n\n",
            "📊 *Macro F1:*\n",
        ]
        for row in results:
            lines.append(
                f"• `{row['method']}` — Acc `{row['accuracy']:.4f}`, F1 `{row['macro_f1']:.4f}`\n"
            )

        if compare_path:
            await _send_photo_safe(update, compare_path, caption="".join(lines), parse_mode='Markdown')
        else:
            await update.message.reply_text("".join(lines), parse_mode='Markdown')
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as e:
        await update.message.reply_text(f"❌ {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        await update.message.reply_text(f"❌ Nieoczekiwany błąd: {e}")


async def handle_add_sentiment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text, label = _parse_two_quoted_args(context.args)
        total = sentiment_provider.add_custom_example(text, label)
        await update.message.reply_text(
            f"✅ Dodano rekord do `sentiment_dataset.csv`.\n"
            f"• Etykieta: `{label.lower()}`\n"
            f"• Liczba rekordow: `{total}`",
            parse_mode='Markdown'
        )
    except ValueError as e:
        await update.message.reply_text(f"❌ {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        await update.message.reply_text(f"❌ Nieoczekiwany błąd: {e}")


async def handle_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = sequence_provider.list_models()
    if not rows:
        await update.message.reply_text(
            "ℹ️ Brak zapisanych modeli w `models/`. Uzyj np. `/train model=simplernn dataset=custom`.",
            parse_mode='Markdown'
        )
        return

    lines = ["📦 *Zapisane modele Lab3*\n\n"]
    for row in rows:
        tokenizer = "tak" if row["has_tokenizer"] else "nie"
        encoder = "tak" if row["has_label_encoder"] else "nie"
        rel_path = os.path.relpath(row["model_path"], BASE_DIR)
        lines.append(
            f"• `{row['model']}` / `{row['dataset']}` — model `{rel_path}`, tokenizer `{tokenizer}`, encoder `{encoder}`\n"
        )
    await update.message.reply_text("".join(lines), parse_mode='Markdown')


# ---------------------------------------------------------------------------
# Lab 4: NER, NEL/NED, tłumaczenie, detekcja języka i podsumowanie
# ---------------------------------------------------------------------------

def _format_entity_rows(entities: list) -> str:
    if not entities:
        return "Nie znaleziono encji."
    lines = []
    for entity in entities:
        lines.append(
            f"- {entity['text']} ({entity['type']}) [{entity['start']}:{entity['end']}]"
        )
        if "linking" in entity:
            for candidate in entity["linking"][:2]:
                lines.append(
                    f"  • {candidate['label']} ({candidate['id']}), "
                    f"confidence={candidate['confidence']:.2f}"
                )
            if entity.get("linking_error"):
                lines.append(f"  • Linkowanie: {entity['linking_error']}")
    return "\n".join(lines)


def _format_candidate_rows(candidates: list, score_key: str = "confidence") -> str:
    if not candidates:
        return "Brak kandydatów."
    lines = []
    for index, candidate in enumerate(candidates, start=1):
        score = candidate.get(score_key, candidate.get("confidence", 0.0))
        lines.append(
            f"{index}. {candidate['label']} ({candidate['id']})\n"
            f"   {candidate.get('description') or 'Brak opisu'}\n"
            f"   Wikidata: {candidate.get('wikidata_url') or '-'}\n"
            f"   Wikipedia: {candidate.get('wikipedia_url') or '-'}\n"
            f"   Confidence: {score:.4f}"
        )
    return "\n".join(lines)


async def handle_ner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        params = _parse_params_preserve_case(context.args)
        method = params.get("method", "").lower()
        text = params.get("text", "")
        if not method or not text:
            raise ValueError('Użyj: /ner method=<spacy|stanza> text="tekst".')
        if method not in SUPPORTED_NER_METHODS:
            raise ValueError(f"Metoda musi być jedną z: {', '.join(SUPPORTED_NER_METHODS)}.")
        await update.message.reply_text(f"⏳ Analizuję encje metodą {method}...")
        result = await asyncio.to_thread(entity_provider.recognize, method, text)
        path = lab4_artifact_provider.save_json("ner", result)
        response = (
            f"✅ NER — {method}\n\n"
            f"{_format_entity_rows(result['entities'])}\n\n"
            f"Wynik zapisano: {_artifact_relative_path(path)}"
        )
        await _reply_long_text(update, response)
    except (ValueError, ImportError, FileNotFoundError, RuntimeError) as exc:
        await update.message.reply_text(f"❌ {exc}")


async def handle_nel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        params = _parse_params_preserve_case(context.args)
        text = params.get("text", "")
        language = params.get("language", "pl").lower()
        if not text:
            raise ValueError('Użyj: /nel text="encja" language=<pl|en>.')
        await update.message.reply_text("⏳ Szukam kandydatów w Wikidata i lokalnej bazie...")
        result = await asyncio.to_thread(entity_provider.link_entity, text, language, 5)
        path = lab4_artifact_provider.save_json("nel", result)
        response = (
            f"✅ NEL\nEncja: {result['entity']}\nŹródło: {result['source']}\n\n"
            f"{_format_candidate_rows(result['candidates'])}\n\n"
            f"Wynik zapisano: {_artifact_relative_path(path)}"
        )
        await _reply_long_text(update, response)
    except (ValueError, RuntimeError, requests.RequestException) as exc:
        await update.message.reply_text(f"❌ {exc}")


async def handle_ned(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        params = _parse_params_preserve_case(context.args)
        entity = params.get("entity", "")
        entity_context = params.get("context", "")
        language = params.get("language", "pl").lower()
        if not entity or not entity_context:
            raise ValueError('Użyj: /ned entity="encja" context="tekst".')
        await update.message.reply_text("⏳ Oceniam kandydatów w kontekście...")
        result = await asyncio.to_thread(
            entity_provider.disambiguate, entity, entity_context, language
        )
        path = lab4_artifact_provider.save_json("ned", result)
        selected = result["selected"]
        if selected:
            selected_text = (
                f"Wybrano: {selected['label']} ({selected['id']})\n"
                f"Score: {selected['ned_score']:.4f}\n"
                f"Opis: {selected.get('description') or '-'}\n"
                f"Wikipedia: {selected.get('wikipedia_url') or '-'}"
            )
        else:
            selected_text = "Nie wybrano kandydata — pewność była zbyt niska."
        response = (
            f"✅ NED\n{selected_text}\n\nRanking:\n"
            f"{_format_candidate_rows(result['candidates'], 'ned_score')}\n\n"
            f"Wynik zapisano: {_artifact_relative_path(path)}"
        )
        await _reply_long_text(update, response)
    except (ValueError, RuntimeError, requests.RequestException) as exc:
        await update.message.reply_text(f"❌ {exc}")


async def handle_analyze_entities(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        params = _parse_params_preserve_case(context.args)
        text = params.get("text", "")
        link_value = params.get("link", "false").lower()
        method = params.get("method", "spacy").lower()
        if not text:
            raise ValueError('Użyj: /analyze_entities text="tekst" link=<true|false>.')
        if link_value not in ("true", "false"):
            raise ValueError("Parametr link musi mieć wartość true albo false.")
        link = link_value == "true"
        await update.message.reply_text(
            "⏳ Rozpoznaję i linkuję encje..." if link else "⏳ Rozpoznaję encje..."
        )
        result = await asyncio.to_thread(entity_provider.analyze_entities, text, link, method)
        path = lab4_artifact_provider.save_json("analyze_entities", result)
        response = (
            f"✅ Analiza encji — {method}, link={str(link).lower()}\n\n"
            f"{_format_entity_rows(result['entities'])}\n\n"
            f"Wynik zapisano: {_artifact_relative_path(path)}"
        )
        await _reply_long_text(update, response)
    except (ValueError, ImportError, FileNotFoundError, RuntimeError) as exc:
        await update.message.reply_text(f"❌ {exc}")


async def handle_language_detect(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        params = _parse_params_preserve_case(context.args)
        text = params.get("text", "")
        if not text:
            raise ValueError('Użyj: /language_detect text="tekst".')
        result = await asyncio.to_thread(translation_provider.detect_language, text)
        path = lab4_artifact_provider.save_json("language_detect", result)
        alternatives = ", ".join(
            f"{row['language']}={row['confidence']:.4f}" for row in result["alternatives"]
        )
        await update.message.reply_text(
            f"✅ Wykryty język: {result['language']}\n"
            f"Confidence: {result['confidence']:.4f}\n"
            f"Kandydaci: {alternatives}\n"
            f"Wynik zapisano: {_artifact_relative_path(path)}"
        )
    except (ValueError, ImportError) as exc:
        await update.message.reply_text(f"❌ {exc}")


async def handle_translate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        params = _parse_params_preserve_case(context.args)
        text = params.get("text", "")
        target_language = params.get("target_lang", "").lower()
        if not text or not target_language:
            raise ValueError(
                '/translate wymaga text="tekst" i target_lang=<en|pl|de|fr|es>.'
            )
        if target_language not in SUPPORTED_TRANSLATION_LANGUAGES:
            raise ValueError(
                f"Język docelowy musi być jednym z: {', '.join(SUPPORTED_TRANSLATION_LANGUAGES)}."
            )
        await update.message.reply_text(
            "⏳ Wykrywam język i tłumaczę lokalnym modelem M2M100. Pierwsze użycie może potrwać chwilę..."
        )
        result = await asyncio.to_thread(
            translation_provider.translate, text, target_language
        )
        path = lab4_artifact_provider.save_json("translate", result)
        response = (
            f"✅ Tłumaczenie ({result['source_language']} → {result['target_language']})\n"
            f"Model: {result['model']}\n\n{result['translation']}\n\n"
            f"Wynik zapisano: {_artifact_relative_path(path)}"
        )
        await _reply_long_text(update, response)
    except (ValueError, ImportError, FileNotFoundError, RuntimeError) as exc:
        await update.message.reply_text(f"❌ {exc}")


async def handle_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        params = _parse_params_preserve_case(context.args)
        text = params.get("text", "")
        summary_type = params.get("summary_type", "abstractive").lower()
        length = params.get("length", "medium").lower()
        custom_prompt = params.get("prompt", "")
        if not text:
            raise ValueError(
                '/summarize wymaga text="tekst"; opcjonalnie summary_type i length.'
            )
        if summary_type not in SUPPORTED_SUMMARY_TYPES:
            raise ValueError(
                f"summary_type musi być jednym z: {', '.join(SUPPORTED_SUMMARY_TYPES)}."
            )
        if length not in SUPPORTED_SUMMARY_LENGTHS:
            raise ValueError(
                f"length musi być jednym z: {', '.join(SUPPORTED_SUMMARY_LENGTHS)}."
            )
        await update.message.reply_text(
            f"⏳ Generuję podsumowanie {summary_type}/{length} przez Ollama..."
        )
        result = await asyncio.to_thread(
            summarization_provider.summarize,
            text,
            summary_type,
            length,
            custom_prompt,
        )
        path = lab4_artifact_provider.save_text(
            "summarize",
            result["summary"],
            {
                "model": result["model"],
                "summary_type": result["summary_type"],
                "length": result["length"],
                "generation_seconds": result["generation_seconds"],
            },
        )
        response = (
            f"✅ Podsumowanie — {result['summary_type']}/{result['length']}\n"
            f"Model: {result['model']}\nCzas: {result['generation_seconds']:.2f}s\n\n"
            f"{result['summary']}\n\nWynik zapisano: {_artifact_relative_path(path)}"
        )
        await _reply_long_text(update, response)
    except (ValueError, FileNotFoundError, RuntimeError, TimeoutError) as exc:
        await update.message.reply_text(f"❌ {exc}")


# ---------------------------------------------------------------------------
# Lab 5: agent Ollama z automatycznym Tool Calling
# ---------------------------------------------------------------------------

async def _run_agent_and_reply(update: Update, prompt: str, image_path: str = None):
    try:
        if not prompt and not image_path:
            raise ValueError(
                "Użyj `/agent <pytanie>` albo wyślij zdjęcie z podpisem `/agent <pytanie>`."
            )
        await update.message.reply_text(
            "⏳ Agent analizuje pytanie i dobiera potrzebne narzędzia..."
        )
        result = await asyncio.to_thread(tool_calling_provider.run, prompt, image_path)
        used_tools = [event["name"] for event in result.tool_calls]
        tools_line = ", ".join(used_tools) if used_tools else "brak"
        response = (
            f"🤖 Odpowiedź agenta\n\n{result.answer}\n\n"
            f"Narzędzia: {tools_line}\n"
            f"Rundy: {result.rounds}, czas: {result.duration_seconds:.2f}s\n"
            f"Historia: {_artifact_relative_path(result.history_path)}"
        )
        await _reply_long_text(update, response)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        await update.message.reply_text(f"❌ {exc}")


async def handle_agent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args).strip()
    await _run_agent_and_reply(update, prompt)


async def handle_agent_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caption = (update.message.caption or "").strip()
    command_and_prompt = caption.split(maxsplit=1)
    prompt = command_and_prompt[1].strip() if len(command_and_prompt) == 2 else ""
    photo = update.message.photo[-1]
    os.makedirs(lab5_tools_provider.uploads_dir, exist_ok=True)
    image_path = os.path.join(
        lab5_tools_provider.uploads_dir, f"{photo.file_unique_id}.jpg"
    )
    try:
        telegram_file = await context.bot.get_file(photo.file_id)
        await telegram_file.download_to_drive(custom_path=image_path)
    except Exception as exc:
        await update.message.reply_text(f"❌ Nie udało się pobrać zdjęcia: {exc}")
        return
    await _run_agent_and_reply(update, prompt, image_path)


# ---------------------------------------------------------------------------
# Lab 6: lokalna moderacja treści i feedback loop
# ---------------------------------------------------------------------------

def _format_moderation_result(result: dict, policy_check: bool = False) -> str:
    icon = {"APPROVE": "✅", "REJECT": "❌", "FLAG_FOR_REVIEW": "⏳"}[result["action"]]
    bielik = result["bielik"]
    qwen = result["qwen"]
    pii = result["pii"]
    sentiment = result["sentiment"]
    entities = result["entities"]
    entity_parts = []
    for key in (
        "usernames_mentioned", "urls", "emails", "phone_numbers",
        "organizations", "locations", "persons",
    ):
        if entities.get(key):
            entity_parts.append(f"{key}={', '.join(entities[key])}")
    context_parts = [
        f"{row['category']}:{','.join(row['mentions'])}"
        for row in entities.get("contextual_targets", [])
    ]
    similar = result.get("similar_cases", [])
    similar_line = ", ".join(
        f"#{row['content_id']} ({row['similarity']:.2f})" for row in similar[:3]
    ) or "brak"
    tools_line = ", ".join(
        item.split()[0] for item in result.get("executed_tools", [])
    ) or "brak (policy check)"
    prefix = "POLICY CHECK (bez zapisu)" if policy_check else f"MODERACJA #{result['content_id']}"
    return (
        f"{icon} {prefix}\n"
        f"Decyzja: {result['action']}\n"
        f"Użytkownik: {result['user_id']}\n"
        f"Powód: {result['reason']}\n"
        f"Consensus: {result['consensus']}\n"
        f"PII: {'tak' if pii['has_pii'] else 'nie'} ({pii['source']})\n"
        f"Bielik: {bielik['label']} ({bielik['score']:.3f})\n"
        f"Qwen: {qwen['risk_level']} ({qwen['confidence']:.3f})\n"
        f"Sentyment: {sentiment['sentiment']}, emocja: {sentiment['emotion']}\n"
        f"Encje: {'; '.join(entity_parts) or 'brak'}\n"
        f"Cele kontekstowe: {'; '.join(context_parts) or 'brak'}\n"
        f"Podobne przypadki: {similar_line}\n"
        f"Głosy: {', '.join(f'{name}={vote}' for name, vote in result['votes'].items())}\n"
        f"Wykonane tools: {tools_line}\n"
        f"Czas: {result['duration_seconds']:.2f}s"
    )


async def handle_moderate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = _parse_quoted_args(context.args, 1, '/moderate "tekst"')[0]
        await update.message.reply_text("⏳ Uruchamiam lokalne modele moderacji...")
        user = getattr(update, "effective_user", None)
        message_id = getattr(update.message, "message_id", None) or int(asyncio.get_running_loop().time() * 1000)
        result = await asyncio.to_thread(
            moderation_provider.moderate,
            text,
            str(message_id),
            str(getattr(user, "id", "unknown")),
            getattr(user, "username", "") or "",
            True,
        )
        await update.message.reply_text(_format_moderation_result(result))
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        await update.message.reply_text(f"❌ {exc}")


async def handle_mod_policy_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = _parse_quoted_args(context.args, 1, '/mod_policy_check "tekst"')[0]
        await update.message.reply_text("⏳ Sprawdzam politykę bez zapisywania decyzji...")
        result = await asyncio.to_thread(
            moderation_provider.moderate, text, "policy-check", "anonymous", "", False
        )
        await update.message.reply_text(_format_moderation_result(result, policy_check=True))
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        await update.message.reply_text(f"❌ {exc}")


async def handle_mod_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        content_id = _parse_quoted_args(context.args, 1, "/mod_status <content_id>")[0]
        row = await asyncio.to_thread(moderation_provider.repository.get_content, content_id)
        if not row:
            raise ValueError(f"Nie znaleziono content_id={content_id}.")
        await update.message.reply_text(
            f"📄 Status #{content_id}\nDecyzja: {row['action']}\n"
            f"Override: {row['moderator_override'] or 'brak'}\n"
            f"Powód: {row['reason']}\nUżytkownik: {row['user_id']}\nCzas: {row['timestamp']}"
        )
    except ValueError as exc:
        await update.message.reply_text(f"❌ {exc}")


async def handle_mod_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_id = _parse_quoted_args(context.args, 1, "/mod_history <user_id>")[0]
        row = await asyncio.to_thread(moderation_provider.get_user_moderation_history, user_id)
        await update.message.reply_text(
            f"👤 Historia użytkownika {user_id}\n"
            f"Naruszenia: {row['violations_count']}\n"
            f"Kategorie: {', '.join(row['categories']) or 'brak'}\n"
            f"Risk score: {row['risk_score']:.2f}\n"
            f"Repeat offender: {'tak' if row['is_repeat_offender'] else 'nie'}\n"
            f"Shadow bany: {row['shadow_bans']}\n"
            f"Ostatnie naruszenie: {row['last_violation'] or 'brak'}"
        )
    except ValueError as exc:
        await update.message.reply_text(f"❌ {exc}")


async def handle_mod_analytics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    analytics = await asyncio.to_thread(moderation_provider.repository.analytics)
    total = analytics["total"]
    actions = analytics["actions"]
    percentages = analytics["percentages"]
    top = "\n".join(f"- {name}: {count}" for name, count in analytics["top_violations"]) or "- brak"
    repeat = "\n".join(
        f"- {row['user_id']}: {row['total_violations']} naruszeń, "
        f"shadow bany: {row['shadow_bans']}"
        for row in analytics["repeat_offenders"]
    ) or "- brak"
    consensus = "\n".join(
        f"- {name}: {count} ({count / total * 100.0 if total else 0.0:.1f}%)"
        for name, count in sorted(analytics["consensus"].items())
    ) or "- brak"
    await _reply_long_text(update,
        f"📊 MODERATION ANALYTICS ({analytics['period']})\n"
        f"Łącznie: {total}\n"
        f"Approved: {actions.get('APPROVE', 0)} ({percentages.get('APPROVE', 0.0):.1f}%)\n"
        f"Rejected: {actions.get('REJECT', 0)} ({percentages.get('REJECT', 0.0):.1f}%)\n"
        f"Review: {actions.get('FLAG_FOR_REVIEW', 0)} ({percentages.get('FLAG_FOR_REVIEW', 0.0):.1f}%)\n"
        f"Human overrides: {analytics['human_overrides']}\n"
        f"Shadow bany: {analytics['shadow_bans']}\n"
        f"Średni czas: {analytics['average_seconds']:.3f}s\n\n"
        f"TOP NARUSZENIA:\n{top}\n\n"
        f"MODEL CONSENSUS:\n{consensus}\n\n"
        f"REPEAT OFFENDERS:\n{repeat}"
    )


async def handle_mod_add_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        content_id, comment, action = _parse_quoted_args(
            context.args, 3,
            '/mod_add_feedback <content_id> "komentarz" "APPROVE|REJECT|FLAG_FOR_REVIEW"',
        )
        row = await asyncio.to_thread(moderation_provider.add_feedback, content_id, comment, action)
        await update.message.reply_text(
            f"✅ Zapisano feedback dla #{content_id}: "
            f"{row['original_bot_decision']} → {row['moderator_override']}."
        )
    except ValueError as exc:
        await update.message.reply_text(f"❌ {exc}")


async def handle_mod_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = await asyncio.to_thread(moderation_provider.repository.watchlist)
    if not rows:
        await update.message.reply_text("ℹ️ Watchlista jest pusta.")
        return
    lines = ["⚠️ WATCHLIST"]
    for row in rows:
        until = f", ban do {row['shadow_ban_until']}" if row["shadow_ban_until"] else ""
        lines.append(f"- {row['user_id']}: {row['reason']}{until}")
    await _reply_long_text(update, "\n".join(lines))


async def handle_mod_train_on_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("⏳ Trenuję lokalny TF-IDF + LogisticRegression...")
        result = await asyncio.to_thread(moderation_provider.train_on_feedback)
        await update.message.reply_text(
            f"✅ Model feedbacku gotowy. Próbki: {result['samples']}, "
            f"klasy: {', '.join(result['classes'])}.\nModel: {_artifact_relative_path(result['model_path'])}"
        )
    except (ValueError, ImportError, OSError) as exc:
        await update.message.reply_text(f"❌ {exc}")


async def handle_mod_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🛡️ Lab6 — komendy moderacji\n"
        '/moderate "tekst"\n/mod_policy_check "tekst"\n'
        "/mod_status <content_id>\n/mod_history <user_id>\n"
        "/mod_analytics\n/mod_watchlist\n"
        '/mod_add_feedback <content_id> "komentarz" "APPROVE|REJECT|FLAG_FOR_REVIEW"\n'
        "/mod_train_on_feedback\n/mod_help\n\n"
        f"Dozwolone decyzje feedbacku: {', '.join(VALID_ACTIONS)}"
    )


# ---------------------------------------------------------------------------
# Uruchomienie bota
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if not BOT_TOKEN:
        raise RuntimeError(
            "Brak zmiennej środowiskowej BOT_TOKEN. "
            "Ustaw ją przed uruchomieniem, np. `export BOT_TOKEN=...`."
        )
    print("Uruchamianie bota NLP (Lab 2 + Lab 3 + Lab 4 + Lab 5 + Lab 6)...")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", send_welcome))
    app.add_handler(CommandHandler("help", send_welcome))
    app.add_handler(CommandHandler("classify", handle_classify))
    app.add_handler(CommandHandler("sentiment", handle_sentiment))
    app.add_handler(CommandHandler("train", handle_train))
    app.add_handler(CommandHandler("compare", handle_compare))
    app.add_handler(CommandHandler("add_sentiment", handle_add_sentiment))
    app.add_handler(CommandHandler("models", handle_models))
    app.add_handler(CommandHandler("ner", handle_ner))
    app.add_handler(CommandHandler("nel", handle_nel))
    app.add_handler(CommandHandler("ned", handle_ned))
    app.add_handler(CommandHandler("analyze_entities", handle_analyze_entities))
    app.add_handler(CommandHandler("language_detect", handle_language_detect))
    app.add_handler(CommandHandler("translate", handle_translate))
    app.add_handler(CommandHandler("summarize", handle_summarize))
    app.add_handler(CommandHandler("agent", handle_agent))
    app.add_handler(CommandHandler("moderate", handle_moderate))
    app.add_handler(CommandHandler("mod_policy_check", handle_mod_policy_check))
    app.add_handler(CommandHandler("mod_status", handle_mod_status))
    app.add_handler(CommandHandler("mod_history", handle_mod_history))
    app.add_handler(CommandHandler("mod_analytics", handle_mod_analytics))
    app.add_handler(CommandHandler("mod_add_feedback", handle_mod_add_feedback))
    app.add_handler(CommandHandler("mod_watchlist", handle_mod_watchlist))
    app.add_handler(CommandHandler("mod_train_on_feedback", handle_mod_train_on_feedback))
    app.add_handler(CommandHandler("mod_help", handle_mod_help))
    app.add_handler(MessageHandler(
        filters.PHOTO & filters.CaptionRegex(r"^/agent(?:@\w+)?(?:\s|$)"),
        handle_agent_photo,
    ))
    app.run_polling()
