---
name: wsei-nlp-labs
description: Use when working in the WSEI_NLP repository on WSEI NLP laboratory tasks such as lab2, lab3, lab4, lab5, or lab6; implementing or stabilizing Python NLP/ML bot functionality; adding README test commands; fixing macOS dependency/runtime issues; or preparing lab deliverables with generated results, plots, and verification notes.
---

# WSEI NLP Labs

## Workflow

1. Ground in the repository before changing code:
   - Check the current branch and dirty files with `git status --short --branch`.
   - Read the relevant lab brief first, for example `lab3.md`, `Lab03.md`, or similarly named files.
   - Inspect `program/README.md`, `program/main.py`, `program/requirements.txt`, and existing providers before choosing an implementation style.

2. Implement in the existing project shape:
   - Prefer the current `program/` layout and provider pattern over creating a new app.
   - Keep Telegram command behavior explicit and easy to demo.
   - Add parameter validation near command parsing so user-facing errors are readable.
   - Check cheap preconditions such as params, files, and saved models before importing TensorFlow, Transformers, Stanza, or loading datasets.
   - Run blocking ML, downloads, plots, and inference outside the Telegram event loop, and send a progress message before slow work.
   - Save generated artifacts under `program/` paths, not the process working directory.
   - Update `program/README.md` with a `LabX testy` section containing copy-paste Telegram commands.
   - Place each new `LabX testy` section at the bottom of `program/README.md`, after existing lab test sections, and mirror their simple scenario-description-plus-command format.

3. Treat macOS as the default student environment:
   - Use `python3` in instructions.
   - Expect system Python 3.9.x and LibreSSL.
   - Keep `requirements.txt` pins installable on macOS/Python 3.9 unless the user says otherwise.
   - For headless plot generation, ensure Matplotlib uses a non-GUI backend such as `Agg`.
   - Put library caches in writable locations when needed, for example temp dirs for Matplotlib/sklearn cache.

4. Verify like a lab submission:
   - Run syntax checks with `python3 -m py_compile` or the local `.venv/bin/python`.
   - Install or check dependencies with `pip3 install -r program/requirements.txt` and `pip check` when feasible.
   - Run the exact public commands documented in `LabX testy`, in their documented order and with their prerequisites.
   - Test at least one positive and one negative semantic example for every classifier or generator; metrics alone are insufficient.
   - Test expected bad inputs through the public command handler, especially invalid params, missing artifacts, and unsupported combinations.
   - Confirm generated artifact paths and mention any tests skipped because of network, dataset download, token, or runtime constraints.

## Runtime Guardrails

- Calculate the absolute sample count and approximate training steps before starting ML work. Never infer safety from a percentage of an unknown dataset.
- For demo commands, cap external datasets to a deliberate absolute-scale sample and target completion in seconds or a few minutes. Keep full runs explicitly optional.
- Select HuggingFace rows before converting columns to NumPy or Python lists; do not materialize millions of rows and sample afterward.
- Use small bounded defaults for epochs, batch size, search grids, generation length, and comparison sets. Add early stopping where applicable.
- Time one representative run before documenting duration. If it exceeds the demo budget, reduce work while preserving meaningful class coverage.
- Cache loaded models and reusable pipelines, use batch prediction in comparisons, and invalidate caches when training data changes.
- Keep downloads explicit. Check local resources first and provide a minimal one-time download command instead of downloading silently inside a Telegram handler.
- After code changes, detect running or suspended `main.py` processes. Explain that `Ctrl+C` stops a bot while `Ctrl+Z` only suspends it, and restart before retesting changed code.

## README Test Sequence

- Put setup or download prerequisites before the command that needs them.
- Order tests from fast and local to trained, downloaded, or otherwise heavy scenarios.
- Train or create each required artifact before its success test; label missing-artifact commands explicitly as expected-error tests.
- Include expected outcomes for error cases so silence, a crash, and a correct rejection cannot be confused.

## Lab Deliverable Checklist

- Lab brief requirements mapped to implemented features.
- Public command/API documented in `program/README.md`.
- Test commands grouped in a section named `LabX testy` at the bottom of `program/README.md`, formatted analogously to the existing lab test sections.
- Dependencies install on the user's macOS setup.
- Results, plots, or generated files are written to stable project paths.
- Final response lists changed files and verification commands.

## Lessons From Lab2 And Lab3

- Telegram token should come from `BOT_TOKEN`, not a hardcoded string.
- `MultinomialNB` should be limited to non-negative sparse/vectorizer features such as `bow` and `tfidf`; reject or skip it for dense embeddings such as `word2vec` and `glove`.
- Use deterministic seeds for sampling and repeated runs.
- Use clear Polish user-facing messages for bot errors and lab instructions.
- Keep demo commands fast first, then include heavier commands separately for GridSearch, neural embeddings, or large datasets.
- Do not use a fixed fraction across differently sized datasets: two percent of Amazon is 72,000 reviews, while two percent of IMDB is only 1,000.
- Re-train generated models after changing the dataset, tokenizer, architecture, padding, or training defaults, then verify predictions from the saved artifact.
- Treat a non-responsive Telegram command as a runtime defect: inspect process state and logs, reproduce at provider level, then verify the same public command after restarting the bot.
