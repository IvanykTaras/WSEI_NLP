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
   - Run at least one light end-to-end experiment or an equivalent provider-level smoke test.
   - Test expected bad inputs, especially invalid params and unsupported model/embedding combinations.
   - Confirm generated artifact paths and mention any tests skipped because of network, dataset download, token, or runtime constraints.

## Lab Deliverable Checklist

- Lab brief requirements mapped to implemented features.
- Public command/API documented in `program/README.md`.
- Test commands grouped in a section named `LabX testy` at the bottom of `program/README.md`, formatted analogously to the existing lab test sections.
- Dependencies install on the user's macOS setup.
- Results, plots, or generated files are written to stable project paths.
- Final response lists changed files and verification commands.

## Patterns From Lab2

- Telegram token should come from `BOT_TOKEN`, not a hardcoded string.
- `MultinomialNB` should be limited to non-negative sparse/vectorizer features such as `bow` and `tfidf`; reject or skip it for dense embeddings such as `word2vec` and `glove`.
- Use deterministic seeds for sampling and repeated runs.
- Use clear Polish user-facing messages for bot errors and lab instructions.
- Keep demo commands fast first, then include heavier commands separately for GridSearch, neural embeddings, or large datasets.
