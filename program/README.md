# NLP Bot — Laboratorium 2

## 📁 Struktura projektu

```
program/
├── main.py                          # Punkt wejścia — bot Telegram (kontroler)
├── requirements.txt                 # Zależności Python
├── lab2results.csv                  # Wyniki eksperymentów (auto-generowany)
├── lab2_similar_words.txt           # Podobne słowa Word2Vec/GloVe (auto-generowany)
├── lab2plots/                       # Wykresy (auto-generowany katalog)
│   ├── confusion_<emb>_<model>.png
│   ├── wordcloud_corpus.png
│   ├── wordcloud_class_<klasa>.png
│   ├── <dataset>_<model>_<emb>_pca_embedding.png
│   ├── <dataset>_<model>_<emb>_tsne_embedding.png
│   ├── <dataset>_<model>_<emb>_svd_embedding.png
│   ├── word_embedding_pca.png
│   ├── word_embedding_tsne.png
│   └── <dataset>_<model>_feature_importance.csv
└── providers/
    ├── dataset_provider.py          # Ładowanie zbiorów danych
    ├── embedding_provider.py        # Metody reprezentacji tekstu
    ├── classification_provider.py   # Modele ML i GridSearch
    └── visualization_provider.py   # Wykresy, WordCloud, redukcja wymiarowości
```

---

## ⚙️ Instalacja

### 1. Utwórz i aktywuj wirtualne środowisko

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Zainstaluj zależności

```bash
pip install -r requirements.txt
```

### 3. Uruchom bota

```bash
python main.py
```

---

## 🧠 Architektura (Clean Architecture)

```
Telegram Bot (Controller)
        │
        ▼
   main.py  ──────────────────────────────────────────────┐
        │                                                  │
        ▼                                                  ▼
DatasetProvider          EmbeddingProvider      VisualizationProvider
(Warstwa Danych)         (Reprezentacja)        (Wykresy / Raporty)
        │                       │
        └───────────────────────┘
                    │
                    ▼
        ClassificationProvider
        (Modele ML / GridSearch)
```

---

## 📊 Obsługiwane datasety

| Parametr        | Opis                                              | Źródło                    |
|-----------------|---------------------------------------------------|---------------------------|
| `20news_group`  | 20 kategorii newsów (~18k dokumentów)             | `sklearn.datasets`        |
| `imdb`          | Recenzje filmów (pos/neg, ~50k)                   | HuggingFace `datasets`    |
| `amazon`        | Recenzje Amazon (pos/neg, ~4M)                    | HuggingFace `datasets`    |
| `ag_news`       | Artykuły newsowe (4 klasy, ~130k)                 | HuggingFace `datasets`    |

---

## 🔤 Embeddingi (metody reprezentacji tekstu)

| Parametr    | Opis                                                  |
|-------------|-------------------------------------------------------|
| `bow`       | Bag of Words (`CountVectorizer`, max 5000 cech)       |
| `tfidf`     | TF-IDF (`TfidfVectorizer`, max 5000 cech)             |
| `word2vec`  | Word2Vec trenowany na bieżącym korpusie (Gensim)      |
| `glove`     | Pretrenowany GloVe 100d (pobierany automatycznie)     |

---

## 🤖 Modele klasyfikacji

| Parametr  | Model                    | GridSearch (siatka)                              |
|-----------|--------------------------|--------------------------------------------------|
| `nb`      | Multinomial Naive Bayes  | `alpha` ∈ {0.1, 0.5, 1.0}                       |
| `rf`      | Random Forest            | `n_estimators` ∈ {100, 300}, `max_depth` ∈ {None, 10, 20} |
| `logreg`  | Logistic Regression      | `C` ∈ {0.1, 1, 10}                              |
| `mlp`     | MLP Classifier           | `hidden_layer_sizes` ∈ {(128,), (256,128)}       |
| `all`     | Wszystkie powyższe       | —                                                |

---

## 📦 Generowane artefakty

| Plik / katalog                              | Opis                                        |
|---------------------------------------------|---------------------------------------------|
| `lab2results.csv`                           | Wyniki: embedding, model, accuracy, f1, seed|
| `lab2_similar_words.txt`                    | Podobne słowa (Word2Vec / GloVe)            |
| `lab2plots/confusion_<emb>_<model>.png`     | Macierz pomyłek                             |
| `lab2plots/wordcloud_corpus.png`            | Chmura słów — cały korpus                  |
| `lab2plots/wordcloud_class_<klasa>.png`     | Chmura słów — każda klasa osobno            |
| `lab2plots/*_pca_embedding.png`             | Wizualizacja PCA embeddingów dokumentów     |
| `lab2plots/*_tsne_embedding.png`            | Wizualizacja t-SNE embeddingów dokumentów   |
| `lab2plots/*_svd_embedding.png`             | Wizualizacja SVD embeddingów dokumentów     |
| `lab2plots/word_embedding_pca.png`          | PCA wybranych słów (W2V / GloVe)            |
| `lab2plots/word_embedding_tsne.png`         | t-SNE wybranych słów (W2V / GloVe)          |
| `lab2plots/*_feature_importance.csv`        | Top-10 najważniejszych słów per klasa       |

---

## 💬 Komenda bota

```
/classify dataset=<nazwa> method=<model> gridsearch=<true/false> run=<n> embedding=<typ>
```

### Parametry

| Parametr      | Opis                                         | Domyślna wartość |
|---------------|----------------------------------------------|------------------|
| `dataset`     | Zbiór danych                                 | —  (wymagany)    |
| `method`      | Model lub `all`                              | —  (wymagany)    |
| `gridsearch`  | Czy uruchomić GridSearchCV                   | `false`          |
| `run`         | Liczba uruchomień (1–3), wyniki uśredniane   | `1`              |
| `embedding`   | Metoda reprezentacji tekstu                  | `tfidf`          |

**Seedy dla `run`:**
- `run=1` → seed 42
- `run=2` → seed 42 + 1337 (wyniki uśrednione)
- `run=3` → seed 42 + 1337 + 2024 (wyniki uśrednione)

---

## 🧪 Przykładowe komendy testowe

Poniższe komendy prezentują kolejno wszystkie funkcje programu. Zalecana kolejność demonstracji.

---

### 1️⃣ Klasyfikacja — najprostsze uruchomienie

Naive Bayes z TF-IDF na datasecie 20 Newsgroups, jedno uruchomienie:

```
/classify dataset=20news_group method=nb gridsearch=false run=1
```

---

### 2️⃣ Porównanie wszystkich embeddingów (ten sam model, różne reprezentacje)

Pokazuje wpływ metody reprezentacji tekstu na jakość klasyfikacji:

```
/classify dataset=20news_group method=logreg gridsearch=false run=1 embedding=bow
```

```
/classify dataset=20news_group method=logreg gridsearch=false run=1 embedding=tfidf
```

```
/classify dataset=20news_group method=logreg gridsearch=false run=1 embedding=word2vec
```

```
/classify dataset=20news_group method=logreg gridsearch=false run=1 embedding=glove
```

---

### 3️⃣ Wszystkie modele naraz (`method=all`)

Bot trenuje `nb`, `rf`, `logreg` i `mlp` w jednym wywołaniu i zwraca porównanie:

```
/classify dataset=20news_group method=all gridsearch=false run=1 embedding=tfidf
```

---

### 4️⃣ Strojenie hiperparametrów (GridSearch)

GridSearchCV dobiera automatycznie najlepsze parametry — każdy model z inną siatką:

```
/classify dataset=20news_group method=nb gridsearch=true run=1 embedding=tfidf
```

```
/classify dataset=20news_group method=logreg gridsearch=true run=1 embedding=tfidf
```

```
/classify dataset=20news_group method=rf gridsearch=true run=1 embedding=tfidf
```

---

### 5️⃣ Wiele uruchomień z uśrednianiem (`run=N`)

`run=2` → dwa seedy (42, 1337), wyniki uśrednione:

```
/classify dataset=20news_group method=logreg gridsearch=false run=2 embedding=tfidf
```

`run=3` → trzy seedy (42, 1337, 2024), wyniki uśrednione:

```
/classify dataset=20news_group method=nb gridsearch=false run=3 embedding=tfidf
```

---

### 6️⃣ Różne datasety

Klasyfikacja sentymentu — recenzje filmów IMDB (2 klasy: pos/neg):

```
/classify dataset=imdb method=logreg gridsearch=false run=1 embedding=tfidf
```

Klasyfikacja kategorii artykułów — AG News (4 klasy: World/Sports/Business/Sci):

```
/classify dataset=ag_news method=nb gridsearch=false run=1 embedding=tfidf
```

```
/classify dataset=ag_news method=all gridsearch=false run=1 embedding=tfidf
```

Recenzje produktów Amazon (sentiment, duży dataset):

```
/classify dataset=amazon method=nb gridsearch=false run=1 embedding=bow
```

---

### 7️⃣ Embeddingi neuronowe — Word2Vec i GloVe

Word2Vec trenowany na bieżącym korpusie — generuje też `lab2_similar_words.txt` i wykresy słów:

```
/classify dataset=20news_group method=rf gridsearch=false run=1 embedding=word2vec
```

GloVe pretrenowany (przy pierwszym użyciu pobiera ~130 MB automatycznie):

```
/classify dataset=20news_group method=logreg gridsearch=false run=1 embedding=glove
```

---

### 8️⃣ Pełny eksperyment (wszystkie funkcje naraz)

Wszystkie modele + GridSearch + 3 seedy z uśrednianiem:

```
/classify dataset=20news_group method=all gridsearch=true run=3 embedding=tfidf
```

Analogicznie na innym datasecie z GloVe:

```
/classify dataset=ag_news method=logreg gridsearch=true run=3 embedding=glove
```
