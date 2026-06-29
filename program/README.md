# NLP Bot — Laboratorium 2, 3 i 4

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

Token Telegrama nie jest zapisany w kodzie. Przed startem ustaw zmienną środowiskową `BOT_TOKEN`.

```bash
# Linux / macOS
export BOT_TOKEN="8622639294:AAGSsDW82owPsvm5vZVJ3zIk7_NnKbnzhLI"
python main.py

# Windows PowerShell
$env:BOT_TOKEN="TU_WKLEJ_TOKEN_BOTA"
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

`nb` działa tylko z `bow` i `tfidf`. Dla `word2vec` oraz `glove` bot pomija `nb` w trybie `method=all` albo zwraca czytelny błąd dla pojedynczej komendy, ponieważ Multinomial Naive Bayes wymaga nieujemnych cech.

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

Bot domyślnie używa małej, deterministycznej próbki datasetu (`5%`) na potrzeby szybkiego uruchamiania podczas laboratorium. Dzięki temu kolejne uruchomienia na tych samych parametrach są porównywalne.

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

---

## Laboratorium 3 — analiza sentymentu

Lab3 rozszerza bota o analizę sentymentu pojedynczych tekstów, trenowanie modeli sekwencyjnych i porównywanie metod na datasetach `amazon`, `imdb` oraz `custom`. Dataset `custom` jest polski i zawiera 90 przykładów: po 30 dla klas `pozytywny`, `neutralny`, `negatywny`.

### Komendy Lab3

```
/sentiment method=<rule|nb|rf|transformer|textblob|stanza|simplernn|lstm|gru> text="tekst"
/train model=<simplernn|lstm|gru> dataset=<amazon|imdb|custom>
/compare dataset=<amazon|imdb|custom> methods=<lista_metod>
/add_sentiment "tekst" "etykieta"
/models
/help
```

### Datasety i artefakty

| Plik / katalog | Opis |
|----------------|------|
| `sentiment_dataset.csv` | Własny polski dataset dla `custom`, format `text,label` |
| `models/` | Modele `.h5`, tokenizery `.pkl` i encodery etykiet `.pkl` |
| `lab3plots/` | Historia uczenia, macierze pomyłek, wordcloud, porównania metod |
| `lab3results.csv` | Wyniki `/compare`: dataset, method, accuracy, precision, recall, macro_f1, model_path |

Etykiety w datasecie `custom`: `pozytywny`, `neutralny`, `negatywny`. Komenda `/add_sentiment` zapisuje wielozdaniowy tekst jako jeden rekord.

### Metody sentymentu

| Metoda | Opis |
|--------|------|
| `rule` | Rozszerzone reguły słów kluczowych dla języka polskiego |
| `nb` | Multinomial Naive Bayes z TF-IDF, trenowany i cache'owany dla wybranego datasetu |
| `rf` | Random Forest z TF-IDF, trenowany i cache'owany dla wybranego datasetu |
| `transformer` | Wielojęzyczny model `nlptown/bert-base-multilingual-uncased-sentiment` |
| `textblob` | Baseline TextBlob; najlepszy dla języka angielskiego |
| `stanza` | Opcjonalny baseline Stanza; sentiment najlepiej działa dla tekstów angielskich |
| `simplernn` | Zapisany model Keras `.h5` |
| `lstm` | Zapisany model Keras `.h5` |
| `gru` | Zapisany model Keras `.h5` |

Dla metod `simplernn`, `lstm` i `gru` trzeba najpierw uruchomić `/train`, ponieważ `/sentiment` wczytuje zapisany model z pliku zamiast trenować go od nowa.

### Parametry modeli sekwencyjnych

Domyślne parametry są dobrane tak, żeby dało się szybko pokazać działanie na laboratorium:

| Parametr | Wartość |
|----------|---------|
| `max_words` | 5000 |
| `max_len` | 100 |
| `embedding_dim` | 64 |
| `epochs` | 5 |
| `batch_size` | 16 dla `custom`, 32 dla większych datasetów |
| `early_stopping` | `patience=2`, `restore_best_weights=True` |

Do szybkiego treningu bot używa całego datasetu `custom`, 5 000 przykładów z IMDB oraz 10 000 przykładów z Amazon. Ograniczenie próbki zapobiega wielogodzinnym treningom modeli GRU/LSTM na pełnych datasetach.

Do eksperymentów warto porównać `max_len` z przedziału 50-200 oraz `embedding_dim` 50-100. Dłuższe sekwencje mogą poprawić wyniki na recenzjach IMDB/Amazon, ale zwiększają czas treningu.

## ⚠️ Uwagi wydajnościowe

- Pierwsze użycie `imdb`, `amazon` albo `ag_news` może pobrać dane przez bibliotekę HuggingFace `datasets`.
- Pierwsze użycie `embedding=glove` może pobrać model `glove-wiki-gigaword-100` przez Gensim.
- `gridsearch=true`, `method=all`, `mlp`, `rf` oraz `glove` mogą działać zauważalnie dłużej niż podstawowe demo `20news_group + tfidf + nb/logreg`.
- t-SNE jest automatycznie ograniczane do próbki dokumentów, żeby generowanie wizualizacji nie blokowało bota na dużych datasetach.

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

---

## Lab2 testy

Podstawowy szybki test klasyfikacji:

```
/classify dataset=20news_group method=nb gridsearch=false run=1 embedding=tfidf
```

Porownanie reprezentacji tekstu:

```
/classify dataset=20news_group method=logreg gridsearch=false run=1 embedding=bow
```

```
/classify dataset=20news_group method=logreg gridsearch=false run=1 embedding=tfidf
```

Wszystkie modele naraz:

```
/classify dataset=20news_group method=all gridsearch=false run=1 embedding=tfidf
```

Wiele uruchomien z usrednianiem:

```
/classify dataset=20news_group method=logreg gridsearch=false run=2 embedding=tfidf
```

GridSearch:

```
/classify dataset=20news_group method=nb gridsearch=true run=1 embedding=tfidf
```

Inne datasety:

```
/classify dataset=ag_news method=nb gridsearch=false run=1 embedding=tfidf
```

```
/classify dataset=imdb method=logreg gridsearch=false run=1 embedding=tfidf
```

Embeddingi neuronowe:

```
/classify dataset=20news_group method=logreg gridsearch=false run=1 embedding=word2vec
```

```
/classify dataset=20news_group method=logreg gridsearch=false run=1 embedding=glove
```

Test blednej kombinacji, powinien zwrocic czytelny blad:

```
/classify dataset=20news_group method=nb gridsearch=false run=1 embedding=glove
```

---

## Lab3 testy

Sprawdzenie dostępnych modeli:

```
/models
```

Dodanie własnego przykładu do polskiego datasetu `custom`:

```
/add_sentiment "Obsługa była poprawna, ale bez zachwytu" "neutralny"
```

Podstawowe metody sentymentu:

```
/sentiment method=rule dataset=custom text="To był świetny film"
```

```
/sentiment method=nb dataset=custom text="Produkt jest fatalny i bardzo słaby"
```

```
/sentiment method=rf dataset=custom text="Bardzo polecam ten zakup"
```

Model Transformer dla polskiego tekstu:

```
/sentiment method=transformer dataset=custom text="Obsługa była szybka i bardzo pomocna"
```

Trening i predykcja modelu SimpleRNN:

```
/train model=simplernn dataset=custom
```

```
/sentiment method=simplernn dataset=custom text="Produkt jest fatalny"
```

Porównanie podstawowych metod:

```
/compare dataset=custom methods=rule,nb,rf,transformer
```

Porównanie metod razem z wytrenowanym SimpleRNN:

```
/compare dataset=custom methods=rule,nb,rf,transformer,simplernn
```

Opcjonalne metody dla tekstów angielskich:

Przed pierwszym użyciem Stanza pobierz angielski model sentymentu w terminalu:

```bash
python3 -c "import stanza; stanza.download('en', processors='tokenize,sentiment')"
```

```
/sentiment method=textblob text="This product is excellent"
```

```
/sentiment method=stanza text="The movie was surprisingly good"
```

Opcjonalny trening pozostałych modeli sekwencyjnych:

```
/train model=lstm dataset=imdb
```

```
/train model=gru dataset=amazon
```

Test błędnej etykiety, powinien zwrócić czytelny błąd:

```
/add_sentiment "Ten tekst ma błędną etykietę" "super"
```

Test nieznanej metody, powinien zwrócić czytelny błąd:

```
/sentiment method=unknown dataset=custom text="Test nieznanej metody"
```

Test brakującego modelu, powinien zwrócić informację o wymaganym treningu:

```
/sentiment method=lstm dataset=custom text="Ten model nie był jeszcze trenowany"
```

---

## Lab4 testy

Lab4 dodaje NER (spaCy i Stanza), NEL/NED z Wikidata i lokalną bazą, detekcję języka,
tłumaczenie modelem M2M100 oraz podsumowania przez lokalne Ollama. Wyniki są zapisywane
w `program/lab4results/`.

Przed testami zainstaluj zależności i jawnie pobierz modele. Bot nie pobiera ich automatycznie
w trakcie obsługi komendy Telegram:

```bash
python3 -m pip install -r program/requirements.txt
python3 -m spacy download pl_core_news_sm
python3 -c "import stanza; stanza.download('pl', processors='tokenize,ner')"
python3 -c "from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer; n='facebook/m2m100_418M'; M2M100Tokenizer.from_pretrained(n); M2M100ForConditionalGeneration.from_pretrained(n)"
ollama pull gemma3:1b
```

NER spaCy — oczekiwane encje osoby, organizacji i lokalizacji:

```
/ner method=spacy text="Steve Jobs, współzałożyciel Apple, urodził się w San Francisco."
```

NER Stanza — drugi wymagany silnik:

```
/ner method=stanza text="Robert Lewandowski grał w klubie Bayern Monachium."
```

NER bez encji — oczekiwany poprawny wynik `Nie znaleziono encji`, bez wyjątku:

```
/ner method=spacy text="Dzisiaj jest bardzo pogodnie."
```

NEL — kandydaci z identyfikatorami Wikidata i linkami Wikipedii:

```
/nel text="Steve Jobs" language=pl
```

NEL encji wieloznacznej:

```
/nel text="Apple" language=en
```

NED — kontekst technologiczny powinien wybrać Apple Inc., a kontekst owocu jabłko:

```
/ned entity="Apple" context="Apple produkuje komputery Mac i telefony iPhone." language=pl
```

```
/ned entity="Apple" context="Apple is a fruit growing on a tree and used in pies." language=en
```

Analiza encji bez sieciowego linkowania:

```
/analyze_entities text="Elon Musk posiada firmę Tesla w Austin." link=false
```

Analiza połączona z NEL:

```
/analyze_entities text="Elon Musk posiada firmę Tesla w Austin." link=true
```

Detekcja języka polskiego i angielskiego:

```
/language_detect text="To jest przykładowe zdanie napisane po polsku."
```

```
/language_detect text="This sentence was written in English."
```

Tłumaczenia obejmujące wszystkie obsługiwane języki docelowe:

```
/translate text="This is a useful book." target_lang=pl
```

```
/translate text="To jest przydatna książka." target_lang=en
```

```
/translate text="This is a useful book." target_lang=de
```

```
/translate text="To jest przydatna książka." target_lang=fr
```

```
/translate text="This is a useful book." target_lang=es
```

Podsumowanie abstrakcyjne, krótkie:

```
/summarize text="Sztuczna inteligencja wspiera analizę dużych zbiorów danych. Modele językowe potrafią odpowiadać na pytania, tłumaczyć oraz streszczać tekst. Wyniki powinny być jednak sprawdzane przez człowieka." summary_type=abstractive length=short
```

Podsumowanie ekstrakcyjne, średnie:

```
/summarize text="Warszawa jest stolicą Polski. Miasto leży nad Wisłą i jest ważnym ośrodkiem gospodarczym. Znajduje się tam wiele uczelni, muzeów i instytucji kultury." summary_type=extractive length=medium
```

Podsumowanie punktowe, długie:

```
/summarize text="Uczenie maszynowe obejmuje przygotowanie danych, wybór modelu, trening i ewaluację. Dane należy oczyścić i podzielić na zbiory treningowe oraz testowe. Metryki trzeba dobrać do problemu. Gotowy model powinien być monitorowany po wdrożeniu." summary_type=bullets length=long
```

Niestandardowy prompt:

```
/summarize text="Projekt obejmuje implementację bota, testy automatyczne i dokumentację. Termin oddania przypada na piątek." summary_type=custom length=short prompt="Wypisz wyłącznie zadania i termin."
```

Test nieznanej metody NER — oczekiwany czytelny błąd:

```
/ner method=bert text="Warszawa"
```

Test nieobsługiwanego języka tłumaczenia — oczekiwany czytelny błąd:

```
/translate text="Test" target_lang=it
```

Test brakującego promptu custom — oczekiwany czytelny błąd:

```
/summarize text="Przykładowy tekst" summary_type=custom length=short
```

---

## Lab5 testy

Lab5 dodaje jedną komendę `/agent`, w której lokalny Qwen sam wybiera narzędzia:
Wikipedia, pogodę Open-Meteo, kalkulator, lokalną bazę Lab4 albo analizę obrazu przez
model Vision. Historia wraz z argumentami i wynikami narzędzi jest zapisywana w
`program/lab5results/tool_history.jsonl`.

Przed uruchomieniem sprawdź zależności, usługę Ollama i jawnie pobierz oba modele.
Bot nie pobiera modeli automatycznie:

```bash
python3 -m pip install -r program/requirements.txt
ollama serve
ollama pull qwen3:1.7b
ollama pull gemma4:latest
export BOT_TOKEN="token_bota"
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_TOOL_MODEL="qwen3:1.7b"
export OLLAMA_VISION_MODEL="gemma4:latest"
python3 program/main.py
```

Zwykła rozmowa — oczekiwane `Narzędzia: brak`:

```
/agent Cześć! Napisz jedno krótkie zdanie o NLP.
```

Kalkulator — oczekiwane użycie `simple_calculator` i wynik `391`:

```
/agent Ile to 17 razy 23?
```

Lokalna baza wiedzy — oczekiwane użycie `local_knowledge` i informacja o Steve Jobsie:

```
/agent Co lokalna baza wiedzy mówi o Steve Jobsie?
```

Aktualna pogoda — oczekiwane użycie `get_weather`:

```
/agent Jaka jest teraz pogoda w Warszawie?
```

Porównanie miast — oczekiwane dwa wywołania `get_weather` i odpowiedź porównawcza:

```
/agent Porównaj aktualną pogodę w Warszawie i Paryżu.
```

Aktualne informacje z internetu — oczekiwane użycie `web_search` i podanie źródła:

```
/agent Wyszukaj w internecie, kto jest CEO Tesli.
```

Scenariusz wieloetapowy — oczekiwane `get_weather` oraz `web_search`:

```
/agent Czy aktualna pogoda w Warszawie jest typowa dla czerwca? Porównaj ją z informacjami z internetu.
```

Vision — wyślij zdjęcie do bota z poniższym podpisem; oczekiwane użycie `analyze_image`:

```
/agent Co znajduje się na tym obrazie?
```

Pusta komenda bez zdjęcia — oczekiwany czytelny błąd z instrukcją użycia:

```
/agent
```

Błędne działanie matematyczne — agent powinien opisać błąd narzędzia, bez awarii bota:

```
/agent Oblicz 10 / 0 przy użyciu kalkulatora.
```

Nieznane miasto — oczekiwany komunikat o braku miasta, bez awarii bota:

```
/agent Jaka jest pogoda w mieście NieistniejąceMiastoXYZ?
```
