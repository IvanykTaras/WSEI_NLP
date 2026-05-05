# Laboratorium 6

## Temat

Content Moderation & Policy Enforcement Bot - Inteligentna moderacja treści z wykorzystaniem dedykowanych modeli bezpieczeństwa, Function Calling, Named Entity Recognition oraz analizy sentymentu.

---

## Cel laboratorium

Celem zadania jest stworzenie bota Telegram do automatycznej moderacji treści (komentarze, posty, recenzje), który:

- automatycznie detektuje toxic/hate speech/spam w tekście,
- kategoryzuje naruszenia polityki zaawansowanymi modelami,
- sugeruje akcje moderacyjne (approve/reject/flag for review/shadow ban),
- ekstraktuje dane kontekstowe z wiadomości (autor, wymienione podmioty),
- analizuje sentyment narażonego tekstu,
- wywołuje narzędzia (tools) do zarządzania moderacją,
- rejestruje repeat offenderów,
- generuje business intelligence dla moderatorów,
- uczy się z decyzji human moderators (feedback loop),
- działa 100% lokalnie na małych modelach.

---

## 🔧 Wymagane modele bezpieczeństwa

### 1. **OpenAI Privacy Filter**
- Model: `openai/privacy-filter`
- URL: https://huggingface.co/openai/privacy-filter
- Funkcja: Detekcja poufnych informacji (SSN, karty kredytowe, adresy email)
- Output: Lista wrażliwych fragmentów tekstu

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def detect_private_info(text: str) -> dict:
    """
    Detect personally identifiable information (PII).
    Returns: {'has_pii': bool, 'entities': List[dict]}
    """
```

### 2. **Bielik Guard 0.1B**
- Model: `speakleash/Bielik-Guard-0.1B-v1.0`
- URL: https://huggingface.co/speakleash/Bielik-Guard-0.1B-v1.0
- Funkcja: Detekcja toxic content, hate speech, spam
- Output: Klasyfikacja zagrożeń z confidence score
- Kategorie: `toxic`, `spam`, `hate_speech`, `self_harm`, `violence`, `sexual`, `clean`

```python
def classify_bielik_guard(text: str) -> dict:
    """
    Classify text using Bielik Guard model.
    Returns: {
        'label': str,
        'score': float,
        'severity': 'low|medium|high|critical'
    }
    """
```

### 3. **Qwen Guard**
- Model: `Qwen2.5-Guard` (mniejsza wersja)
- URL: https://huggingface.co/Qwen
- Funkcja: Zaawansowana kategoryzacja zagrożeń bazowana na LLM
- Output: Structured JSON z kategoriami i rekomendacjami

```python
def classify_qwen_guard(text: str) -> dict:
    """
    Classify text using Qwen Guard LLM-based model.
    Returns: {
        'risk_level': 'safe|low|medium|high|critical',
        'categories': List[str],
        'confidence': float,
        'recommended_action': 'approve|review|reject'
    }
    """
```

---

## 📊 Integracja modeli - Ensemble Strategy

Bot powinien kombinować decyzje wszystkich 3 modeli:

```
Jeśli OpenAI Privacy Filter → DETECT PII:
  action = "REJECT" (zawsze)
  reason = "Personal information detected"

Jeśli Bielik Guard confidence > 0.8:
  zagrożenie_określone = True

Jeśli Qwen Guard risk_level == "critical":
  action = "REJECT"
  flag_account = True

Otherwise:
  Głosowanie: jeśli ≥2 modele zgadzają się → ACTION
  Jeśli brak zgody → "FLAG_FOR_REVIEW"
```

---

## 😊 Sentiment Analysis (z Lab03)

Dla każdego tekstu należy również wykonać:

```python
def analyze_sentiment_for_moderation(text: str) -> dict:
    """
    Analyze sentiment to provide context for moderation.
    Returns: {
        'sentiment': 'positive|neutral|negative',
        'confidence': float,
        'emotion': 'anger|joy|sadness|fear|surprise|neutral',
        'sarcasm_detected': bool
    }
    """
```

**Ważne**: Tekst może być negatywny (negative) ale zgodny z polityką! Przykład: "Kupiłem ten produkt i jestem rozczarowany" = negative sentiment BUT approve.

---

## 🏷️ Named Entity Recognition (z Lab04)

Ekstrakcja danych z wiadomości:

```python
def extract_moderation_entities(text: str) -> dict:
    """
    Extract key entities for moderation context.
    Returns: {
        'usernames_mentioned': List[str],
        'urls': List[str],
        'emails': List[str],
        'phone_numbers': List[str],
        'organizations': List[str],
        'locations': List[str],
        'persons': List[str]
    }
    """
```

---

## 🔨 Function Calling / Tool Calling (z Lab05)

Bot powinien mieć dostęp do narzędzi do wykonania akcji moderacyjnych:

### Tool 1: Approve Content

```python
def approve_content(content_id: str, moderator_id: str) -> str:
    """Approve and publish flagged content."""
    # Save to DB: content_id, action=APPROVE, timestamp, moderator_id
```

### Tool 2: Reject Content

```python
def reject_content(content_id: str, reason: str, moderator_id: str) -> str:
    """Reject and remove content."""
    # Save to DB: content_id, action=REJECT, reason, timestamp
```

### Tool 3: Flag for Review

```python
def flag_for_human_review(content_id: str, priority: str, reason: str) -> str:
    """
    Flag content for manual review by human moderator.
    priority: 'low|medium|high|critical'
    """
```

### Tool 4: Shadow Ban User

```python
def shadow_ban_user(user_id: str, duration_hours: int, reason: str) -> str:
    """Limit user visibility (shadow ban) for defined period."""
```

### Tool 5: Get User History

```python
def get_user_moderation_history(user_id: str) -> dict:
    """
    Return user's moderation record.
    Returns: {
        'violations_count': int,
        'last_violation': datetime,
        'categories': List[str],
        'risk_score': float,
        'is_repeat_offender': bool
    }
    """
```

### Tool 6: Get Similar Cases

```python
def find_similar_violations(text: str, limit: int = 5) -> List[dict]:
    """Find similar previously moderated cases."""
```

### Tool 7: Add to Watchlist

```python
def add_to_watchlist(user_id: str, reason: str) -> str:
    """Add user to watchlist for increased monitoring."""
```

---

## 💬 Nowe komendy bota

```text
/moderate "tekst do sprawdzenia"
/mod_status <content_id>
/mod_history <user_id>
/mod_analytics
/mod_add_feedback <content_id> "komentarz" "poprawna_decyzja"
/mod_watchlist
/mod_train_on_feedback
/mod_policy_check "tekst"
/mod_help
```

---

## 📋 Workflow moderacji

### Scenariusz 1: Automatyczne approve

```
User: "Uwielbiam ten produkt, najlepszy zakup!"

Bot actions:
1. detect_private_info() → no PII
2. classify_bielik_guard() → "clean" (confidence 0.99)
3. classify_qwen_guard() → "safe"
4. analyze_sentiment() → "positive"
5. extract_entities() → no red flags
6. Action: APPROVE
7. Tool: approve_content()

Response: ✅ Wiadomość zatwierdzona
```

---

### Scenariusz 2: Automatyczne reject (toxic)

```
User: "Jesteś głupszy niż cegła, powinieneś się zabić"

Bot actions:
1. detect_private_info() → no PII
2. classify_bielik_guard() → "hate_speech + self_harm" (0.95)
3. classify_qwen_guard() → "critical"
4. analyze_sentiment() → "negative + anger detected"
5. extract_entities() → no specific target
6. Action: REJECT (consensus = 3/3 models)
7. Tool: reject_content(reason="hate_speech + self_harm")
8. Tool: shadow_ban_user(duration=24, reason="hate_speech")

Response: ❌ Wiadomość odrzucona, użytkownik wyciszony na 24h
Admin alert: User flagged as repeat offender
```

---

### Scenariusz 3: Flag for human review

```
User: "Ci politykanci to wszystko złodzieje! Wszyscy!!"

Bot actions:
1. detect_private_info() → no PII
2. classify_bielik_guard() → "hate_speech?" (confidence 0.65)
3. classify_qwen_guard() → "medium - political/opinion?"
4. analyze_sentiment() → "negative + anger"
5. extract_entities() → PERSON:politykanci, ORGANIZATION:government
6. Consensus: UNDECIDED (2 vs 1)
7. Action: FLAG_FOR_REVIEW
8. Tool: flag_for_human_review(priority="medium", reason="political_opinion_or_hate_speech")

Response: ⏳ Wiadomość czeka na weryfikację (priorytet: średni)
Moderator notified
```

---

### Scenariusz 4: PII Detection

```
User: "Mój numer to +48-123-456-789, email to john@example.com"

Bot actions:
1. detect_private_info() → PII DETECTED (phone, email)
2. Action: REJECT (mandatory)
3. Tool: reject_content(reason="personally_identifiable_information")

Response: ❌ Wiadomość zawiera dane osobowe i została usunięta
```

---

### Scenariusz 5: Learning from feedback

```
Moderator overrides bot decision:
Bot: REJECT (spam)
Moderator: APPROVE + comment "To legalne zaproszenie"

Bot actions:
1. Tool: add_feedback(content_id, moderator_override=APPROVE, reason="legalne")
2. Zapisz do train_data.csv
3. Po N decyzji: Tool: train_on_feedback()
4. Fine-tune modelu lokalnie

Response: ✅ Dziękujemy za feedback! Model będzie bardziej dokładny.
```

---

## 📊 Analytics & Reporting

Bot powinien generować raporty dla moderatorów:

### Komenda: `/mod_analytics`

Może zwrócić coś podobnego:

```
📊 MODERATION ANALYTICS (Today)
================================================

Total posts reviewed:     2,543
Approved:                 2,180 (85.7%)
Rejected:                 245   (9.6%)
Flagged for review:       118   (4.6%)

TOP VIOLATIONS:
1. Hate speech:           87 cases
2. Spam:                  65 cases
3. Self-harm:             23 cases
4. Violence:              18 cases
5. Sexual content:        15 cases

REPEAT OFFENDERS:
1. user_123:    45 violations (shadow ban 3x)
2. user_456:    28 violations (shadow ban 1x)
3. user_789:    15 violations

MODEL CONSENSUS:
- All 3 models agree:     1,854 (72.8%)
- 2/3 agree:              512   (20.1%)
- Conflicting:            177   (7.0%)
- Human overrides:        42    (1.6%)

RESPONSE TIME: avg 0.23s per post
```

---

## 📁 Dane do przechowywania

### `moderation_log.csv`

```csv
timestamp,content_id,user_id,text,model_bielik_decision,model_bielik_score,model_qwen_decision,model_qwen_score,pii_detected,sentiment,action,moderator_override,reason,appeal_filed
2024-05-05T10:23:14,1001,user123,"tekst...",toxic,0.87,high,0.92,False,negative,REJECT,False,hate_speech,False
2024-05-05T10:24:31,1002,user456,"tekst...",clean,0.98,safe,0.96,False,neutral,APPROVE,False,,False
```

### `user_moderation_history.csv`

```csv
user_id,username,total_violations,last_violation_date,categories,risk_score,is_repeat_offender,shadow_bans,appeals_filed
user123,JohnDoe,12,2024-05-05,hate_speech;spam,0.82,True,3,1
user456,JaneSmith,2,2024-05-04,spam,0.31,False,0,0
```

### `feedback_log.csv`

```csv
content_id,original_bot_decision,moderator_override,text_sample,category,confidence_before,confidence_after,timestamp
1045,REJECT,APPROVE,"legalne zaproszenie...",spam,0.75,0.65,2024-05-05T14:30:22
1078,APPROVE,REJECT,"wulgarne słowa...",hate_speech,0.45,0.89,2024-05-05T15:45:11
```

---
