import csv
import atexit
import json
import math
import os
import re
import select
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB6_RESULTS_DIR = os.path.join(BASE_DIR, "lab6results")
BIELIK_MODEL = "speakleash/Bielik-Guard-0.1B-v1.0"
QWEN_GUARD_MODEL = "Qwen/Qwen3Guard-Gen-0.6B"
PRIVACY_MODEL = "openai/privacy-filter"
VALID_ACTIONS = ("APPROVE", "REJECT", "FLAG_FOR_REVIEW")
MAX_MODERATION_TEXT_LENGTH = 3000

MODERATION_FIELDS = (
    "timestamp", "content_id", "user_id", "username", "text",
    "model_bielik_decision", "model_bielik_score", "model_qwen_decision",
    "model_qwen_score", "pii_detected", "pii_source", "sentiment", "emotion",
    "action", "moderator_override", "reason", "appeal_filed", "consensus",
    "duration_seconds", "moderator_id", "tool_action",
)
HISTORY_FIELDS = (
    "user_id", "username", "total_violations", "last_violation_date",
    "categories", "risk_score", "is_repeat_offender", "shadow_bans",
    "appeals_filed",
)
FEEDBACK_FIELDS = (
    "content_id", "original_bot_decision", "moderator_override", "comment",
    "text_sample", "category", "confidence_before", "confidence_after", "timestamp",
)
WATCHLIST_FIELDS = ("user_id", "reason", "added_at", "shadow_ban_until")


class ModerationRepository:
    """Thread-safe CSV persistence for Lab6 artifacts."""

    def __init__(self, results_dir: str = LAB6_RESULTS_DIR):
        self.results_dir = os.path.realpath(results_dir)
        self.moderation_file = os.path.join(self.results_dir, "moderation_log.csv")
        self.history_file = os.path.join(self.results_dir, "user_moderation_history.csv")
        self.feedback_file = os.path.join(self.results_dir, "feedback_log.csv")
        self.watchlist_file = os.path.join(self.results_dir, "watchlist.csv")
        self.feedback_model_file = os.path.join(self.results_dir, "feedback_model.joblib")
        self._lock = threading.RLock()
        os.makedirs(self.results_dir, exist_ok=True)
        for path, fields in (
            (self.moderation_file, MODERATION_FIELDS),
            (self.history_file, HISTORY_FIELDS),
            (self.feedback_file, FEEDBACK_FIELDS),
            (self.watchlist_file, WATCHLIST_FIELDS),
        ):
            self._ensure_schema(path, fields)

    def append_moderation(self, row: dict) -> None:
        with self._lock:
            self._append(self.moderation_file, MODERATION_FIELDS, row)

    def get_content(self, content_id: str) -> Optional[dict]:
        rows = self._read(self.moderation_file)
        return next((row for row in reversed(rows) if row["content_id"] == str(content_id)), None)

    def moderation_rows(self) -> list:
        return self._read(self.moderation_file)

    def feedback_rows(self) -> list:
        return self._read(self.feedback_file)

    def add_feedback(self, row: dict) -> None:
        with self._lock:
            content_id = str(row["content_id"])
            feedback = self._read(self.feedback_file)
            feedback = [item for item in feedback if item["content_id"] != content_id]
            feedback.append(row)
            self._rewrite(self.feedback_file, FEEDBACK_FIELDS, feedback)
            moderation = self._read(self.moderation_file)
            for item in moderation:
                if item["content_id"] == content_id:
                    item["moderator_override"] = str(row["moderator_override"])
            self._rewrite(self.moderation_file, MODERATION_FIELDS, moderation)

    def rewrite_feedback(self, rows: list) -> None:
        with self._lock:
            self._rewrite(self.feedback_file, FEEDBACK_FIELDS, rows)

    def apply_content_action(
        self, content_id: str, action: str, reason: str, moderator_id: str,
        tool_action: str,
    ) -> dict:
        with self._lock:
            content_id = str(content_id)
            rows = self._read(self.moderation_file)
            row = next((item for item in reversed(rows) if item["content_id"] == content_id), None)
            if row is None:
                row = {field: "" for field in MODERATION_FIELDS}
                row.update({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "content_id": content_id,
                    "action": action,
                    "reason": reason,
                    "consensus": "manual_tool",
                })
                rows.append(row)
            row["action"] = action
            row["reason"] = reason
            row["moderator_id"] = str(moderator_id)
            row["tool_action"] = tool_action
            self._rewrite(self.moderation_file, MODERATION_FIELDS, rows)
            return dict(row)

    def get_user_history(self, user_id: str) -> dict:
        user_id = str(user_id)
        rows = self._read(self.history_file)
        row = next((item for item in rows if item["user_id"] == user_id), None)
        return row or {
            "user_id": user_id,
            "username": "",
            "total_violations": "0",
            "last_violation_date": "",
            "categories": "",
            "risk_score": "0.0",
            "is_repeat_offender": "False",
            "shadow_bans": "0",
            "appeals_filed": "0",
        }

    def record_user_action(
        self, user_id: str, username: str, action: str, categories: list,
        shadow_ban: bool = False,
    ) -> dict:
        with self._lock:
            rows = self._read(self.history_file)
            user_id = str(user_id)
            row = next((item for item in rows if item["user_id"] == user_id), None)
            if row is None:
                row = self.get_user_history(user_id)
                rows.append(row)
            row["username"] = username or row.get("username", "")
            if action == "REJECT":
                violations = int(row["total_violations"]) + 1
                previous = set(filter(None, row.get("categories", "").split(";")))
                previous.update(categories)
                row["total_violations"] = str(violations)
                row["last_violation_date"] = datetime.now(timezone.utc).isoformat()
                row["categories"] = ";".join(sorted(previous))
                row["risk_score"] = f"{min(1.0, violations / 5.0):.2f}"
                row["is_repeat_offender"] = str(violations >= 3)
            if shadow_ban:
                row["shadow_bans"] = str(int(row["shadow_bans"]) + 1)
            self._rewrite(self.history_file, HISTORY_FIELDS, rows)
            return dict(row)

    def add_watchlist(self, user_id: str, reason: str, shadow_ban_until: str = "") -> dict:
        with self._lock:
            rows = self._read(self.watchlist_file)
            user_id = str(user_id)
            row = next((item for item in rows if item["user_id"] == user_id), None)
            payload = {
                "user_id": user_id,
                "reason": reason,
                "added_at": datetime.now(timezone.utc).isoformat(),
                "shadow_ban_until": shadow_ban_until,
            }
            if row is None:
                rows.append(payload)
            else:
                row.update(payload)
            self._rewrite(self.watchlist_file, WATCHLIST_FIELDS, rows)
            return payload

    def watchlist(self) -> list:
        return self._read(self.watchlist_file)

    def analytics(self, today_only: bool = True) -> dict:
        rows = self._read(self.moderation_file)
        period = "all"
        if today_only:
            warsaw = ZoneInfo("Europe/Warsaw")
            today = datetime.now(warsaw).date()
            filtered = []
            for row in rows:
                try:
                    stamp = datetime.fromisoformat(row["timestamp"])
                    if stamp.astimezone(warsaw).date() == today:
                        filtered.append(row)
                except (KeyError, TypeError, ValueError):
                    continue
            rows = filtered
            period = today.isoformat()
        actions = Counter(row["action"] for row in rows)
        categories = Counter()
        consensus = Counter()
        durations = []
        for row in rows:
            if row["action"] == "REJECT":
                categories.update(filter(None, row.get("reason", "").split(";")))
            consensus[row.get("consensus", "unknown")] += 1
            try:
                durations.append(float(row.get("duration_seconds", 0)))
            except ValueError:
                pass
        histories = self._read(self.history_file)
        repeat = sorted(
            (item for item in histories if item["is_repeat_offender"] == "True"),
            key=lambda item: int(item["total_violations"]), reverse=True,
        )
        total = len(rows)
        percentages = {
            action: (count / total * 100.0 if total else 0.0)
            for action, count in actions.items()
        }
        return {
            "period": period,
            "total": len(rows),
            "actions": dict(actions),
            "percentages": percentages,
            "top_violations": categories.most_common(5),
            "repeat_offenders": repeat[:5],
            "consensus": dict(consensus),
            "human_overrides": sum(bool(row.get("moderator_override")) for row in rows),
            "average_seconds": sum(durations) / len(durations) if durations else 0.0,
            "shadow_bans": sum(int(item.get("shadow_bans", 0)) for item in histories),
        }

    @staticmethod
    def _ensure_schema(path: str, fields: tuple) -> None:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=fields).writeheader()
            return
        with open(path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fields = tuple(reader.fieldnames or ())
            rows = list(reader)
        if existing_fields != fields:
            ModerationRepository._rewrite(path, fields, rows)

    @staticmethod
    def _read(path: str) -> list:
        with open(path, encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _append(path: str, fields: tuple, row: dict) -> None:
        with open(path, "a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore").writerow(row)

    @staticmethod
    def _rewrite(path: str, fields: tuple, rows: list) -> None:
        directory = os.path.dirname(path)
        fd, temporary = tempfile.mkstemp(prefix="lab6_", suffix=".csv", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class PrivacyFilterWorker:
    """Persistent local Transformers.js worker for OpenAI Privacy Filter q4."""

    def __init__(self, model_dir: str, timeout_seconds: int = 120):
        node = shutil.which("node")
        worker = os.path.join(os.path.dirname(__file__), "privacy_filter_worker.mjs")
        node_package = os.path.join(BASE_DIR, "node_modules", "@huggingface", "transformers")
        if not node:
            raise FileNotFoundError("Brak Node.js wymaganego przez lokalny Privacy Filter q4.")
        if not os.path.isdir(node_package):
            raise FileNotFoundError(
                "Brak zależności Transformers.js. Uruchom `npm install --prefix program`."
            )
        if not os.path.isfile(os.path.join(model_dir, "onnx", "model_q4.onnx")):
            raise FileNotFoundError(
                f"Brak lokalnego modelu {PRIVACY_MODEL}. Uruchom `hf download {PRIVACY_MODEL}`."
            )
        environment = os.environ.copy()
        environment["PRIVACY_FILTER_MODEL_DIR"] = model_dir
        self.process = subprocess.Popen(
            [node, worker],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._request_id = 0
        self._closed = False
        atexit.register(self.close)
        ready = self._read_json(timeout_seconds)
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(
                f"Nie udało się uruchomić Privacy Filter q4: {ready.get('error', 'nieznany błąd')}"
            )

    def __call__(self, text: str) -> list:
        with self._lock:
            if self.process.poll() is not None:
                raise RuntimeError("Proces Privacy Filter zakończył działanie.")
            self._request_id += 1
            request_id = self._request_id
            payload = json.dumps({"id": request_id, "text": text}, ensure_ascii=False)
            self.process.stdin.write(payload + "\n")
            self.process.stdin.flush()
            response = self._read_json(self.timeout_seconds)
            if response.get("error"):
                raise RuntimeError(f"Błąd Privacy Filter: {response['error']}")
            if response.get("id") != request_id:
                raise RuntimeError("Privacy Filter zwrócił odpowiedź z niezgodnym ID.")
            return response.get("rows", [])

    def _read_json(self, timeout_seconds: int) -> dict:
        if self.process.stdout is None:
            raise RuntimeError("Brak strumienia odpowiedzi Privacy Filter.")
        ready, _, _ = select.select([self.process.stdout], [], [], timeout_seconds)
        if not ready:
            raise TimeoutError(
                f"Privacy Filter nie odpowiedział w ciągu {timeout_seconds} sekund."
            )
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Privacy Filter zakończył działanie bez odpowiedzi.")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Niepoprawna odpowiedź Privacy Filter: {line[:200]}") from exc

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        process = getattr(self, "process", None)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        for stream_name in ("stdin", "stdout"):
            stream = getattr(process, stream_name, None) if process else None
            if stream and not stream.closed:
                stream.close()

    def __del__(self):
        self.close()


class ModerationProvider:
    def __init__(
        self,
        sentiment_provider=None,
        entity_provider=None,
        repository: Optional[ModerationRepository] = None,
        bielik_classifier: Optional[Callable] = None,
        qwen_classifier: Optional[Callable] = None,
        privacy_classifier: Optional[Callable] = None,
    ):
        self.sentiment_provider = sentiment_provider
        self.entity_provider = entity_provider
        self.repository = repository or ModerationRepository()
        self._bielik_classifier = bielik_classifier
        self._qwen_classifier = qwen_classifier
        self._privacy_classifier = privacy_classifier
        self._feedback_model = None
        self._model_lock = threading.RLock()

    def moderate(self, text: str, content_id: str, user_id: str, username: str = "", persist: bool = True) -> dict:
        text = self._validate_text(text)
        started = time.monotonic()
        pii = self.detect_private_info(text)
        if pii["has_pii"]:
            bielik = {
                "label": "skipped_pii", "score": 0.0, "severity": "low",
                "categories": [], "scores": {},
            }
            qwen = {
                "risk_level": "skipped_pii", "categories": [], "confidence": 0.0,
                "recommended_action": "approve",
            }
        else:
            bielik = self.classify_bielik_guard(text)
            qwen = self.classify_qwen_guard(text)
        sentiment = self.analyze_sentiment_for_moderation(text)
        entities = self.extract_moderation_entities(text)
        similar_cases = self.find_similar_violations(text)
        feedback = self._predict_feedback(text)
        decision = self.ensemble_decision(pii, bielik, qwen, feedback, text=text)
        duration = time.monotonic() - started
        result = {
            "content_id": str(content_id), "user_id": str(user_id), "username": username,
            "text": text, "pii": pii, "bielik": bielik, "qwen": qwen,
            "sentiment": sentiment, "entities": entities, "feedback_model": feedback,
            "similar_cases": similar_cases,
            "action": decision["action"], "reason": decision["reason"],
            "consensus": decision["consensus"], "votes": decision["votes"],
            "flag_account": decision["flag_account"], "duration_seconds": duration,
            "executed_tools": [],
        }
        if persist:
            result["executed_tools"] = self._execute_decision(result)
        return result

    @property
    def tool_schemas(self) -> list:
        """Ollama-compatible schemas for the seven moderation tools from Lab06."""
        return [
            self._tool_schema("approve_content", "Zatwierdź treść.", {
                "content_id": "string", "moderator_id": "string",
            }, ["content_id", "moderator_id"]),
            self._tool_schema("reject_content", "Odrzuć treść z podaniem powodu.", {
                "content_id": "string", "reason": "string", "moderator_id": "string",
            }, ["content_id", "reason", "moderator_id"]),
            self._tool_schema("flag_for_human_review", "Przekaż treść moderatorowi.", {
                "content_id": "string", "priority": "string", "reason": "string",
            }, ["content_id", "priority", "reason"]),
            self._tool_schema("shadow_ban_user", "Ogranicz widoczność użytkownika.", {
                "user_id": "string", "duration_hours": "integer", "reason": "string",
            }, ["user_id", "duration_hours", "reason"]),
            self._tool_schema("get_user_moderation_history", "Pobierz historię moderacji użytkownika.", {
                "user_id": "string",
            }, ["user_id"]),
            self._tool_schema("find_similar_violations", "Znajdź podobne naruszenia.", {
                "text": "string", "limit": "integer",
            }, ["text"]),
            self._tool_schema("add_to_watchlist", "Dodaj użytkownika do watchlisty.", {
                "user_id": "string", "reason": "string",
            }, ["user_id", "reason"]),
        ]

    def call_tool(self, name: str, arguments: dict):
        tools = {
            "approve_content": self.approve_content,
            "reject_content": self.reject_content,
            "flag_for_human_review": self.flag_for_human_review,
            "shadow_ban_user": self.shadow_ban_user,
            "get_user_moderation_history": self.get_user_moderation_history,
            "find_similar_violations": self.find_similar_violations,
            "add_to_watchlist": self.add_to_watchlist,
        }
        handler = tools.get(name)
        if not handler:
            raise ValueError(f"Nieznane narzędzie moderacji: {name}.")
        return handler(**(arguments or {}))

    def detect_private_info(self, text: str) -> dict:
        text = self._validate_text(text)
        if self._privacy_classifier is None:
            with self._model_lock:
                if self._privacy_classifier is None:
                    try:
                        self._privacy_classifier = self._load_privacy_classifier()
                    except (ImportError, OSError, RuntimeError, ValueError, TimeoutError):
                        if os.getenv("LAB6_ALLOW_PII_FALLBACK", "false").lower() != "true":
                            raise
                        self._privacy_classifier = False
        if self._privacy_classifier:
            try:
                rows = self._privacy_classifier(text)
                model_entities = [
                    {
                        "type": row.get("entity_group") or row.get("entity", "private"),
                        "text": row.get("word", "").strip(),
                        "score": float(row.get("score", 0.0)),
                    }
                    for row in rows if float(row.get("score", 0.0)) >= 0.5
                ]
                regex_entities = self._regex_pii(text)
                entities = list(model_entities)
                seen = {(row["type"], row["text"].casefold()) for row in entities}
                regex_added = False
                for row in regex_entities:
                    key = (row["type"], row["text"].casefold())
                    if key not in seen:
                        entities.append(row)
                        seen.add(key)
                        regex_added = True
                source = "model+regex" if regex_added else "model"
                return {"has_pii": bool(entities), "entities": entities, "source": source}
            except Exception:
                if os.getenv("LAB6_ALLOW_PII_FALLBACK", "false").lower() != "true":
                    raise
        entities = self._regex_pii(text)
        return {"has_pii": bool(entities), "entities": entities, "source": "regex_fallback"}

    def classify_bielik_guard(self, text: str) -> dict:
        text = self._validate_text(text)
        if self._bielik_classifier is None:
            with self._model_lock:
                if self._bielik_classifier is None:
                    self._bielik_classifier = self._load_bielik_classifier()
        classifier = self._bielik_classifier
        rows = classifier(text)
        if rows and isinstance(rows[0], list):
            rows = rows[0]
        scores = {
            self._normalize_bielik_label(row.get("label", "")): float(row.get("score", 0.0))
            for row in rows
        }
        active = sorted(
            ((label, score) for label, score in scores.items() if label != "clean" and score >= 0.5),
            key=lambda item: item[1], reverse=True,
        )
        spam_score = self._spam_score(text)
        if spam_score >= 0.5:
            active.append(("spam", spam_score))
            active.sort(key=lambda item: item[1], reverse=True)
            scores["spam"] = spam_score
        label, score = active[0] if active else ("clean", max([1.0 - max(scores.values(), default=0.0), scores.get("clean", 0.0)]))
        severity = "high" if score >= 0.8 and label != "clean" else "medium" if label != "clean" else "low"
        return {"label": label, "score": score, "severity": severity, "categories": [x[0] for x in active], "scores": scores}

    def classify_qwen_guard(self, text: str) -> dict:
        text = self._validate_text(text)
        if self._qwen_classifier is None:
            with self._model_lock:
                if self._qwen_classifier is None:
                    self._qwen_classifier = self._load_qwen_classifier()
        classifier = self._qwen_classifier
        raw = classifier(text)
        if isinstance(raw, dict):
            return self._normalize_qwen_result(raw)
        content, confidence = raw if isinstance(raw, tuple) else (str(raw), 0.8)
        safety = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", content, re.I)
        if not safety:
            raise RuntimeError("Qwen Guard zwrócił wynik bez pola Safety.")
        categories_match = re.search(r"Categories?:\s*(.+)", content, re.I)
        categories = []
        if categories_match:
            categories = [x.strip() for x in re.split(r"[,;]", categories_match.group(1)) if x.strip().lower() != "none"]
        return self._normalize_qwen_result({"safety": safety.group(1), "categories": categories, "confidence": confidence})

    @staticmethod
    def _normalize_qwen_result(row: dict) -> dict:
        safety = str(row.get("safety") or row.get("risk_level") or "").lower()
        categories = [str(item) for item in row.get("categories", [])]
        severe = any(re.search(r"violent|suicide|self.harm|pii", item, re.I) for item in categories)
        if safety in ("unsafe", "high", "critical"):
            risk = "critical" if severe or safety == "critical" else "high"
            action = "reject"
        elif safety in ("controversial", "medium", "low"):
            risk, action = "medium", "review"
        else:
            risk, action = "safe", "approve"
        return {
            "risk_level": risk, "categories": categories,
            "confidence": float(row.get("confidence", 0.8)),
            "recommended_action": action,
        }

    def analyze_sentiment_for_moderation(self, text: str) -> dict:
        lowered = text.casefold()
        raw = self.sentiment_provider.predict("rule", text) if self.sentiment_provider else {"label": "neutralny", "score": 0.0}
        labels = {"pozytywny": "positive", "negatywny": "negative", "neutralny": "neutral"}
        emotion_terms = {
            "anger": (
                "nienawidzę", "wściek", "złodziej", "idiot", "głupi", "głupszy",
                "zabij", "zabić", "powinieneś się",
            ),
            "joy": ("uwielbiam", "cieszę", "świetny", "super", "najlepszy"),
            "sadness": ("smut", "rozczar", "płacz", "żałuję"),
            "fear": ("boję", "strach", "przeraż"),
            "surprise": ("wow", "niesamowite", "nie wierzę"),
        }
        emotion_scores = {key: sum(term in lowered for term in terms) for key, terms in emotion_terms.items()}
        emotion = max(emotion_scores, key=emotion_scores.get) if max(emotion_scores.values(), default=0) else "neutral"
        sarcasm = bool(re.search(r"(/s|taa+k|jasne[,!.]|no pewnie|brawo[!.])", lowered))
        dangerous_negative = any(
            phrase in lowered
            for phrase in ("zabić", "zabij", "umrzeć", "samobój", "nienawidzę", "idiot", "głupi")
        )
        sentiment = labels.get(raw.get("label"), "neutral")
        confidence = float(raw.get("score") or 0.0)
        if dangerous_negative and sentiment == "neutral":
            sentiment = "negative"
            confidence = max(confidence, 0.8)
        return {
            "sentiment": sentiment,
            "confidence": confidence, "emotion": emotion,
            "sarcasm_detected": sarcasm,
        }

    def extract_moderation_entities(self, text: str) -> dict:
        lowered = text.casefold()
        contextual_targets = []
        target_groups = {
            "politicians": ("politykanci", "politycy", "rząd", "government"),
            "religious_groups": ("muzułmanie", "chrześcijanie", "żydzi"),
            "national_groups": ("polacy", "niemcy", "ukraińcy"),
        }
        for category, phrases in target_groups.items():
            matches = [phrase for phrase in phrases if phrase in lowered]
            if matches:
                contextual_targets.append({"category": category, "mentions": matches})
        result = {
            "usernames_mentioned": re.findall(r"(?<!\w)@[A-Za-z0-9_]{3,32}", text),
            "urls": re.findall(r"https?://[^\s<>()]+", text),
            "emails": re.findall(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text),
            "phone_numbers": re.findall(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)", text),
            "organizations": [], "locations": [], "persons": [],
            "contextual_targets": contextual_targets,
            "political_context": any(
                phrase in lowered
                for phrase in ("polityk", "rząd", "sejm", "senat", "government")
            ),
        }
        if self.entity_provider:
            try:
                entities = self.entity_provider.recognize("spacy", text)["entities"]
                mapping = {"ORG": "organizations", "GPE": "locations", "LOCATION": "locations", "PERSON": "persons"}
                for entity in entities:
                    target = mapping.get(entity["type"])
                    if target and entity["text"] not in result[target]:
                        result[target].append(entity["text"])
            except (ImportError, FileNotFoundError, RuntimeError, ValueError) as exc:
                result["ner_error"] = str(exc)
        return result

    @staticmethod
    def ensemble_decision(
        pii: dict, bielik: dict, qwen: dict, feedback: Optional[dict] = None,
        text: str = "",
    ) -> dict:
        bielik_vote = (
            "APPROVE" if bielik["label"] in ("clean", "skipped_pii")
            else "REJECT" if bielik["score"] >= 0.8
            else "FLAG_FOR_REVIEW"
        )
        qwen_vote = {
            "approve": "APPROVE", "reject": "REJECT", "review": "FLAG_FOR_REVIEW"
        }[qwen["recommended_action"]]
        privacy_vote = "REJECT" if pii["has_pii"] else "NO_OBJECTION"
        votes = {"privacy": privacy_vote, "bielik": bielik_vote, "qwen": qwen_vote}
        if feedback and feedback.get("confidence", 0.0) >= 0.7:
            votes["feedback"] = feedback["action"]
        if pii["has_pii"]:
            return {"action": "REJECT", "reason": "personally_identifiable_information", "consensus": "mandatory_pii", "votes": votes, "flag_account": False}
        if qwen["risk_level"] == "critical":
            reason = ";".join(qwen["categories"]) or "critical_safety_risk"
            return {"action": "REJECT", "reason": reason, "consensus": "critical_qwen", "votes": votes, "flag_account": True}

        categories = set(bielik.get("categories", []))
        lowered = text.casefold()
        political_context = any(
            phrase in lowered for phrase in ("polityk", "rząd", "sejm", "senat", "government")
        )
        explicit_threat = any(
            phrase in lowered
            for phrase in ("zabić", "zabij", "powinieneś umrzeć", "spalić", "bombę")
        )
        severe_categories = {"self_harm", "violence", "sexual"}
        if "self_harm" in categories and bielik["score"] >= 0.7:
            return {
                "action": "REJECT", "reason": "self_harm",
                "consensus": "policy_self_harm", "votes": votes,
                "flag_account": explicit_threat,
            }
        if "spam" in categories and bielik["score"] >= 0.8:
            return {
                "action": "REJECT", "reason": "spam",
                "consensus": "policy_spam", "votes": votes, "flag_account": False,
            }
        if political_context and not explicit_threat and not (categories & severe_categories):
            reasons = bielik.get("categories", []) + qwen.get("categories", [])
            return {
                "action": "FLAG_FOR_REVIEW",
                "reason": ";".join(dict.fromkeys(reasons)) or "political_opinion",
                "consensus": "political_review", "votes": votes,
                "flag_account": False,
            }
        if bielik["label"] != "clean" and bielik["score"] >= 0.8:
            return {
                "action": "REJECT",
                "reason": ";".join(bielik.get("categories", [])) or bielik["label"],
                "consensus": "high_confidence_bielik", "votes": votes,
                "flag_account": False,
            }

        decision_votes = [vote for vote in votes.values() if vote != "NO_OBJECTION"]
        counts = Counter(decision_votes)
        action, count = counts.most_common(1)[0]
        if list(counts.values()).count(count) > 1 or count < 2:
            action = "FLAG_FOR_REVIEW"
            consensus = "conflicting"
        else:
            consensus = "all_agree" if count == len(decision_votes) else "majority"
        reasons = bielik.get("categories", []) + qwen.get("categories", [])
        return {"action": action, "reason": ";".join(dict.fromkeys(reasons)) or "clean", "consensus": consensus, "votes": votes, "flag_account": False}

    def approve_content(self, content_id: str, moderator_id: str) -> str:
        self.repository.apply_content_action(
            content_id, "APPROVE", "approved", moderator_id, "approve_content"
        )
        return f"APPROVE content={content_id} moderator={moderator_id}"

    def reject_content(self, content_id: str, reason: str, moderator_id: str) -> str:
        self.repository.apply_content_action(
            content_id, "REJECT", reason, moderator_id, "reject_content"
        )
        return f"REJECT content={content_id} moderator={moderator_id} reason={reason}"

    def flag_for_human_review(self, content_id: str, priority: str, reason: str) -> str:
        self.repository.apply_content_action(
            content_id, "FLAG_FOR_REVIEW", reason, "bot", "flag_for_human_review"
        )
        return f"FLAG_FOR_REVIEW content={content_id} priority={priority} reason={reason}"

    def shadow_ban_user(self, user_id: str, duration_hours: int, reason: str) -> str:
        until = datetime.fromtimestamp(time.time() + max(1, int(duration_hours)) * 3600, timezone.utc).isoformat()
        self.repository.add_watchlist(user_id, reason, until)
        return f"SHADOW_BAN user={user_id} until={until}"

    def get_user_moderation_history(self, user_id: str) -> dict:
        row = self.repository.get_user_history(str(user_id))
        return {
            "user_id": row["user_id"], "violations_count": int(row["total_violations"]),
            "last_violation": row["last_violation_date"],
            "categories": list(filter(None, row["categories"].split(";"))),
            "risk_score": float(row["risk_score"]),
            "is_repeat_offender": row["is_repeat_offender"] == "True",
            "shadow_bans": int(row["shadow_bans"]),
        }

    def find_similar_violations(self, text: str, limit: int = 5) -> list:
        text = self._validate_text(text)
        rows = [row for row in self.repository.moderation_rows() if row["action"] != "APPROVE" and row["text"]]
        if not rows:
            return []
        limit = max(1, min(int(limit), 20))
        try:
            matrix = TfidfVectorizer(ngram_range=(1, 2), max_features=3000).fit_transform([text] + [row["text"] for row in rows])
        except ValueError:
            return []
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        ranked = sorted(zip(scores, rows), key=lambda item: item[0], reverse=True)[:limit]
        return [{"content_id": row["content_id"], "action": row["action"], "reason": row["reason"], "similarity": round(float(score), 4)} for score, row in ranked if score > 0]

    def add_to_watchlist(self, user_id: str, reason: str) -> str:
        self.repository.add_watchlist(str(user_id), reason)
        return f"WATCHLIST user={user_id} reason={reason}"

    def add_feedback(self, content_id: str, comment: str, correct_action: str) -> dict:
        correct_action = correct_action.strip().upper()
        if correct_action not in VALID_ACTIONS:
            raise ValueError(f"Poprawna decyzja musi być jedną z: {', '.join(VALID_ACTIONS)}.")
        content = self.repository.get_content(str(content_id))
        if not content:
            raise ValueError(f"Nie znaleziono content_id={content_id}.")
        row = {
            "content_id": content_id, "original_bot_decision": content["action"],
            "moderator_override": correct_action, "comment": comment.strip(),
            "text_sample": content["text"], "category": content["reason"],
            "confidence_before": f"{max(self._safe_float(content.get('model_bielik_score')), self._safe_float(content.get('model_qwen_score'))):.4f}",
            "confidence_after": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.repository.add_feedback(row)
        self._feedback_model = None
        if os.path.exists(self.repository.feedback_model_file):
            os.remove(self.repository.feedback_model_file)
        return row

    def train_on_feedback(self) -> dict:
        rows_by_content = {}
        for row in self.repository.feedback_rows():
            rows_by_content[row["content_id"]] = row
        rows = list(rows_by_content.values())
        self.repository.rewrite_feedback(rows)
        if len(rows) < 6:
            raise ValueError(
                "Trening wymaga co najmniej 6 różnych content_id z feedbackiem."
            )
        labels = [row["moderator_override"] for row in rows]
        if len(set(labels)) < 2:
            raise ValueError("Trening wymaga feedbacku z co najmniej 2 różnymi decyzjami.")
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("classifier", LogisticRegression(
                C=10.0, max_iter=500, class_weight="balanced", random_state=42
            )),
        ])
        pipeline.fit([row["text_sample"] for row in rows], labels)
        probabilities = pipeline.predict_proba([row["text_sample"] for row in rows])
        for row, row_probabilities in zip(rows, probabilities):
            row["confidence_after"] = f"{float(max(row_probabilities)):.4f}"
        self.repository.rewrite_feedback(rows)
        import joblib
        joblib.dump(pipeline, self.repository.feedback_model_file)
        self._feedback_model = pipeline
        return {"samples": len(rows), "classes": sorted(set(labels)), "model_path": self.repository.feedback_model_file}

    def _execute_decision(self, result: dict) -> list:
        action = result["action"]
        shadow = result["flag_account"]
        self.repository.append_moderation({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_id": result["content_id"], "user_id": result["user_id"],
            "username": result["username"], "text": result["text"],
            "model_bielik_decision": result["bielik"]["label"],
            "model_bielik_score": f"{result['bielik']['score']:.4f}",
            "model_qwen_decision": result["qwen"]["risk_level"],
            "model_qwen_score": f"{result['qwen']['confidence']:.4f}",
            "pii_detected": str(result["pii"]["has_pii"]),
            "pii_source": result["pii"]["source"],
            "sentiment": result["sentiment"]["sentiment"],
            "emotion": result["sentiment"]["emotion"],
            "action": action, "moderator_override": "", "reason": result["reason"],
            "appeal_filed": "False", "consensus": result["consensus"],
            "duration_seconds": f"{result['duration_seconds']:.4f}",
            "moderator_id": "bot", "tool_action": "",
        })
        executed = []
        if action == "APPROVE":
            executed.append(self.approve_content(result["content_id"], "bot"))
        elif action == "REJECT":
            executed.append(
                self.reject_content(result["content_id"], result["reason"], "bot")
            )
        else:
            priority = "high" if result["qwen"]["risk_level"] == "high" else "medium"
            executed.append(
                self.flag_for_human_review(result["content_id"], priority, result["reason"])
            )
        if shadow:
            executed.append(
                self.shadow_ban_user(result["user_id"], 24, result["reason"])
            )
        history_categories = (
            result["bielik"]["categories"]
            + result["qwen"]["categories"]
            + list(filter(None, result["reason"].split(";")))
        )
        history = self.repository.record_user_action(
            result["user_id"], result["username"], action,
            list(dict.fromkeys(history_categories)), shadow,
        )
        if history["is_repeat_offender"] == "True":
            executed.append(self.add_to_watchlist(result["user_id"], "repeat_offender"))
        return executed

    def _predict_feedback(self, text: str) -> Optional[dict]:
        unique_content_ids = {
            row["content_id"] for row in self.repository.feedback_rows()
        }
        if len(unique_content_ids) < 6:
            return None
        if self._feedback_model is None and os.path.exists(self.repository.feedback_model_file):
            import joblib
            self._feedback_model = joblib.load(self.repository.feedback_model_file)
        if self._feedback_model is None:
            return None
        probabilities = self._feedback_model.predict_proba([text])[0]
        index = int(probabilities.argmax())
        return {"action": str(self._feedback_model.classes_[index]), "confidence": float(probabilities[index])}

    @staticmethod
    def _safe_float(value) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _load_bielik_classifier():
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                BIELIK_MODEL, local_files_only=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                BIELIK_MODEL, local_files_only=True
            )
            return pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                top_k=None,
                framework="pt",
            )
        except Exception as exc:
            raise FileNotFoundError(
                f"Brak lokalnego modelu {BIELIK_MODEL}. Pobierz go jawnie przez `hf download {BIELIK_MODEL}` po `hf auth login`."
            ) from exc

    @staticmethod
    def _load_qwen_classifier():
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(QWEN_GUARD_MODEL, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(QWEN_GUARD_MODEL, local_files_only=True, torch_dtype="auto")
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model.to(device)
            model.eval()
        except Exception as exc:
            raise FileNotFoundError(
                f"Brak lokalnego modelu {QWEN_GUARD_MODEL}. Pobierz go jawnie przez `hf download {QWEN_GUARD_MODEL}`."
            ) from exc

        def classify(text: str):
            messages = [{"role": "user", "content": text}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer([rendered], return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=64, do_sample=False, return_dict_in_generate=True, output_scores=True)
            new_ids = generated.sequences[0][inputs["input_ids"].shape[-1]:]
            content = tokenizer.decode(new_ids, skip_special_tokens=True)
            token_confidences = []
            for token_id, scores in zip(new_ids, generated.scores):
                token_confidences.append(float(torch.softmax(scores[0], dim=-1)[token_id].item()))
            confidence = math.exp(sum(math.log(max(x, 1e-9)) for x in token_confidences) / len(token_confidences)) if token_confidences else 0.0
            return content, confidence
        return classify

    @staticmethod
    def _load_privacy_classifier():
        try:
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(PRIVACY_MODEL, local_files_only=True)
        except Exception as exc:
            raise FileNotFoundError(
                f"Brak lokalnego modelu {PRIVACY_MODEL}. Uruchom `hf download {PRIVACY_MODEL}`."
            ) from exc
        return PrivacyFilterWorker(model_dir)

    @staticmethod
    def _normalize_bielik_label(label: str) -> str:
        value = label.strip().lower().replace("_", "-")
        mapping = {"hate": "hate_speech", "vulgar": "toxic", "sex": "sexual", "crime": "violence", "self-harm": "self_harm", "safe": "clean", "none": "clean"}
        return mapping.get(value, value)

    @staticmethod
    def _regex_pii(text: str) -> list:
        patterns = {
            "private_email": r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
            "private_phone": r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)",
            "secret": r"(?i)\b(?:api[_-]?key|token|password|hasło)\s*[:=]\s*[^\s,;]{6,}",
            "account_number": r"\b(?:PL\s*)?\d{2}(?:[ -]?\d{4}){6}\b",
            "private_id": r"\b\d{11}\b",
        }
        entities = []
        occupied = set()
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                span = match.span()
                if span in occupied:
                    continue
                occupied.add(span)
                entities.append({"type": entity_type, "text": match.group(0), "score": 1.0})
        return entities

    @staticmethod
    def _spam_score(text: str) -> float:
        lowered = text.casefold()
        tokens = re.findall(r"\w+", lowered)
        marketing = (
            "kup teraz", "kliknij tutaj", "darmowa oferta", "gwarantowany zysk",
            "promocja", "zarób szybko", "limited offer",
        )
        score = 0.25 * sum(phrase in lowered for phrase in marketing)
        if len(re.findall(r"https?://", lowered)) >= 2:
            score += 0.5
        if tokens and Counter(tokens).most_common(1)[0][1] >= 4:
            score += 0.5
        if re.search(r"(.)\1{7,}", lowered):
            score += 0.25
        return min(1.0, score)

    @staticmethod
    def _tool_schema(name: str, description: str, properties: dict, required: list) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        key: {"type": value} for key, value in properties.items()
                    },
                    "required": required,
                },
            },
        }

    @staticmethod
    def _validate_text(text: str) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("Tekst moderacji nie może być pusty.")
        if len(text) > MAX_MODERATION_TEXT_LENGTH:
            raise ValueError(f"Tekst może mieć maksymalnie {MAX_MODERATION_TEXT_LENGTH} znaków.")
        return text
