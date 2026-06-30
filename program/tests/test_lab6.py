import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch


PROGRAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROGRAM_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_DIR)

import main
from providers.moderation_provider import (
    MODERATION_FIELDS,
    ModerationProvider,
    ModerationRepository,
)


class FakeSentiment:
    def predict(self, method, text):
        if "rozczarowany" in text.lower():
            return {"label": "negatywny", "score": 1.0}
        if "uwielbiam" in text.lower():
            return {"label": "pozytywny", "score": 1.0}
        return {"label": "neutralny", "score": 0.0}


class FakeEntities:
    def recognize(self, method, text):
        return {"entities": [{"text": "Warszawa", "type": "GPE"}] if "Warszawa" in text else []}


def clean_bielik(text):
    return [[
        {"label": "hate", "score": 0.02},
        {"label": "vulgar", "score": 0.01},
        {"label": "sex", "score": 0.01},
        {"label": "crime", "score": 0.01},
        {"label": "self-harm", "score": 0.01},
    ]]


def safe_qwen(text):
    return {"safety": "Safe", "categories": [], "confidence": 0.95}


class FakeUser:
    id = 123
    username = "tester"


class FakeMessage:
    message_id = 456

    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


class FakeUpdate:
    def __init__(self):
        self.message = FakeMessage()
        self.effective_user = FakeUser()


class FakeContext:
    def __init__(self, args):
        self.args = args


def moderation_row(content_id, user_id="u1", text="tekst", action="REJECT", reason="toxic"):
    row = {field: "" for field in MODERATION_FIELDS}
    row.update({
        "timestamp": "2026-06-29T10:00:00+00:00", "content_id": str(content_id),
        "user_id": str(user_id), "text": text, "action": action,
        "reason": reason, "consensus": "majority", "duration_seconds": "0.1",
    })
    return row


class ModerationModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = ModerationRepository(self.temp.name)
        self.provider = ModerationProvider(
            sentiment_provider=FakeSentiment(), entity_provider=FakeEntities(),
            repository=self.repo, bielik_classifier=clean_bielik,
            qwen_classifier=safe_qwen, privacy_classifier=False,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_negative_sentiment_does_not_reject_clean_content(self):
        result = self.provider.moderate(
            "Kupiłem ten produkt i jestem rozczarowany", "1", "u1", persist=False
        )
        self.assertEqual(result["sentiment"]["sentiment"], "negative")
        self.assertEqual(result["action"], "APPROVE")

    def test_pii_regex_forces_reject_and_reports_source(self):
        self.provider._bielik_classifier = lambda text: self.fail("Bielik should be skipped for PII")
        self.provider._qwen_classifier = lambda text: self.fail("Qwen should be skipped for PII")
        result = self.provider.moderate(
            "Napisz do mnie: jan@example.com", "2", "u1", persist=False
        )
        self.assertEqual(result["action"], "REJECT")
        self.assertEqual(result["consensus"], "mandatory_pii")
        self.assertEqual(result["pii"]["source"], "regex_fallback")

    def test_qwen_critical_forces_reject_and_account_flag(self):
        self.provider._qwen_classifier = lambda text: {
            "safety": "Unsafe", "categories": ["Suicide & Self-Harm"], "confidence": 0.97,
        }
        result = self.provider.moderate("powinieneś się zabić", "3", "u1", persist=False)
        self.assertEqual(result["action"], "REJECT")
        self.assertTrue(result["flag_account"])

    def test_critical_persist_executes_log_history_and_watchlist(self):
        self.provider._qwen_classifier = lambda text: {
            "safety": "Unsafe", "categories": ["Violent"], "confidence": 0.97,
        }
        result = self.provider.moderate("groźba przemocy", "33", "u1", persist=True)
        self.assertEqual(self.repo.get_content("33")["action"], "REJECT")
        self.assertEqual(self.provider.get_user_moderation_history("u1")["violations_count"], 1)
        self.assertEqual(self.repo.watchlist()[0]["user_id"], "u1")
        self.assertEqual(result["similar_cases"], [])

    def test_conflicting_votes_go_to_review(self):
        decision = self.provider.ensemble_decision(
            {"has_pii": False},
            {"label": "toxic", "score": 0.6, "categories": ["toxic"]},
            {"recommended_action": "reject", "risk_level": "high", "categories": []},
            None,
        )
        self.assertEqual(decision["action"], "FLAG_FOR_REVIEW")
        self.assertEqual(decision["consensus"], "conflicting")

    def test_self_harm_policy_rejects_from_point_seven(self):
        decision = self.provider.ensemble_decision(
            {"has_pii": False},
            {"label": "self_harm", "score": 0.75, "categories": ["self_harm"]},
            {"recommended_action": "review", "risk_level": "medium", "categories": ["Violent"]},
            None,
            text="powinieneś się zabić",
        )
        self.assertEqual(decision["action"], "REJECT")
        self.assertEqual(decision["consensus"], "policy_self_harm")
        self.assertTrue(decision["flag_account"])

    def test_general_political_opinion_is_sent_to_review(self):
        decision = self.provider.ensemble_decision(
            {"has_pii": False},
            {"label": "hate_speech", "score": 0.82, "categories": ["hate_speech"]},
            {"recommended_action": "reject", "risk_level": "high", "categories": ["Unethical Acts"]},
            None,
            text="Ci politykanci to wszystko złodzieje",
        )
        self.assertEqual(decision["action"], "FLAG_FOR_REVIEW")
        self.assertEqual(decision["consensus"], "political_review")

    def test_detected_spam_is_rejected_even_when_qwen_is_safe(self):
        decision = self.provider.ensemble_decision(
            {"has_pii": False},
            {"label": "spam", "score": 1.0, "categories": ["spam"]},
            {"recommended_action": "approve", "risk_level": "safe", "categories": []},
        )
        self.assertEqual(decision["action"], "REJECT")
        self.assertEqual(decision["consensus"], "policy_spam")

    def test_bielik_labels_are_normalized(self):
        self.provider._bielik_classifier = lambda text: [[
            {"label": "hate", "score": 0.91},
            {"label": "self-harm", "score": 0.83},
        ]]
        result = self.provider.classify_bielik_guard("test")
        self.assertEqual(result["label"], "hate_speech")
        self.assertEqual(result["categories"], ["hate_speech", "self_harm"])

    def test_qwen_text_output_is_parsed(self):
        self.provider._qwen_classifier = lambda text: (
            "Safety: Controversial\nCategories: Politically Sensitive Topics", 0.81
        )
        result = self.provider.classify_qwen_guard("test")
        self.assertEqual(result["risk_level"], "medium")
        self.assertEqual(result["recommended_action"], "review")

    def test_spam_rule_augments_bielik_categories(self):
        result = self.provider.classify_bielik_guard(
            "PROMOCJA kup teraz kliknij tutaj https://a.pl https://b.pl"
        )
        self.assertEqual(result["label"], "spam")
        self.assertIn("spam", result["categories"])

    def test_seven_tool_schemas_and_dispatch_are_available(self):
        names = [row["function"]["name"] for row in self.provider.tool_schemas]
        self.assertEqual(len(names), 7)
        self.assertIn("find_similar_violations", names)
        result = self.provider.call_tool(
            "approve_content", {"content_id": "1", "moderator_id": "m1"}
        )
        self.assertIn("APPROVE", result)
        with self.assertRaisesRegex(ValueError, "Nieznane narzędzie"):
            self.provider.call_tool("unknown", {})

    def test_entities_mix_regex_and_ner(self):
        result = self.provider.extract_moderation_entities(
            "@tester z Warszawa: https://example.com, a@example.com"
        )
        self.assertEqual(result["locations"], ["Warszawa"])
        self.assertIn("@tester", result["usernames_mentioned"])
        self.assertIn("a@example.com", result["emails"])

    def test_contextual_targets_do_not_fake_named_entities(self):
        result = self.provider.extract_moderation_entities("Ci politykanci to złodzieje")
        self.assertEqual(result["persons"], [])
        self.assertEqual(result["contextual_targets"][0]["category"], "politicians")
        self.assertTrue(result["political_context"])

    def test_model_and_regex_pii_are_merged(self):
        provider = ModerationProvider(
            repository=self.repo,
            privacy_classifier=lambda text: [{
                "entity_group": "private_person", "word": "Jan Kowalski", "score": 0.99,
            }],
        )
        result = provider.detect_private_info("Jan Kowalski, jan@example.com")
        self.assertEqual(result["source"], "model+regex")
        self.assertEqual({row["type"] for row in result["entities"]}, {"private_person", "private_email"})


class ModerationRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = ModerationRepository(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_content_status_and_analytics(self):
        self.repo.append_moderation(moderation_row("10", action="REJECT", reason="hate_speech"))
        self.repo.append_moderation(moderation_row("11", action="APPROVE", reason="clean"))
        self.assertEqual(self.repo.get_content("10")["action"], "REJECT")
        analytics = self.repo.analytics(today_only=False)
        self.assertEqual(analytics["total"], 2)
        self.assertEqual(analytics["actions"]["APPROVE"], 1)

    def test_today_analytics_contains_percentages_and_consensus(self):
        row = moderation_row("today", action="REJECT", reason="spam")
        row["timestamp"] = datetime.now(timezone.utc).isoformat()
        row["consensus"] = "policy_spam"
        self.repo.append_moderation(row)
        analytics = self.repo.analytics()
        self.assertEqual(analytics["total"], 1)
        self.assertEqual(analytics["percentages"]["REJECT"], 100.0)
        self.assertEqual(analytics["consensus"]["policy_spam"], 1)
        self.assertNotEqual(analytics["period"], "all")

    def test_third_rejection_marks_repeat_offender(self):
        row = None
        for _ in range(3):
            row = self.repo.record_user_action("u1", "jan", "REJECT", ["toxic"])
        self.assertEqual(row["total_violations"], "3")
        self.assertEqual(row["is_repeat_offender"], "True")

    def test_watchlist_is_updated_not_duplicated(self):
        self.repo.add_watchlist("u1", "critical")
        self.repo.add_watchlist("u1", "repeat_offender")
        self.assertEqual(len(self.repo.watchlist()), 1)
        self.assertEqual(self.repo.watchlist()[0]["reason"], "repeat_offender")

    def test_similar_violations_returns_ranked_case(self):
        self.repo.append_moderation(moderation_row("1", text="obraźliwy toksyczny komentarz"))
        provider = ModerationProvider(repository=self.repo)
        rows = provider.find_similar_violations("toksyczny komentarz")
        self.assertEqual(rows[0]["content_id"], "1")
        self.assertGreater(rows[0]["similarity"], 0)

    def test_feedback_training_and_prediction(self):
        provider = ModerationProvider(repository=self.repo)
        examples = [
            ("dobry legalny wpis", "APPROVE"), ("świetna wiadomość", "APPROVE"),
            ("normalny komentarz", "APPROVE"), ("toksyczny atak", "REJECT"),
            ("obraźliwa groźba", "REJECT"), ("niebezpieczna treść", "REJECT"),
        ]
        for index, (text, action) in enumerate(examples):
            self.repo.append_moderation(moderation_row(index, text=text))
            provider.add_feedback(str(index), "korekta", action)
        trained = provider.train_on_feedback()
        self.assertEqual(trained["samples"], 6)
        self.assertTrue(os.path.exists(trained["model_path"]))
        self.assertTrue(all(row["confidence_after"] for row in self.repo.feedback_rows()))
        prediction = provider._predict_feedback("świetna wiadomość")
        self.assertIn(prediction["action"], ("APPROVE", "REJECT"))
        self.assertGreaterEqual(prediction["confidence"], 0.7)

    def test_feedback_replaces_duplicate_content_id(self):
        provider = ModerationProvider(repository=self.repo)
        self.repo.append_moderation(moderation_row("1", text="przykład"))
        provider.add_feedback("1", "pierwsza", "APPROVE")
        provider.add_feedback("1", "korekta", "REJECT")
        rows = self.repo.feedback_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["moderator_override"], "REJECT")
        self.assertTrue(rows[0]["confidence_before"])

    def test_direct_tool_call_persists_action(self):
        provider = ModerationProvider(repository=self.repo)
        provider.call_tool("reject_content", {
            "content_id": "tool-1", "reason": "spam", "moderator_id": "m1",
        })
        row = self.repo.get_content("tool-1")
        self.assertEqual(row["action"], "REJECT")
        self.assertEqual(row["moderator_id"], "m1")
        self.assertEqual(row["tool_action"], "reject_content")

    def test_feedback_training_rejects_too_few_samples(self):
        provider = ModerationProvider(repository=self.repo)
        with self.assertRaisesRegex(ValueError, "co najmniej 6"):
            provider.train_on_feedback()


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_moderate_success_uses_telegram_identifiers(self):
        update = FakeUpdate()
        with tempfile.TemporaryDirectory() as directory:
            provider = ModerationProvider(
                sentiment_provider=FakeSentiment(), entity_provider=FakeEntities(),
                repository=ModerationRepository(directory),
                bielik_classifier=clean_bielik, qwen_classifier=safe_qwen,
                privacy_classifier=False,
            )
            with patch.object(main, "moderation_provider", provider):
                await main.handle_moderate(update, FakeContext(['"Uwielbiam produkt"']))
        self.assertTrue(any("MODERACJA #456" in reply for reply in update.message.replies))
        self.assertTrue(any("Użytkownik: 123" in reply for reply in update.message.replies))

    async def test_moderate_rejects_missing_text(self):
        update = FakeUpdate()
        await main.handle_moderate(update, FakeContext([]))
        self.assertIn("❌", update.message.replies[-1])

    async def test_policy_check_rejects_missing_text(self):
        update = FakeUpdate()
        await main.handle_mod_policy_check(update, FakeContext([]))
        self.assertIn("/mod_policy_check", update.message.replies[-1])

    async def test_status_reports_unknown_id(self):
        update = FakeUpdate()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            main, "moderation_provider", ModerationProvider(repository=ModerationRepository(directory))
        ):
            await main.handle_mod_status(update, FakeContext(["404"]))
        self.assertIn("Nie znaleziono", update.message.replies[-1])

    async def test_history_validates_arguments(self):
        update = FakeUpdate()
        await main.handle_mod_history(update, FakeContext([]))
        self.assertIn("/mod_history", update.message.replies[-1])

    async def test_analytics_returns_empty_report(self):
        update = FakeUpdate()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            main, "moderation_provider", ModerationProvider(repository=ModerationRepository(directory))
        ):
            await main.handle_mod_analytics(update, FakeContext([]))
        self.assertIn("Łącznie: 0", update.message.replies[-1])

    async def test_add_feedback_validates_arguments(self):
        update = FakeUpdate()
        await main.handle_mod_add_feedback(update, FakeContext([]))
        self.assertIn("/mod_add_feedback", update.message.replies[-1])

    async def test_watchlist_reports_empty(self):
        update = FakeUpdate()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            main, "moderation_provider", ModerationProvider(repository=ModerationRepository(directory))
        ):
            await main.handle_mod_watchlist(update, FakeContext([]))
        self.assertIn("pusta", update.message.replies[-1])

    async def test_train_reports_missing_feedback(self):
        update = FakeUpdate()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            main, "moderation_provider", ModerationProvider(repository=ModerationRepository(directory))
        ):
            await main.handle_mod_train_on_feedback(update, FakeContext([]))
        self.assertIn("co najmniej 6", update.message.replies[-1])

    async def test_mod_help_lists_commands(self):
        update = FakeUpdate()
        await main.handle_mod_help(update, FakeContext([]))
        self.assertIn("/moderate", update.message.replies[-1])
        self.assertIn("/mod_train_on_feedback", update.message.replies[-1])


if __name__ == "__main__":
    unittest.main()
