import os
import sys
import unittest


PROGRAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROGRAM_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_DIR)


@unittest.skipUnless(
    os.getenv("RUN_LAB6_INTEGRATION") == "1",
    "Ustaw RUN_LAB6_INTEGRATION=1, aby uruchomić lokalne checkpointy Lab6.",
)
class RealModelAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import main
        cls.provider = main.moderation_provider

    def check_case(self, text, expected_action):
        result = self.provider.moderate(
            text, "integration", "integration-user", persist=False
        )
        self.assertEqual(result["action"], expected_action, result)
        return result

    def test_required_semantic_scenarios(self):
        self.check_case("Uwielbiam ten produkt, najlepszy zakup!", "APPROVE")
        negative = self.check_case(
            "Kupiłem ten produkt i jestem rozczarowany", "APPROVE"
        )
        self.assertEqual(negative["sentiment"]["sentiment"], "negative")

        toxic = self.check_case(
            "Jesteś głupszy niż cegła, powinieneś się zabić", "REJECT"
        )
        self.assertEqual(toxic["sentiment"]["emotion"], "anger")

        self.check_case(
            "Ci politykanci to wszystko złodzieje! Wszyscy!!",
            "FLAG_FOR_REVIEW",
        )
        self.check_case(
            "PROMOCJA! Kup teraz, kliknij tutaj: https://a.pl https://b.pl",
            "REJECT",
        )

        pii = self.check_case(
            "My name is Alice Smith, email alice@example.com, phone +48 123 456 789",
            "REJECT",
        )
        self.assertIn(pii["pii"]["source"], ("model", "model+regex"))
        self.assertTrue(any(row["type"] == "private_person" for row in pii["pii"]["entities"]))


if __name__ == "__main__":
    unittest.main()
