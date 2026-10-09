"""Checks for leakage, missing labels, inference contracts, and the demo API."""
import json
from pathlib import Path
from http.server import ThreadingHTTPServer
import threading
import unittest
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

from absa.core import Analyzer, ROOT, evidence, normalize, polarity
from train import make_splits, score
from app import create_handler


class DataContracts(unittest.TestCase):
    def test_missing_rating_never_becomes_neutral(self):
        values = polarity([1, 2, 3, 4, 5, np.nan])
        np.testing.assert_array_equal(values[:5], [0, 0, 1, 2, 2])
        self.assertTrue(np.isnan(values[5]))

    def test_negation_survives_cleaning(self):
        self.assertEqual(normalize("<b>Room</b> was NOT clean &amp; quiet."), "room was not clean & quiet.")

    def test_repeat_reviewers_stay_together(self):
        frame = pd.DataFrame({"user_id": [f"guest-{i // 3}" for i in range(120)], "text_hash": [f"text-{i}" for i in range(120)]})
        splits = make_splits(frame, 42)
        owner = {}
        for name, indices in splits.items():
            for user in frame.iloc[indices].user_id:
                if user in owner: self.assertEqual(owner[user], name)
                owner[user] = name
        self.assertEqual(sum(map(len, splits.values())), len(frame))

    def test_macro_f1_penalizes_positive_only_predictions(self):
        metric = score(np.array([0, 1, 2, 2]), np.array([2, 2, 2, 2]), "polarity")
        self.assertEqual(metric["accuracy"], 0.5)
        self.assertLess(metric["macro_f1"], 0.3)

    def test_aspect_evidence_respects_contrast_clause(self):
        text = "The room was clean but the location was inconvenient."
        self.assertEqual(evidence(text, "location"), ["the location was inconvenient"])


@unittest.skipUnless((ROOT / "models" / "hotel_absa.joblib").exists(), "Train the model first")
class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = Analyzer()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(cls.analyzer))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_unavailable_aspects_are_explicit(self):
        result = self.analyzer.analyze("The staff were helpful and the internet was fast.")
        by_name = {r["aspect"]: r for r in result["aspects"]}
        self.assertEqual(by_name["service"]["status"], "unavailable")
        self.assertIsNone(by_name["service"]["sentiment"])
        self.assertEqual(by_name["business_service"]["status"], "unavailable")
        self.assertEqual(by_name["cleanliness"]["status"], "not_mentioned")

    def test_inference_is_deterministic_and_ratings_bounded(self):
        text = "The room was spotless and the bed was comfortable."
        result = self.analyzer.analyze(text, True)
        self.assertEqual(result, self.analyzer.analyze(text, True))
        for r in result["aspects"]:
            if r["status"] == "predicted":
                self.assertIn(r["sentiment"], ["Positive", "Neutral", "Negative"])
                self.assertIn(r["rating"], range(1, 6))

    def test_sleep_complaint_is_not_overwhelmed_by_other_aspects(self):
        complaint = "We did not sleep well in the uncomfortable bed."
        review = "Our room was spotless and spacious. " + complaint + " The location was perfect for walking around town, and the hotel was excellent value for money."
        for include_unmentioned in [False, True]:
            rows = {r["aspect"]: r for r in self.analyzer.analyze(review, include_unmentioned)["aspects"]}
            sleep = rows["sleep_quality"]
            self.assertEqual(sleep["sentiment"], "Negative")
            self.assertIn(sleep["rating"], [1, 2])
            self.assertEqual(sleep["prediction_source"], "aspect_excerpts")
            self.assertEqual(sleep["evidence"], [complaint.rstrip(".")])
            self.assertEqual(rows["rooms"]["sentiment"], "Positive")
            self.assertEqual(rows["location"]["sentiment"], "Positive")
            self.assertEqual(rows["value"]["sentiment"], "Positive")

    def test_irrelevant_praise_does_not_change_sleep_prediction(self):
        complaints = [
            "We did not sleep well in the uncomfortable bed.",
            "The mattress was uncomfortable and street noise kept us awake.",
            "We could not sleep because of the noise and the hard mattress.",
        ]
        for complaint in complaints:
            alone = next(r for r in self.analyzer.analyze(complaint)["aspects"] if r["aspect"] == "sleep_quality")
            mixed = next(r for r in self.analyzer.analyze("The room was spacious and spotless. " + complaint + " The location was perfect and the price was excellent.")["aspects"] if r["aspect"] == "sleep_quality")
            self.assertEqual((alone["sentiment"], alone["rating"]), (mixed["sentiment"], mixed["rating"]))
            self.assertEqual(mixed["sentiment"], "Negative")
            self.assertIn(mixed["rating"], [1, 2])

    def test_positive_sleep_is_not_overwhelmed_by_other_complaints(self):
        review = "The room was filthy and cramped. We slept well in the comfortable bed. The hotel was overpriced and the location was inconvenient."
        sleep = next(r for r in self.analyzer.analyze(review)["aspects"] if r["aspect"] == "sleep_quality")
        self.assertEqual(sleep["sentiment"], "Positive")
        self.assertIn(sleep["rating"], [4, 5])

    def test_unmentioned_fallback_has_explicit_source(self):
        text = "The room was spotless."
        hidden = next(r for r in self.analyzer.analyze(text)["aspects"] if r["aspect"] == "sleep_quality")
        inferred = next(r for r in self.analyzer.analyze(text, True)["aspects"] if r["aspect"] == "sleep_quality")
        self.assertEqual(hidden["status"], "not_mentioned")
        self.assertIsNone(hidden["prediction_source"])
        self.assertEqual(inferred["prediction_source"], "full_review")
        self.assertEqual(inferred["evidence"], [])

    def test_api_preserves_sleep_complaint_in_mixed_review(self):
        text = "Our room was spotless and spacious. We did not sleep well in the uncomfortable bed. The location was perfect for walking around town, and the hotel was excellent value for money."
        request = urllib.request.Request(self.base + "/api/analyze", data=json.dumps({"text": text, "include_unmentioned": True}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
        self.assertEqual(result["inference_version"], 2)
        sleep = next(r for r in result["aspects"] if r["aspect"] == "sleep_quality")
        self.assertEqual(sleep["sentiment"], "Negative")
        self.assertIn(sleep["rating"], [1, 2])

    def test_api_and_page(self):
        with urllib.request.urlopen(self.base + "/") as r:
            self.assertIn(b"Hotel Review Lens", r.read())
        request = urllib.request.Request(self.base + "/api/analyze", data=json.dumps({"text": "The room was dirty."}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as r:
            self.assertEqual(len(json.load(r)["aspects"]), 7)

    def test_api_rejects_empty_and_malformed_requests(self):
        for body in [b'{"text":""}', b'[]', b'broken']:
            request = urllib.request.Request(self.base + "/api/analyze", data=body, headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as context:
                urllib.request.urlopen(request)
            self.assertEqual(context.exception.code, 400)

    def test_actual_split_has_no_reviewer_or_text_overlap(self):
        reviews = pd.read_csv(ROOT / "data" / "reviews.csv").fillna("")
        joined = reviews.merge(pd.read_csv(ROOT / "results" / "split_assignments.csv"), on="review_id")
        self.assertEqual(len(joined), len(reviews))
        for column in ["user_id", "text_hash"]:
            nonempty = joined.loc[joined[column] != ""]
            self.assertTrue(nonempty.groupby(column)["split"].nunique().le(1).all())


if __name__ == "__main__":
    unittest.main()
