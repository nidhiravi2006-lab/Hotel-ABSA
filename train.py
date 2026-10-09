"""Compare models, choose with validation data, and evaluate on held-out users."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from absa.core import ASPECTS, ROOT, normalize, polarity


def make_splits(frame, seed):
    groups = frame["user_id"].fillna("").astype(str).to_numpy()
    groups = np.array([g if g else "unknown-" + str(i) for i, g in enumerate(groups)])
    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    development, test = next(first.split(frame, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    tr, va = next(second.split(development, groups=groups[development]))
    splits = {"train": development[tr], "validation": development[va], "test": test}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not set(groups[splits[a]]) & set(groups[splits[b]])
        assert not set(frame.iloc[splits[a]].text_hash) & set(frame.iloc[splits[b]].text_hash)
    return splits


def score(y, predictions, task):
    if task == "rating":
        continuous = np.clip(np.asarray(predictions, dtype=float), 1, 5)
        predicted = np.clip(np.floor(continuous + 0.5), 1, 5).astype(int)
        return {
            "accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)),
            "mae": float(mean_absolute_error(y, continuous)),
            "polarity_accuracy_from_rating": float(accuracy_score(polarity(y), polarity(predicted))),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[1, 2, 3, 4, 5]).tolist(),
        }
    p, r, f, support = precision_recall_fscore_support(y, predictions, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1, 2]).tolist(),
        "classes": {name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])} for i, name in enumerate(["Negative", "Neutral", "Positive"])},
    }


def train(data, seed=42, max_features=40000):
    started = time.time()
    frame = pd.read_csv(data).reset_index(drop=True)
    required = {"review_id", "title", "text", "user_id", "text_hash", "overall"}
    if not required.issubset(frame.columns):
        raise ValueError("CSV needs columns: " + ", ".join(sorted(required)))
    texts = (frame.title.fillna("") + " " + frame.text.fillna("")).map(normalize).to_numpy()
    if frame.text_hash.duplicated().any():
        raise ValueError("Remove duplicate review text before splitting.")
    splits = make_splits(frame, seed)
    features = {
        "counts12": CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=max_features),
        "counts23": CountVectorizer(ngram_range=(2, 3), min_df=2, max_df=0.2, max_features=max_features),
        "tfidf12": TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=max_features, sublinear_tf=True),
    }
    matrices = {}
    for key, vectorizer in features.items():
        vectorizer.fit(texts[splits["train"]])
        matrices[key] = {s: vectorizer.transform(texts[idx]) for s, idx in splits.items()}
    configurations = [
        ("Naive Bayes counts 1-2", "counts12", lambda: MultinomialNB(alpha=0.5)),
        ("SVM counts 1-2", "counts12", lambda: LinearSVC(C=0.1, class_weight="balanced", dual="auto", max_iter=5000, random_state=seed)),
        ("SVM counts 2-3", "counts23", lambda: LinearSVC(C=0.1, class_weight="balanced", dual="auto", max_iter=5000, random_state=seed)),
        ("SVM TF-IDF 1-2", "tfidf12", lambda: LinearSVC(C=1, class_weight="balanced", dual="auto", max_iter=5000, random_state=seed)),
    ]
    bundle = {"version": 1, "vectorizers": features, "models": {}, "seed": seed}
    report = {
        "split_method": "GroupShuffleSplit by user_id, approximately 60/20/20; disjoint reviewer groups and deduplicated text",
        "seed": seed, "split_counts": {s: len(idx) for s, idx in splits.items()},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "csv_sha256": hashlib.file_digest(open(data, "rb"), "sha256").hexdigest(),
        "features": {k: len(v.vocabulary_) for k, v in features.items()},
        "selection": "Highest validation macro-F1 separately for polarity and rating. Fixed hyperparameters. Test data never chooses a model.",
        "targets": {}, "unavailable": [],
    }
    comparisons = []
    test_rows = []
    for target in ["overall", *ASPECTS]:
        ratings = pd.to_numeric(frame[target], errors="coerce").to_numpy() if target in frame else np.full(len(frame), np.nan)
        masks = {s: np.isfinite(ratings[idx]) & (ratings[idx] >= 1) & (ratings[idx] <= 5) & (ratings[idx] % 1 == 0) for s, idx in splits.items()}
        if min(sum(m) for m in masks.values()) < 20 or len(np.unique(ratings[splits["train"]][masks["train"]])) < 2:
            report["unavailable"].append(target)
            continue
        bundle["models"][target] = {}
        report["targets"][target] = {"support": {s: int(sum(m)) for s, m in masks.items()}}
        for task in ["polarity", "rating"]:
            labels = polarity(ratings) if task == "polarity" else ratings
            ys = {s: labels[idx][masks[s]].astype(int) for s, idx in splits.items()}
            candidates = list(configurations)
            if task == "rating":
                candidates.append(("Ridge regression TF-IDF", "tfidf12", lambda: Ridge(alpha=10, solver="lsqr")))
            candidates.append(("Majority baseline", "counts12", lambda: DummyClassifier(strategy="most_frequent")))
            records = []
            for name, key, factory in candidates:
                xs = {s: matrices[key][s][masks[s]] for s in splits}
                estimator = factory().fit(xs["train"], ys["train"])
                val = score(ys["validation"], estimator.predict(xs["validation"]), task)
                # Test comparisons are descriptive. Selection uses validation only.
                test_pred = estimator.predict(xs["test"])
                test = score(ys["test"], test_pred, task)
                records.append((name, key, estimator, val, test, test_pred))
                comparisons.append({"target": target, "task": task, "model": name, "validation_macro_f1": val["macro_f1"], "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"], "test_mae": test.get("mae")})
            chosen = max(records, key=lambda r: r[3]["macro_f1"])
            name, key, estimator, val, test, test_pred = chosen
            bundle["models"][target][task] = {"name": name, "feature": key, "estimator": estimator}
            report["targets"][target][task] = {"selected_model": name, "validation": val, "test": test, "majority_baseline_test": records[-1][4]}
            for review_id, y, p in zip(frame.iloc[splits["test"]].review_id.to_numpy()[masks["test"]], ys["test"], test_pred):
                test_rows.append({"review_id": review_id, "target": target, "task": task, "truth": int(y), "prediction": float(p)})
            print(f"{target:16} {task:8} {name:25} test accuracy={test['accuracy']:.3f} macro-F1={test['macro_f1']:.3f}", flush=True)
    report["elapsed_seconds"] = round(time.time() - started, 2)
    aspects = [v for k, v in report["targets"].items() if k != "overall"]
    report["aspect_summary"] = {task: {"mean_accuracy": float(np.mean([v[task]["test"]["accuracy"] for v in aspects])), "mean_macro_f1": float(np.mean([v[task]["test"]["macro_f1"] for v in aspects]))} for task in ["polarity", "rating"]}
    report["limitation"] = "Different dataset and protocol from the supplied study. Missing service/business-service labels. Keyword evidence is heuristic and unscored. Whole-review aspect ratings do not annotate individual sentences. No verified hotel-independent split. No calibrated probability estimates."
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT / "models" / "hotel_absa.joblib", compress=3)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(comparisons).to_csv(ROOT / "results" / "model_comparison.csv", index=False)
    pd.DataFrame(test_rows).to_csv(ROOT / "results" / "test_predictions.csv", index=False)
    assignments = np.empty(len(frame), dtype=object)
    for s, idx in splits.items(): assignments[idx] = s
    pd.DataFrame({"review_id": frame.review_id, "split": assignments}).to_csv(ROOT / "results" / "split_assignments.csv", index=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "reviews.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=40000)
    args = parser.parse_args()
    train(args.data, args.seed, args.max_features)
