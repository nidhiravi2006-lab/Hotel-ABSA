# Measured experiment results

Input: 20,000 reviews. Train: 12,007; validation: 3,986; test: 4,007. Reviewer groups remain disjoint.

Models below were selected using validation macro-F1, not test accuracy. These scores evaluate the original full-review baseline. The corrected demo routes matching excerpts to each aspect model; its accuracy needs separate evaluation.

| Target | Sentiment accuracy | Sentiment macro-F1 | Majority accuracy | Exact rating accuracy | Rating MAE |
|---|---:|---:|---:|---:|---:|
| Overall | 84.5% | 0.713 | 77.3% | 64.4% | 0.404 |
| Rooms | 80.9% | 0.655 | 73.5% | 56.4% | 0.521 |
| Location | 81.7% | 0.513 | 86.9% | 62.6% | 0.477 |
| Value | 77.1% | 0.617 | 72.1% | 51.7% | 0.591 |
| Cleanliness | 86.4% | 0.624 | 82.6% | 60.6% | 0.476 |
| Sleep quality | 79.5% | 0.599 | 78.1% | 55.1% | 0.562 |

Five-aspect mean sentiment accuracy: **81.1%**. Mean macro-F1: **0.602**. Mean exact rating accuracy: **57.3%**. These are unweighted aspect means, excluding overall.

Majority sentiment accuracy averages **78.7%**. The location model has lower accuracy than its majority baseline but improves macro-F1. Macro-F1 gives equal importance to Negative, Neutral, and Positive classes.

| Target | Selected sentiment model | Selected rating model |
|---|---|---|
| Overall | Naive Bayes counts 1-2 | Naive Bayes counts 1-2 |
| Rooms | SVM TF-IDF 1-2 | Naive Bayes counts 1-2 |
| Location | Naive Bayes counts 1-2 | Naive Bayes counts 1-2 |
| Value | SVM TF-IDF 1-2 | Naive Bayes counts 1-2 |
| Cleanliness | SVM TF-IDF 1-2 | SVM TF-IDF 1-2 |
| Sleep quality | Naive Bayes counts 1-2 | Naive Bayes counts 1-2 |

Service and business service: no labels in the source redistribution, so no measured predictions.

See `results/metrics.json` for per-class recall, confusion matrices, and validation scores. `model_comparison.csv` includes all candidates as descriptive comparisons; it does not choose the model on test data.

The supplied paper reports its own historical results on another dataset. Those values are not results from this run. Keyword evidence and mention filtering have no sentence-level performance evaluation. A held-out hotel evaluation has not been established.

Training runtime on this machine: 105.66 seconds. Libraries: python 3.12.10, sklearn 1.9.1, pandas 3.0.6, numpy 2.5.3, joblib 1.6.0.
