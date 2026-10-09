# Review and live demonstration guide

Nidhi - PES1UG24CS582 and Nikhil.P - PES1UG24CS583, Section J.

## Suggested 5-minute live demonstration

1. Start `python app.py` from the project folder and open `http://127.0.0.1:8000` before the review. Use the virtual environment from the README.
2. Explain the problem: a single rating hides differences among room quality, location, cleanliness, value, and sleep quality.
3. Select the mixed example, then **Analyze review**. Discuss the actual predictions, including any mistakes. The examples are illustrative text, not test examples or guaranteed outputs.
4. Compare the positive and poor-stay examples. Point to matching excerpts. Explain that mentioned aspect models receive their excerpts, while the overall model uses the complete review. Try the checked mixed review below.
5. Show "No training labels" for service/business service. Show "No keyword mention" for other absent aspects. Missing data is not Neutral sentiment.
6. Expand the measured-performance table. Explain held-out reviewers, validation-based selection, macro-F1, and the majority baseline. These scores measure the original full-review baseline; excerpt inference needs its own evaluation.
7. Show `train.py`, `absa/core.py`, and `results/metrics.json` if asked. Use `predict.py` as an alternative demo if a browser is unavailable.

Use `Hotel_ABSA_Writeup_Updated.pdf` and `Hotel_ABSA_Presentation_Updated.pptx` for submission. The older versions describe the original demo.

Checked correction example:

> Our room was spotless and spacious. We did not sleep well in the uncomfortable bed. The location was perfect for walking around town, and the hotel was excellent value for money.

Sleep quality changes from Positive, 5/5 with full-review inference to **Negative, 1/5** with excerpt inference. The same trained classifier produces both results. Praise for unrelated aspects no longer enters the sleep prediction. This example and the 15 passing tests do not establish overall excerpt-inference accuracy.

## Questions you should be ready to answer

**What is aspect-based sentiment analysis?**  
It estimates sentiment about particular features, such as rooms or location, rather than assigning only one sentiment to the complete review.

**How did you use the uploaded files?**  
The Yu paper and poster informed the supervised learning approach, aspect list, n-gram comparisons, imbalance handling, and rating-to-polarity mapping. The course PDF defined the deliverables. The supplied RecurJail slides informed the presentation style, not the hotel project's facts or team composition.

**Are you reproducing Yu's reported 85-90% accuracy?**  
No. We use an accessible different dataset and a held-out-reviewer protocol. Our report and demo show scores measured by our own code. The paper's scores remain reference findings.

**Why only five trained aspects?**  
The accessible redistribution contains five aspect rating columns. The other two have no ground-truth labels. Our schema supports them, but training and prediction remain unavailable until labeled data is added.

**What is the ground truth?**  
Guest-provided aspect ratings, mapped to sentiment: 1-2 negative, 3 neutral, 4-5 positive. These are review-level weak labels, not manually annotated aspect sentences.

**Why use two models per target?**  
One learns three sentiment classes directly, and another predicts a rating. Direct polarity training need not make the same mistakes as five-class rating prediction. The outputs can disagree.

**Why use n-grams?**  
Phrases can represent negation and local context, such as "not clean" or "good location". Bigrams/trigrams need repeated examples and can be sparse.

**Why preserve stop words?**  
Removing "not" can reverse meaning. The experiment keeps words and limits features using document-frequency thresholds instead.

**What does TF-IDF add?**  
It weights terms by frequency within a review and rarity across training reviews. Counts can also work well because common sentiment terms are informative. We compare both rather than assuming one always wins.

**How do you handle imbalance?**  
Linear SVM uses inverse-frequency class weights. We compare against a majority-class baseline and report macro-F1 plus class recall. Naive Bayes remains a separate unweighted comparison.

**Why macro-F1?**  
It averages class F1 values equally. A classifier that predicts Positive for everyone can have high accuracy on a skewed dataset but poor macro-F1.

**How do you avoid leakage?**  
We deduplicate normalized text, separate reviewer groups, fit vocabulary/IDF only on training text, and select models with validation scores. The test set does not determine the selected configuration.

**Does this demonstrate performance on unseen hotels?**  
No. Reliable hotel grouping is unavailable in the redistribution. The split supports unseen-reviewer evaluation. A hotel-independent study requires verified hotel identifiers.

**Does the model identify exact opinion spans?**  
Keywords route matching clauses into each aspect classifier. This is approximate extraction rather than a trained span detector. The heuristic lacks sentence-level accuracy evaluation, and models trained on full reviews can behave differently on excerpts.

**Why did a negative sleep sentence originally get 5/5?**  
The classifier received the whole mixed review. Positive room, location, and value terms overwhelmed the sleep complaint. The corrected demo classifies matching sleep excerpts separately. It keeps negation and does not hardcode a result for this sentence.

**Why compare Ridge regression?**  
Ratings are ordered numbers. Ridge provides a regularized linear regression comparison, whereas the classifiers treat star ratings as five classes. We clip regression predictions to 1-5 and round them for discrete accuracy.

**What is the main limitation?**  
Review ratings may describe aspects absent from the text. Matching excerpts reduce interference from unrelated sentences, but keyword extraction and models trained on full reviews still make mistakes. Exact aspect sentence labels would support better training and evaluation. Reported accuracy covers the original full-review baseline.

**What did each member contribute?**  
Describe each member's actual work. Agree on responsibilities, commit your own contributions, and ensure both members can explain the code. The package does not claim a contribution split on your behalf.
