# Aspect-based Sentiment Analysis on Hotel Reviews

**UE24CS352A - Machine Learning Mini-Project, PES University**  
**Section J**  
Nidhi - PES1UG24CS582  
Nikhil.P - PES1UG24CS583

This project predicts Positive, Neutral, or Negative sentiment and a 1-5 rating for individual hotel aspects using supervised text classifiers. It includes a local browser demo, command-line prediction, a reproducible model comparison, a two-page PDF write-up, and presentation slides.

## Set up and run the demo

Use Python 3.12. Clone this repository, or extract the complete project ZIP supplied separately. Open a terminal inside the resulting project folder.

```powershell
git clone https://github.com/nidhiravi2006-lab/Hotel-ABSA.git
cd Hotel-ABSA
```

Skip cloning if you already extracted the ZIP. Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

**GitHub clone:** the dataset sample and fitted model are excluded from Git. Download the data and train before starting the app:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --sample-size 20000
.\.venv\Scripts\python.exe train.py
```

**Complete ZIP:** the sample and fitted model are already included, so skip data preparation and training for an immediate demo.

Start the app:

```powershell
.\.venv\Scripts\python.exe app.py --port 8001
```

Open **http://127.0.0.1:8001**. Press Ctrl+C in the terminal to stop the demo. The version lock matches the environment used to fit the included model. `requirements.txt` supports fresh training with compatible library versions, but fitted scikit-learn models should load using matching versions.

macOS/Linux equivalents:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python scripts/prepare_data.py --sample-size 20000
.venv/bin/python train.py
.venv/bin/python app.py --port 8001
```

Skip the data preparation and training commands when using the complete ZIP.

Command-line demo:

```powershell
.\.venv\Scripts\python.exe predict.py --text "The room was dirty, but the location was convenient."
```

## Data and scope

- Dataset: [jniimi/tripadvisor-review-rating](https://huggingface.co/datasets/jniimi/tripadvisor-review-rating), a public redistribution of reviews associated with Li, Ott, and Cardie (2013).
- Source file: 201,295 English reviews. The local experiment uses a reproducible random sample of **20,000** reviews after preprocessing and duplicate removal, with seed 42.
- Available labels: overall rating, rooms, location, value, cleanliness, and sleep quality.
- The schema supports the supplied paper's seven aspects. **Service and business service have no labels in this redistribution**, so the demo displays "No training labels" and never trains them using overall ratings.
- Rating mapping: 1-2 = Negative, 3 = Neutral, 4-5 = Positive. Missing aspect ratings remain missing.
- Reviewer identifiers define disjoint train/validation/test groups, with approximately 60/20/20 proportions. Verified hotel grouping is unavailable, so this does not establish hotel-independent generalization.
- `data/provenance.json` records source URLs, label coverage, rating distributions, and SHA-256 hashes. `results/split_assignments.csv` records the actual split.

The uploaded Yu paper uses a different dataset and reports its own results. Its 70-75% rating accuracy and 85-90% polarity accuracy are **reference-study findings**, not this project's results.

## Reproduce training and evaluation

The complete ZIP includes the sampled CSV and fitted model for an immediate local demonstration. GitHub excludes both; prepare the sample first using the setup instructions above. The 221 MB original Parquet file is excluded from the ZIP and from Git.

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

To recreate the CSV from the public source:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --sample-size 20000
.\.venv\Scripts\python.exe train.py
```

The source download requires internet access. If you already have the Parquet file, add `--input "path\to\source.parquet"`. A publisher may update the source, so compare hashes in `provenance.json`; the included CSV is the fixed input for the reported run.

## Method

1. Combine title and body, decode HTML entities, remove tags, lowercase, and normalize spaces. Keep negation words such as "not".
2. Remove identical normalized reviews before splitting. Separate reviewer groups before fitting text features.
3. Fit word count features with unigrams/bigrams, counts with bigrams/trigrams, and TF-IDF with unigrams/bigrams on training text only. Use `min_df=2` and a 40,000-feature cap.
4. Compare Multinomial Naive Bayes, balanced Linear SVM variants, and a majority-class baseline. Compare Ridge regression for rating prediction only.
5. Fit an independent three-class sentiment model and a separate rating model for each available target. Select each using **validation macro-F1**, with fixed hyperparameters. Do not choose models using test scores.
6. Evaluate selected models on held-out reviewer groups. Report accuracy, macro-F1, class precision/recall, confusion matrices, rating MAE, and baseline scores.

The corrected demo sends **matching aspect excerpts** to each selected aspect classifier. For example, a sleep complaint is classified separately from praise for the room or location. The overall model still receives the complete review. Selecting "Show predictions for aspects without a keyword mention" uses complete-review estimates for those unmentioned aspects, labeled explicitly in the page and JSON. All matching excerpts enter the model, although the page displays at most four per aspect. Extraction is a keyword heuristic, not a learned aspect detector. Sentiment and rating models can disagree because they learn separate targets.

**Evaluation scope:** training and `train.py` evaluation still use full review text. The reported 81.1% mean aspect accuracy belongs to that original baseline. It is not a measured accuracy for the corrected excerpt-based demo. Models trained on full reviews can behave differently on excerpts, and independently annotated aspect sentences are needed for a stronger evaluation.

## Mixed-review correction

The review below originally produced an incorrect Positive, 5/5 sleep prediction:

> Our room was spotless and spacious. We did not sleep well in the uncomfortable bed. The location was perfect for walking around town, and the hotel was excellent value for money.

The corrected demo uses "We did not sleep well in the uncomfortable bed" for sleep quality. The same fitted model now returns **Negative, 1/5**. This is a checked example, not an overall accuracy claim. Fifteen tests include this case, additional sleep complaints, positive sleep among other complaints, and the API response. No phrase-specific sentiment override or model retraining was used.

After updating the files, stop a running demo with Ctrl+C, restart it using the same command and port, and refresh the browser. Use the updated submission files: [PDF write-up](submission/Hotel_ABSA_Writeup_Updated.pdf) and [presentation](submission/Hotel_ABSA_Presentation_Updated.pptx).

## Results

Read [RESULTS.md](RESULTS.md) for measured scores and interpretation. Machine-readable outputs:

- `results/metrics.json`: selected models, validation and test metrics, confusion matrices, baseline comparison, runtime, and library versions.
- `results/model_comparison.csv`: every candidate's validation macro-F1 and descriptive test results.
- `results/test_predictions.csv`: predictions and true labels for the selected models, without review text.
- `results/split_assignments.csv`: review IDs and split membership.

The browser demo reads these same metrics rather than hardcoding performance.

## Files

```text
absa/core.py                 labels, normalization, evidence, model inference
scripts/prepare_data.py      safe Parquet download, sampling, provenance
train.py                    feature fitting, model comparison, evaluation
predict.py                  command-line prediction
app.py                      local HTTP server and API
web/index.html              browser demo
tests/test_project.py       data, leakage, inference, and API checks
data/reviews.csv             fixed sample used in the reported experiment
models/hotel_absa.joblib     fitted model for the local demo
results/                    reproducible metrics and prediction outputs
submission/                 PDF write-up, slides, demo guide, and Q&A notes
```

## Limitations and conclusion

Review-level ratings supervise text sentiment indirectly. A guest can rate an aspect without mentioning it. Full-review classifiers can confuse mixed opinions. Excerpt routing reduces interference from unrelated sentences but can miss implied aspects, attach one clause to multiple aspects, or retain conflicting opinions. These full-review-trained models and keyword extraction lack sentence-level ground-truth evaluation. Positive reviews dominate, so accuracy alone is insufficient; inspect macro-F1 and neutral/negative recall. No claim of calibrated confidence is made.

This is a working classical machine learning baseline for five aspect targets, with explicit missing-label handling for the remaining two. Extend it with service/business-service labels, independently annotated aspect sentences, stronger aspect extraction, hotel-independent splits, and error analysis before claiming broad reliability.

## GitHub repository and faculty access

Repository: [nidhiravi2006-lab/Hotel-ABSA](https://github.com/nidhiravi2006-lab/Hotel-ABSA).

1. Keep this repository **Private**, as required by the assignment.
2. In a clone of this repository, commit and push your actual changes:

```powershell
git add .
git commit -m "Update hotel review aspect sentiment project"
git push origin main
```

3. In the repository, open **Settings**, then **Collaborators**, and add your teammate plus the faculty/TA accounts provided by your course. They must accept the invitations. See [GitHub's collaborator instructions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository).
4. Confirm that the repository is private, the README is visible, and faculty access works. Submit the repository URL together with the PDF and presentation.
5. Let each member commit their actual contributions using their own Git identity. Do not invent contribution history. Keep the repository updated after changes.

The `.gitignore` excludes the raw data, sample CSV, virtual environment, and serialized model. A fresh clone must run `prepare_data.py` and `train.py`. To give faculty an immediate demo, share the complete local ZIP through your submission portal. If your course wants the small sample/model in the private repository too, add them explicitly with `git add -f data/reviews.csv models/hotel_absa.joblib` after checking the source terms and file sizes.

## Assignment checklist

- Private GitHub repository with faculty/TA access and setup instructions.
- Working code and a live demo.
- PDF write-up covering problem, dataset, approach, implementation, and conclusions.
- Presentation and preparation for methodology/code Q&A.
- Actual individual contributions.

The guideline says "One-Page Write-up" in its heading but asks for a concise two-page summary in its body. This package follows the detailed two-page instruction. The stated submission deadline is **October 10, 2026, 11:59 PM**. Reviews run **October 5-9, 2026**.

## References

1. Yangyang Yu. *Aspect-based Sentiment Analysis on Hotel Reviews*. Stanford CS229, Spring 2016. Supplied `032 (1).pdf` and poster `032.pdf`. [Paper](https://cs229.stanford.edu/proj2016spr/report/032.pdf).
2. Jiwei Li, Myle Ott, and Claire Cardie. *Identifying Manipulated Offerings on Review Portals*. EMNLP 2013. [ACL Anthology](https://aclanthology.org/D13-1199/).
3. Junichiro Niimi. *Hotel Review Dataset (English)*, 2024. [Dataset card](https://huggingface.co/datasets/jniimi/tripadvisor-review-rating).
4. scikit-learn documentation: [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html), [F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html), and [data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
