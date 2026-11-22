# Fake News Detector

A text classifier that labels a news article as **Fake** or **Real**, served through a Streamlit web app. The app screens live headlines, classifies and saves articles you paste in, and reports the measured accuracy of the trained pipeline.

## Features

| Page | Description |
| --- | --- |
| **Live news** | Fetches headlines for a topic from Google News and labels each one. Filter by keyword and by prediction. |
| **Add news** | Paste a headline and article text to get a verdict. Saved articles can be filtered by label and keyword. |
| **Accuracy** | Test-set metrics, per-class report, confusion matrix, and the pipeline settings. |

## Results

Measured on a locked test set of 7,817 articles from the ISOT corpus. The test set was not used to choose the model.

| Metric | Value |
| --- | --- |
| Accuracy | 99.35% |
| Precision (Fake) | 98.86% |
| Recall (Fake) | 99.72% |
| F1 (Fake) | 99.29% |
| Precision (Real) | 99.76% |
| Recall (Real) | 99.03% |
| Macro F1 | 99.34% |
| Matthews correlation | 0.9869 |
| ROC-AUC | 0.9998 |

These numbers come from a benchmark with known shortcuts. See [Limitations](#limitations) before drawing conclusions about new news.

## How it works

1. **Data.** The ISOT Fake News corpus has two files: `Fake.csv` (articles from outlets flagged by PolitiFact and Wikipedia) and `True.csv` (Reuters wire copy). The notebook downloads both.
2. **Preprocessing.** Decode HTML entities, remove URLs and Reuters datelines, lowercase, drop digits and punctuation, and remove English stop words. Negations such as "not" are kept. The title is included twice, and each article is capped at 500 tokens.
3. **Features.** TF-IDF over unigrams and bigrams, with up to 60,000 features.
4. **Model.** A linear SVM (C = 2.0). Multinomial Naive Bayes and logistic regression were also tuned on the validation split and compared against it.
5. **Decision threshold.** Chosen on the validation split to maximize the selection score S = ½ · accuracy + ½ · fake-class recall. The threshold is −0.1785 on the SVM decision score. Scores at or above the threshold are labelled Fake.

The full procedure, including the audit for shortcut features, is in [`Fake_News_Detection_NLP.ipynb`](Fake_News_Detection_NLP.ipynb).

## Project structure

```
fake_news_app/
├── app.py                              # Streamlit pages
├── ui.py                               # Light-mode styling and HTML components
├── pipeline.py                         # Preprocessing and inference (mirrors the notebook)
├── news_feed.py                        # Live headline fetch and parsing
├── news_store.py                       # SQLite store for saved articles
├── Fake_News_Detection_NLP.ipynb       # Training notebook
├── artifacts/
│   ├── best_fake_news_pipeline.joblib  # Trained model, saved by the notebook
│   ├── metrics.json                    # Test-set results shown on the Accuracy page
│   └── test_model_comparison.csv       # Optional: frozen models on the test set
├── .streamlit/config.toml              # Light theme
├── requirements.txt
└── .gitignore
```

The `data/` folder is created on first run and holds the saved articles in `news.db`.

## Getting started

**Requirements:** Python 3.10 or later. The app was tested with Python 3.12 and Streamlit 1.45. An internet connection is needed for the live feed and for the one-time NLTK stop-word download.

1. Open a terminal in the `fake_news_app` folder.

2. Create and activate a virtual environment (recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   On macOS or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`.

3. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Make sure the model is in place. The file `artifacts/best_fake_news_pipeline.joblib` is required for the Live news and Add news pages. If it is missing, train it with the notebook:
   1. Upload `Fake_News_Detection_NLP.ipynb` to [Google Colab](https://colab.research.google.com/).
   2. Select a CPU runtime, then choose **Run all**. The first run downloads about 120 MB.
   3. Download `best_fake_news_pipeline.joblib` from the Colab Files panel and place it in `artifacts/`. Optionally, also download `test_model_comparison.csv`.

5. Start the app:

   ```powershell
   streamlit run app.py
   ```

   Streamlit opens the app at `http://localhost:8501`.

## Configuration

| Setting | How to change it | Default |
| --- | --- | --- |
| Model file | Set the `FAKE_NEWS_MODEL` environment variable to the path of the `.joblib` file | `artifacts/best_fake_news_pipeline.joblib` |
| Theme | Edit `.streamlit/config.toml` | Light |
| Live feed topic | Type a topic on the Live news page | `world news` |

Example with a custom model path:

```powershell
$env:FAKE_NEWS_MODEL = "D:\models\best_fake_news_pipeline.joblib"
streamlit run app.py
```

## Troubleshooting

- **"Model file not found".** Check that the `.joblib` file is in `artifacts/`, or set `FAKE_NEWS_MODEL`. The Accuracy page works without the model.
- **"Could not load the live feed".** Check your internet connection, proxy, or firewall. The app uses an unofficial Google News RSS endpoint, which can change or rate-limit requests. Use the Refresh button after a short wait.
- **NLTK `LookupError` or download failure.** The app needs the NLTK `stopwords` corpus. Download it once while online:

  ```powershell
  python -c "import nltk; nltk.download('stopwords')"
  ```

- **`InconsistentVersionWarning` when loading the model.** The model was saved with a different scikit-learn version. Install the version used in the notebook runtime, or ignore the warning if predictions look correct.
- **Port 8501 is in use.** Run `streamlit run app.py --server.port 8502`.

## Limitations

- **Shortcut learning.** In ISOT, real articles are Reuters wire copy and fake articles come from other outlets. The model can learn that writing style instead of the difference between true and false content. Accuracy on new sources is likely lower than the test results show.
- **Headline-only predictions.** The Live news page classifies headlines, but the model was trained on full article text. Treat these predictions as a screening signal.
- **Not a fact-checker.** The model estimates how similar an article is to the training data. It does not verify claims, sources, or events.

## Privacy

Articles you add are stored locally in `data/news.db` and are not sent anywhere. The only outbound request is the live headline query to Google News.

## Retraining

Run the notebook again to produce a new `best_fake_news_pipeline.joblib`, then replace the file in `artifacts/`. Update `artifacts/metrics.json` with the new test results so the Accuracy page stays accurate.

## Data and citation

The ISOT Fake News Dataset is provided by the ISOT Research Lab at the University of Victoria. If you use it in academic work, cite:

> Ahmed, H., Traore, I., & Saad, S. (2017). Detection of online fake news using n-gram analysis and machine learning techniques. In *Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments*.

Dataset page: <https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/>
