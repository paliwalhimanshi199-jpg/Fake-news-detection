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

## Appendix: notebook code

The core code from `Fake_News_Detection_NLP.ipynb`, in run order. Each block is complete. Cells 22 and 24 (the term table and the error-case display) are left out because they only inspect results and do not change the pipeline.

### Cell 3, 1. Setup

```python
import gc
import hashlib
import html
import importlib
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

def require(package, import_name=None):
    name = import_name or package
    try:
        return importlib.import_module(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        return importlib.import_module(name)

require("nltk")

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from IPython.display import display
from joblib import dump
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import ConfusionMatrixDisplay

SEED = 42
TEST_SIZE = 0.20
VAL_FRACTION = 0.20
MAX_TOKENS = 500
TITLE_CAP = 40
TITLE_REPEATS = 2
MIN_LETTERS = 40
UNIGRAM_FEATURES = 30_000
BIGRAM_FEATURES = 60_000
NEGATION = {"no", "not", "nor", "never"}
REUTERS_TOKENS = {"reuters", "reuter"}

np.random.seed(SEED)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "font.size": 11,
})
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 100)

COLOR_REAL = "#1b7f5a"
COLOR_FAKE = "#b23a48"
COLOR_INK = "#1f4e79"

print("pandas", pd.__version__)
print("numpy", np.__version__)
print("scikit-learn", __import__("sklearn").__version__)
print("nltk", nltk.__version__)
```

### Cell 5, 2. Metric helpers

```python
def metric_dict(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_real": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_real": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_real": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def selection_score(metrics):
    return 0.5 * metrics["accuracy"] + 0.5 * metrics["recall_fake"]


def positive_scores(model, matrix):
    # Higher score means the article looks more fake.
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
    return np.asarray(model.decision_function(matrix), dtype=float)


def choose_threshold(y_true, scores, kind):
    # Pick the validation threshold that maximises the selection score.
    scores = np.asarray(scores, dtype=float)
    if kind == "proba":
        default = 0.5
        grid = np.linspace(0.15, 0.85, 36)
    else:
        default = 0.0
        lo, hi = np.quantile(scores, [0.02, 0.98])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.min(scores)), float(np.max(scores))
        if lo == hi:
            grid = np.array([default], dtype=float)
        else:
            grid = np.linspace(lo, hi, 41)
    grid = np.unique(np.concatenate([np.asarray(grid, dtype=float), [default]]))

    best_rank = None
    best_threshold = default
    best_metrics = metric_dict(y_true, (scores >= default).astype(np.int64))
    for threshold in grid:
        metrics = metric_dict(y_true, (scores >= threshold).astype(np.int64))
        rank = (
            selection_score(metrics),
            metrics["recall_fake"],
            metrics["accuracy"],
            -abs(float(threshold) - default),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def evaluate_fitted(model, matrix, y_true):
    if list(model.classes_) != [0, 1]:
        raise RuntimeError(f"Unexpected class order: {list(model.classes_)}")
    kind = "proba" if hasattr(model, "predict_proba") else "decision"
    scores = positive_scores(model, matrix)
    default = 0.5 if kind == "proba" else 0.0
    default_metrics = metric_dict(y_true, (scores >= default).astype(np.int64))
    threshold, tuned = choose_threshold(y_true, scores, kind)
    return {
        "kind": kind,
        "threshold": threshold,
        "metrics": tuned,
        "default_metrics": default_metrics,
        "selection_score": selection_score(tuned),
    }


_demo = metric_dict(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
assert abs(_demo["accuracy"] - 0.75) < 1e-9
assert abs(_demo["recall_fake"] - 0.5) < 1e-9
assert abs(selection_score(_demo) - 0.625) < 1e-9
_t, _m = choose_threshold(
    np.array([0, 0, 1, 1]),
    np.array([0.1, 0.2, 0.8, 0.9]),
    "proba",
)
assert _m["accuracy"] == 1.0 and _m["recall_fake"] == 1.0
print("Metric self-check passed.")
print("Fake is label 1. Real is label 0. Selection score = 0.5 * accuracy + 0.5 * fake recall.")
```

### Cell 7, 3. Dataset collection

```python
FAKE_URLS = [
    "https://huggingface.co/datasets/koliskos/fake_news/resolve/main/Fake.csv?download=true",
    "https://huggingface.co/scdong/Fake-news-classifier/resolve/main/Fake.csv?download=true",
]
TRUE_URLS = [
    "https://huggingface.co/datasets/koliskos/fake_news/resolve/main/True.csv?download=true",
    "https://huggingface.co/scdong/Fake-news-classifier/resolve/main/True.csv?download=true",
]
ZIP_URL = "https://www.dropbox.com/s/35kt1gsji0u2lip/ISOT%20Fake%20News%20Dataset.zip?dl=1"
DATA_DIR = "data"


def download_file(url, dest, timeout=300):
    tmp = dest + ".part"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; FakeNewsResearch/1.0)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, open(tmp, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            next_report = 10 * 1024 * 1024
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    if total:
                        print(f"  {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
                    else:
                        print(f"  {downloaded / 1e6:.0f} MB", flush=True)
                    next_report += 10 * 1024 * 1024
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def file_is_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
        return False
    with open(path, "rb") as handle:
        head = handle.read(400).lower()
    if b"<html" in head or b"<!doctype" in head or head.startswith(b"pk"):
        return False
    return b"title" in head and b"text" in head


def ensure_csv(filename, urls):
    dest = os.path.join(DATA_DIR, filename)
    if file_is_csv(dest):
        print(f"Using cached {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
        return dest
    errors = []
    for url in urls:
        print(f"Downloading {filename}")
        print(f"  {url}")
        try:
            download_file(url, dest)
            if file_is_csv(dest):
                print(f"Saved {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
                return dest
            errors.append(f"{url}: downloaded file is not the ISOT CSV")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
        if os.path.exists(dest):
            os.remove(dest)
    raise RuntimeError(" | ".join(errors))


def extract_isot_zip(zip_path):
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError("The fallback download was not a zip archive.")
    found = set()
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            base = os.path.basename(name)
            if base not in {"Fake.csv", "True.csv"}:
                continue
            target = os.path.join(DATA_DIR, base)
            with archive.open(name) as source, open(target, "wb") as handle:
                shutil.copyfileobj(source, handle)
            found.add(base)
    if found != {"Fake.csv", "True.csv"}:
        raise RuntimeError(f"The archive did not contain both CSV files. Found {sorted(found)}.")


def ensure_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")
    if file_is_csv(fake_path) and file_is_csv(true_path):
        print("ISOT files already present.")
        return fake_path, true_path
    try:
        ensure_csv("Fake.csv", FAKE_URLS)
        ensure_csv("True.csv", TRUE_URLS)
    except Exception as exc:
        print("Direct CSV mirrors failed.")
        print(exc)
        print("Trying the ISOT Dropbox archive...")
        zip_path = os.path.join(DATA_DIR, "ISOT_Fake_News_Dataset.zip")
        download_file(ZIP_URL, zip_path)
        extract_isot_zip(zip_path)
    if not (file_is_csv(fake_path) and file_is_csv(true_path)):
        raise RuntimeError("Could not obtain readable Fake.csv and True.csv files.")
    return fake_path, true_path


def read_news_csv(path):
    last_error = None
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not decode {path}: {last_error}")


def load_half(path, label):
    frame = read_news_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame = frame.loc[:, ~frame.columns.str.contains(r"^unnamed", case=False, regex=True)]
    missing = {"title", "text"} - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing {sorted(missing)}. Columns: {list(frame.columns)}")
    keep = [column for column in ("title", "text", "subject", "date") if column in frame.columns]
    frame = frame[keep].copy()
    frame["title"] = frame["title"].fillna("").astype(str)
    frame["text"] = frame["text"].fillna("").astype(str)
    if "subject" in frame.columns:
        frame["subject"] = frame["subject"].fillna("(missing)").astype(str)
    frame["label"] = int(label)
    return frame


fake_path, true_path = ensure_dataset()
fake_df = load_half(fake_path, label=1)
true_df = load_half(true_path, label=0)
corpus = pd.concat([fake_df, true_df], ignore_index=True)

print(f"Fake.csv rows: {len(fake_df):,}")
print(f"True.csv rows: {len(true_df):,}")
if 20_000 <= len(fake_df) <= 26_000 and 18_000 <= len(true_df) <= 24_000:
    print("Row counts match the published ISOT release (about 23.5k fake and 21.4k real).")
else:
    print("Row counts differ from the usual ISOT release. Check that both files downloaded fully.")
print("Columns:", list(corpus.columns))
print("Label map: 1 = Fake, 0 = Real")
```

### Cell 9, 4. Audit before modelling

```python
def funnel_report(step, frame):
    n_fake = int((frame["label"] == 1).sum())
    n_real = int((frame["label"] == 0).sum())
    print(f"{step}: {len(frame):,} articles ({n_fake:,} fake, {n_real:,} real)")


def document_key(frame):
    blob = (
        frame["title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        + "\n"
        + frame["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    return blob.map(lambda text: hashlib.md5(text.encode("utf-8")).hexdigest())


print("Legacy files as loaded")
funnel_report("loaded", corpus)

has_reuters = (corpus["title"] + " " + corpus["text"]).str.contains(r"\breuters?\b", case=False, regex=True)
print()
print("Share of articles whose raw title or body contains 'Reuters':")
for label, name in ((0, "Real"), (1, "Fake")):
    rate = has_reuters[corpus["label"] == label].mean()
    print(f"  {name}: {rate:.1%}")

if "subject" in corpus.columns:
    subject_table = pd.crosstab(
        corpus["subject"],
        corpus["label"].map({0: "Real", 1: "Fake"}),
    )
    print()
    print("Articles by subject and class:")
    display(subject_table)
    real_col = subject_table["Real"] if "Real" in subject_table.columns else 0
    fake_col = subject_table["Fake"] if "Fake" in subject_table.columns else 0
    overlap = int(((real_col > 0) & (fake_col > 0)).sum())
    print(f"Subject values present in both classes: {overlap}")
    if overlap == 0:
        print("Each subject belongs to one class, so subject is excluded from every feature vector.")

news = corpus.copy()
letters = (news["title"] + " " + news["text"]).str.count(r"[A-Za-z]")
news = news.loc[letters >= MIN_LETTERS].copy()
funnel_report(f"after removing items with fewer than {MIN_LETTERS} letters", news)

news["doc_key"] = document_key(news)
label_counts = news.groupby("doc_key")["label"].nunique()
conflicts = set(label_counts[label_counts > 1].index)
if conflicts:
    news = news.loc[~news["doc_key"].isin(conflicts)].copy()
funnel_report("after removing documents that appear with both labels", news)

before_dedup = len(news)
news = news.drop_duplicates(subset="doc_key", keep="first").copy()
print(f"Exact duplicate documents removed: {before_dedup - len(news):,}")
funnel_report("modelling set", news)
news = news.drop(columns=["doc_key"])
news = news.reset_index(drop=True)

news["word_count"] = news["text"].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
counts = news["label"].value_counts().sort_index()
axes[0].bar(
    ["Real", "Fake"],
    [counts.get(0, 0), counts.get(1, 0)],
    color=[COLOR_REAL, COLOR_FAKE],
)
axes[0].set_ylabel("Articles")
axes[0].set_title("Class balance in the modelling set")
for label, name, color in ((0, "Real", COLOR_REAL), (1, "Fake", COLOR_FAKE)):
    axes[1].hist(
        news.loc[news["label"] == label, "word_count"].clip(upper=2500),
        bins=40,
        alpha=0.55,
        label=name,
        color=color,
    )
axes[1].set_xlabel("Words in the raw body (display clipped at 2,500)")
axes[1].set_ylabel("Articles")
axes[1].set_title("Article length by class")
axes[1].legend()
fig.tight_layout()
fig.savefig("class_balance_and_length.png", dpi=150, bbox_inches="tight")
plt.show()

print()
print(news.groupby(news["label"].map({0: "Real", 1: "Fake"}))["word_count"].describe().round(1))

kept = [column for column in ("title", "text", "label") if column in news.columns]
news = news[kept].copy()
del corpus
gc.collect()
print()
print("Columns passed forward:", list(news.columns))
print("Subject, date, and source tags are not among them.")
```

### Cell 11, 5. Text preprocessing

```python
def ensure_nltk():
    for resource, path in (
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


ensure_nltk()

STOP = set(nltk.corpus.stopwords.words("english")) - NEGATION
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()
_lemma_cache = {}
_stem_cache = {}

URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+", re.IGNORECASE)
DATELINE_RE = re.compile(
    r"\b[A-Za-z][A-Za-z .'\-]{0,60}\(\s*reuters\s*\)\s*[-:\u2013\u2014]?\s*",
    re.IGNORECASE,
)
NON_LETTER_RE = re.compile(r"[^a-z\s]+")


def lemmatize(token):
    cached = _lemma_cache.get(token)
    if cached is None:
        cached = LEMMATIZER.lemmatize(LEMMATIZER.lemmatize(token, pos="n"), pos="v")
        _lemma_cache[token] = cached
    return cached


def stem_token(token):
    cached = _stem_cache.get(token)
    if cached is None:
        cached = STEMMER.stem(token)
        _stem_cache[token] = cached
    return cached


def base_tokens(text):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = html.unescape(text)
    text = URL_RE.sub(" ", text)
    text = DATELINE_RE.sub(" ", text)
    text = text.lower()
    text = NON_LETTER_RE.sub(" ", text)
    return [
        token
        for token in text.split()
        if len(token) >= 2 and token not in STOP and token not in REUTERS_TOKENS
    ]


def tokens_from_raw(title, body):
    title_tokens = base_tokens(title)[:TITLE_CAP]
    body_tokens = base_tokens(body)
    return (title_tokens * TITLE_REPEATS + body_tokens)[:MAX_TOKENS]


def render(tokens, variant):
    if variant == "lemma":
        produced = [lemmatize(token) for token in tokens]
    elif variant == "stem":
        produced = [stem_token(token) for token in tokens]
    elif variant == "stopwords":
        produced = list(tokens)
    else:
        raise ValueError(f"Unknown representation: {variant}")
    return " ".join(
        token
        for token in produced
        if len(token) >= 2 and token not in STOP and token not in REUTERS_TOKENS
    )


_example = " ".join(tokens_from_raw(
    "Reuters bulletin",
    "WASHINGTON (Reuters) - Markets rose 2% after the vote, officials said.",
))
assert "reuters" not in _example and "washington" not in _example
assert "markets" in _example and "vote" in _example
assert "not" in " ".join(tokens_from_raw("", "The bill did not pass."))
print("Preprocessing self-check passed.")
print("Example:", _example)
```

### Cell 12, 5. Text preprocessing

```python
VARIANT_COLUMNS = {
    "stopwords": "clean_stop",
    "lemma": "clean_lemma",
    "stem": "clean_stem",
}

titles = news["title"].tolist()
bodies = news["text"].tolist()
clean_stop = []
clean_lemma = []
clean_stem = []
started = time.perf_counter()
total = len(news)

for done, (title, body) in enumerate(zip(titles, bodies), start=1):
    tokens = tokens_from_raw(title, body)
    clean_stop.append(render(tokens, "stopwords"))
    clean_lemma.append(render(tokens, "lemma"))
    clean_stem.append(render(tokens, "stem"))
    if done % 5000 == 0 or done == total:
        elapsed = time.perf_counter() - started
        print(f"Preprocessed {done:,}/{total:,} articles in {elapsed:.0f}s", flush=True)

news["clean_stop"] = clean_stop
news["clean_lemma"] = clean_lemma
news["clean_stem"] = clean_stem
del titles, bodies, clean_stop, clean_lemma, clean_stem

empty = news["clean_stop"].str.strip().eq("")
print(f"Articles with no content tokens: {int(empty.sum()):,}")
news = news.loc[~empty].reset_index(drop=True)

reuters_left = int(news["clean_stop"].str.contains(r"(?:^| )reuter(?:s)?(?: |$)").sum())
print(f"Cleaned articles that still contain a Reuters token: {reuters_left}")
if reuters_left != 0:
    raise RuntimeError("Reuters tokens survived preprocessing.")

classic_wire = news["text"].str.match(r"[A-Z][A-Za-z .'\-]{0,40}\(Reuters\)")
any_wire = news["text"].str.contains(r"\(Reuters\)", regex=True)
if classic_wire.any():
    example_id = int(news.index[classic_wire][0])
elif any_wire.any():
    example_id = int(news.index[any_wire][0])
else:
    example_id = None
if example_id is not None:
    print()
    print("Dateline removal, real article")
    print("TITLE:", news.at[example_id, "title"][:160])
    print("RAW:  ", " ".join(news.at[example_id, "text"].split())[:280])
    print("CLEAN:", news.at[example_id, "clean_stop"][:280])

fake_id = int(news.index[news["label"] == 1][0])
print()
print("Fake article after lemmatization")
print("RAW TITLE:", news.at[fake_id, "title"][:180])
print("CLEAN:    ", news.at[fake_id, "clean_lemma"][:280])
print()
print(f"Articles ready to split: {len(news):,}")
print(f"Unique content types cached: {len(_lemma_cache):,} lemmas, {len(_stem_cache):,} stems")
```

### Cell 14, 6. Training, validation, and test splits

```python
all_index = news.index.to_numpy()
labels = news["label"].to_numpy(dtype=np.int64)

idx_rest, idx_te = train_test_split(
    all_index,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=labels,
)
idx_tr, idx_va = train_test_split(
    idx_rest,
    test_size=VAL_FRACTION,
    random_state=SEED,
    stratify=labels[idx_rest],
)

assert set(idx_tr).isdisjoint(idx_va)
assert set(idx_tr).isdisjoint(idx_te)
assert set(idx_va).isdisjoint(idx_te)
assert len(idx_tr) + len(idx_va) + len(idx_te) == len(news)

y_train = news.loc[idx_tr, "label"].to_numpy(dtype=np.int64)
y_val = news.loc[idx_va, "label"].to_numpy(dtype=np.int64)
y_test = news.loc[idx_te, "label"].to_numpy(dtype=np.int64)


def split_mix(name, y):
    print(f"{name:12} n={len(y):,}  fake={int((y == 1).sum()):,} ({(y == 1).mean():.1%})  real={int((y == 0).sum()):,}")


split_mix("training", y_train)
split_mix("validation", y_val)
split_mix("test", y_test)
print("The three index sets are disjoint and cover every article.")
```

### Cell 16, 7. Choosing the text representation

```python
def probe_models():
    return [
        ("Multinomial Naive Bayes", "alpha=0.5", MultinomialNB(alpha=0.5)),
        (
            "Logistic Regression",
            "C=1.0, class_weight=balanced",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=SEED,
            ),
        ),
        (
            "Linear SVM",
            "C=1.0, class_weight=balanced",
            LinearSVC(C=1.0, class_weight="balanced", max_iter=8000, random_state=SEED),
        ),
    ]


bakeoff_rows = []
bakeoff_best = None
for variant, column in VARIANT_COLUMNS.items():
    print(f"Representation: {variant}", flush=True)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=BIGRAM_FEATURES,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    x_train = vectorizer.fit_transform(news.loc[idx_tr, column].astype(str).tolist())
    x_val = vectorizer.transform(news.loc[idx_va, column].astype(str).tolist())
    print(f"  TF-IDF shape {x_train.shape}", flush=True)
    for model_name, params, model in probe_models():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        point = evaluate_fitted(model, x_val, y_val)
        elapsed = time.perf_counter() - started
        row = {
            "representation": variant,
            "model": model_name,
            "hyperparameters": params,
            "threshold": point["threshold"],
            "val_accuracy": point["metrics"]["accuracy"],
            "val_recall_fake": point["metrics"]["recall_fake"],
            "val_precision_fake": point["metrics"]["precision_fake"],
            "val_f1_fake": point["metrics"]["f1_fake"],
            "val_selection_score": point["selection_score"],
        }
        bakeoff_rows.append(row)
        print(
            f"  {model_name:24} S={point['selection_score']:.4f}  "
            f"acc={point['metrics']['accuracy']:.4f}  rec={point['metrics']['recall_fake']:.4f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        rank = (point["selection_score"], point["metrics"]["recall_fake"], point["metrics"]["accuracy"])
        if bakeoff_best is None or rank > bakeoff_best["rank"]:
            bakeoff_best = {"rank": rank, "variant": variant}
    del x_train, x_val, vectorizer
    gc.collect()

bakeoff = pd.DataFrame(bakeoff_rows).sort_values(
    ["val_selection_score", "val_recall_fake", "val_accuracy"],
    ascending=False,
).reset_index(drop=True)
bakeoff.insert(0, "rank", bakeoff.index + 1)
print()
display(bakeoff.round(4))
bakeoff.to_csv("representation_bakeoff.csv", index=False)

TEXT_VARIANT = bakeoff_best["variant"]
TEXT_COL = VARIANT_COLUMNS[TEXT_VARIANT]
print(f"Selected representation: {TEXT_VARIANT}  (column {TEXT_COL})")
print("The hyperparameter search below uses this representation and no other.")
```

### Cell 18, 8. TF-IDF and the classifier search

```python
def model_factories():
    specs = []
    for alpha in (0.1, 0.5, 1.0):
        specs.append((
            "Multinomial Naive Bayes",
            f"alpha={alpha}",
            lambda alpha=alpha: MultinomialNB(alpha=alpha),
        ))
    for C in (1.0, 5.0):
        for class_weight in (None, "balanced"):
            weight_name = "None" if class_weight is None else class_weight
            specs.append((
                "Logistic Regression",
                f"C={C}, class_weight={weight_name}",
                lambda C=C, class_weight=class_weight: LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ))
    for C in (0.5, 1.0, 2.0):
        for class_weight in (None, "balanced"):
            weight_name = "None" if class_weight is None else class_weight
            specs.append((
                "Linear SVM",
                f"C={C}, class_weight={weight_name}",
                lambda C=C, class_weight=class_weight: LinearSVC(
                    C=C,
                    class_weight=class_weight,
                    max_iter=8000,
                    random_state=SEED,
                ),
            ))
    return specs


VECTORIZER_SPECS = [
    ("TF-IDF unigrams", dict(ngram_range=(1, 1), max_features=UNIGRAM_FEATURES)),
    ("TF-IDF unigrams+bigrams", dict(ngram_range=(1, 2), max_features=BIGRAM_FEATURES)),
]

train_text = news.loc[idx_tr, TEXT_COL].astype(str).tolist()
val_text = news.loc[idx_va, TEXT_COL].astype(str).tolist()
if any(text.strip() == "" for text in train_text):
    raise RuntimeError("Empty training documents reached TF-IDF.")

factories = model_factories()
total_fits = len(VECTORIZER_SPECS) * len(factories)
print(f"Representation: {TEXT_VARIANT}")
print(f"Candidates: {total_fits}", flush=True)

search_rows = []
family_best = {}
fit_number = 0
search_started = time.perf_counter()

for vec_name, vec_params in VECTORIZER_SPECS:
    vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
        **vec_params,
    )
    started = time.perf_counter()
    x_train = vectorizer.fit_transform(train_text)
    x_val = vectorizer.transform(val_text)
    print(
        f"{vec_name}: {x_train.shape[1]:,} features, fit in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    for model_name, params, factory in factories:
        fit_number += 1
        started = time.perf_counter()
        model = factory()
        try:
            model.fit(x_train, y_train)
            point = evaluate_fitted(model, x_val, y_val)
        except Exception as exc:
            print(f"[{fit_number}/{total_fits}] FAILED {model_name} {params}: {exc}", flush=True)
            continue
        elapsed = time.perf_counter() - started
        rank = (
            point["selection_score"],
            point["metrics"]["recall_fake"],
            point["metrics"]["accuracy"],
        )
        search_rows.append({
            "representation": TEXT_VARIANT,
            "model": model_name,
            "vectorizer": vec_name,
            "hyperparameters": params,
            "threshold": point["threshold"],
            "score_kind": point["kind"],
            "val_accuracy": point["metrics"]["accuracy"],
            "val_precision_fake": point["metrics"]["precision_fake"],
            "val_recall_fake": point["metrics"]["recall_fake"],
            "val_f1_fake": point["metrics"]["f1_fake"],
            "val_f1_macro": point["metrics"]["f1_macro"],
            "val_selection_score": point["selection_score"],
            "val_selection_at_default_threshold": selection_score(point["default_metrics"]),
        })
        print(
            f"[{fit_number}/{total_fits}] {model_name:24} {params:28} {vec_name:28} "
            f"S={point['selection_score']:.4f} acc={point['metrics']['accuracy']:.4f} "
            f"rec={point['metrics']['recall_fake']:.4f} ({elapsed:.1f}s)",
            flush=True,
        )
        previous = family_best.get(model_name)
        if previous is None or rank > previous["rank"]:
            family_best[model_name] = {
                "rank": rank,
                "model_name": model_name,
                "vectorizer_name": vec_name,
                "hyperparameters": params,
                "variant": TEXT_VARIANT,
                "kind": point["kind"],
                "threshold": point["threshold"],
                "val_metrics": point["metrics"],
                "val_selection_score": point["selection_score"],
                "default_selection_score": selection_score(point["default_metrics"]),
                "model": model,
                "vectorizer": vectorizer,
            }
            print(f"    new best {model_name}: S={point['selection_score']:.4f}", flush=True)
    del x_train, x_val
    gc.collect()

if set(family_best) != {"Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"}:
    missing = {"Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"} - set(family_best)
    raise RuntimeError(f"These model families did not finish: {sorted(missing)}")

BEST = max(family_best.values(), key=lambda bundle: bundle["rank"])
results = pd.DataFrame(search_rows).sort_values(
    ["val_selection_score", "val_recall_fake", "val_accuracy"],
    ascending=False,
).reset_index(drop=True)
results.insert(0, "rank", results.index + 1)
results.to_csv("validation_model_comparison.csv", index=False)

print()
print(f"Search finished in {(time.perf_counter() - search_started) / 60:.1f} min")
print(
    "Validation choice:",
    BEST["model_name"],
    "|",
    BEST["vectorizer_name"],
    "|",
    BEST["hyperparameters"],
    "|",
    f"threshold={BEST['threshold']:.4f}",
    "|",
    f"S={BEST['val_selection_score']:.4f}",
)
gain = BEST["val_selection_score"] - BEST["default_selection_score"]
print(f"Moving the threshold changed this model's validation score by {gain:+.4f}.")
top_row = results.iloc[0]
if top_row["model"] != BEST["model_name"] or abs(top_row["val_selection_score"] - BEST["val_selection_score"]) > 1e-9:
    raise RuntimeError("The ranked table and the stored winner disagree.")
display(results.round(4))

chart = results.head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
positions = np.arange(len(chart))
labels = chart["model"] + " | " + chart["vectorizer"] + " | " + chart["hyperparameters"]
ax.barh(positions - 0.16, chart["val_accuracy"], height=0.32, color=COLOR_INK, label="Validation accuracy")
ax.barh(positions + 0.16, chart["val_recall_fake"], height=0.32, color=COLOR_FAKE, label="Validation recall (Fake)")
ax.set_yticks(positions)
ax.set_yticklabels(labels)
floor = max(0.0, float(min(chart["val_accuracy"].min(), chart["val_recall_fake"].min()) - 0.03))
ax.set_xlim(floor, 1.0)
ax.set_xlabel("Score")
ax.set_title("Top validation candidates")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Cell 20, 9. Locked test evaluation

```python
test_text = news.loc[idx_te, TEXT_COL].astype(str).tolist()
family_order = ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"]
test_rows = []
stored_predictions = {}

for model_name in family_order:
    bundle = family_best[model_name]
    matrix = bundle["vectorizer"].transform(test_text)
    scores = positive_scores(bundle["model"], matrix)
    pred = (scores >= bundle["threshold"]).astype(np.int64)
    metrics = metric_dict(y_test, pred)
    try:
        roc_auc = float(roc_auc_score(y_test, scores))
        average_precision = float(average_precision_score(y_test, scores))
    except ValueError:
        roc_auc = float("nan")
        average_precision = float("nan")
    test_rows.append({
        "model": model_name,
        "selected": model_name == BEST["model_name"],
        "representation": bundle["variant"],
        "vectorizer": bundle["vectorizer_name"],
        "hyperparameters": bundle["hyperparameters"],
        "threshold": bundle["threshold"],
        "accuracy": metrics["accuracy"],
        "precision_fake": metrics["precision_fake"],
        "recall_fake": metrics["recall_fake"],
        "f1_fake": metrics["f1_fake"],
        "precision_real": metrics["precision_real"],
        "recall_real": metrics["recall_real"],
        "f1_real": metrics["f1_real"],
        "f1_macro": metrics["f1_macro"],
        "mcc": metrics["mcc"],
        "roc_auc": roc_auc,
        "average_precision_fake": average_precision,
        "selection_score": selection_score(metrics),
        "val_selection_score": bundle["val_selection_score"],
    })
    stored_predictions[model_name] = {"pred": pred, "scores": scores, "metrics": metrics}

test_comparison = pd.DataFrame(test_rows)
test_comparison.to_csv("test_model_comparison.csv", index=False)
display(test_comparison.round(4))

majority = int(np.bincount(y_train).argmax())
majority_accuracy = float((y_test == majority).mean())
winner_pred = stored_predictions[BEST["model_name"]]
TEST = {
    "index": np.asarray(idx_te),
    "y": y_test,
    "pred": winner_pred["pred"],
    "scores": winner_pred["scores"],
    "metrics": winner_pred["metrics"],
}

report = classification_report(
    y_test,
    TEST["pred"],
    labels=[0, 1],
    target_names=["Real", "Fake"],
    digits=4,
    zero_division=0,
)
metrics = TEST["metrics"]
summary = "\n".join([
    "LOCKED TEST EVALUATION",
    "Dataset: ISOT Fake News Dataset (legacy article corpus)",
    f"Articles in the test set: {len(y_test):,}  |  fake {(y_test == 1).sum():,}  |  real {(y_test == 0).sum():,}",
    f"Pipeline: {BEST['variant']} text | {BEST['vectorizer_name']} | {BEST['model_name']}",
    f"Hyperparameters: {BEST['hyperparameters']}",
    f"Threshold: {BEST['threshold']:.4f} ({BEST['kind']} score; predict Fake when the score is at least this value)",
    "",
    f"Accuracy:              {metrics['accuracy']:.4f}",
    f"Precision (Fake):      {metrics['precision_fake']:.4f}",
    f"Recall (Fake):         {metrics['recall_fake']:.4f}",
    f"F1 (Fake):             {metrics['f1_fake']:.4f}",
    f"Precision (Real):      {metrics['precision_real']:.4f}",
    f"Recall (Real):         {metrics['recall_real']:.4f}",
    f"F1 (Real):             {metrics['f1_real']:.4f}",
    f"Macro F1:              {metrics['f1_macro']:.4f}",
    f"Matthews correlation:  {metrics['mcc']:.4f}",
    f"ROC-AUC:               {float(test_comparison.loc[test_comparison['model'] == BEST['model_name'], 'roc_auc'].iloc[0]):.4f}",
    f"Selection score S:     {selection_score(metrics):.4f}",
    f"Majority baseline accuracy: {majority_accuracy:.4f}",
    "",
    report.rstrip(),
])
print()
print(summary)
with open("test_report.txt", "w", encoding="utf-8") as handle:
    handle.write(summary + "\n")

test_leader = test_comparison.sort_values("selection_score", ascending=False).iloc[0]["model"]
print()
print(f"Validation selected: {BEST['model_name']}")
print(f"Highest test selection score among the three frozen models: {test_leader}")
print("The reported pipeline is the validation choice. The test set measures it.")

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(test_comparison))
width = 0.36
ax.bar(x - width / 2, test_comparison["accuracy"], width, color=COLOR_INK, label="Accuracy")
ax.bar(x + width / 2, test_comparison["recall_fake"], width, color=COLOR_FAKE, label="Recall (Fake)")
ax.set_xticks(x)
ax.set_xticklabels(test_comparison["model"])
floor = max(0.0, float(min(test_comparison["accuracy"].min(), test_comparison["recall_fake"].min()) - 0.03))
ax.set_ylim(floor, 1.0)
ax.set_ylabel("Test score")
ax.set_title(f"Locked test scores  |  selected: {BEST['model_name']}")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("test_scores.png", dpi=150, bbox_inches="tight")
plt.show()

disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    TEST["pred"],
    labels=[0, 1],
    display_labels=["Real", "Fake"],
    cmap="Blues",
    colorbar=False,
)
disp.ax_.set_title(f"Test confusion matrix — {BEST['model_name']}")
disp.figure_.tight_layout()
disp.figure_.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

precision, recall, _ = precision_recall_curve(y_test, TEST["scores"], pos_label=1)
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(recall, precision, color=COLOR_INK, label="Frozen model")
ax.scatter(
    [metrics["recall_fake"]],
    [metrics["precision_fake"]],
    color=COLOR_FAKE,
    zorder=3,
    label="Chosen threshold",
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Recall (Fake)")
ax.set_ylabel("Precision (Fake)")
ax.set_title("Test precision-recall curve for the fake class")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig("precision_recall_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print("Wrote test_report.txt, test_model_comparison.csv, and the figures beside the notebook.")
```

### Cell 26, 12. Screening a new article

```python
def screen_article(title, article):
    text = render(tokens_from_raw(title, article), BEST["variant"])
    if not text.strip():
        return {
            "label": "Unusable",
            "score": None,
            "threshold": BEST["threshold"],
            "detail": "Preprocessing removed every token. Paste a longer article.",
        }
    matrix = BEST["vectorizer"].transform([text])
    score = float(positive_scores(BEST["model"], matrix)[0])
    label = "Fake" if score >= BEST["threshold"] else "Real"
    if BEST["kind"] == "proba":
        detail = f"Estimated P(Fake) = {score:.4f}. Predict Fake at or above {BEST['threshold']:.4f}."
    else:
        detail = (
            f"SVM decision score = {score:.4f}. "
            f"Predict Fake at or above {BEST['threshold']:.4f}."
        )
    return {"label": label, "score": score, "threshold": BEST["threshold"], "detail": detail}


def print_screening(title, article, heading):
    result = screen_article(title, article)
    print(heading)
    print(f"  prediction: {result['label']}")
    print(f"  {result['detail']}")
    print()


check_ids = list(idx_te[:8])
rebuilt = [
    render(tokens_from_raw(news.at[i, "title"], news.at[i, "text"]), BEST["variant"])
    for i in check_ids
]
assert rebuilt == news.loc[check_ids, TEXT_COL].astype(str).tolist()
batch_scores = positive_scores(BEST["model"], BEST["vectorizer"].transform(rebuilt))
batch_pred = (batch_scores >= BEST["threshold"]).astype(int)
single_pred = [
    1 if screen_article(news.at[i, "title"], news.at[i, "text"])["label"] == "Fake" else 0
    for i in check_ids
]
assert np.array_equal(batch_pred, np.asarray(single_pred))
print("Single-article screening matches the batch path used in the test evaluation.")
print()

print_screening(
    "Shocking leak proves the election servers were swapped overnight",
    "Anonymous patriots are reporting that crates of replacement voting servers were rolled "
    "into a warehouse at 3am while the media stayed silent. Share this before they delete it. "
    "Officials refuse to comment, and unnamed experts say the numbers are impossible. "
    "You will not see this on cable news. Wake up.",
    "Demo A — sensational claim",
)
print_screening(
    "Wholesale inventories rise 0.2 percent in February",
    "Wholesale inventories in the United States increased 0.2 percent in February after a "
    "revised 0.1 percent gain in January, the Commerce Department said on Thursday. Sales at "
    "wholesalers rose 0.4 percent. Economists surveyed ahead of the release had expected "
    "inventories to be unchanged. The stock-to-sales ratio held at 1.25.",
    "Demo B — dry economic report",
)
print_screening(
    "City council approves a new east-side bus route",
    "The council voted 7 to 2 on Tuesday to add an east-side bus route starting in September. "
    "The route runs from the central station to the hospital every 20 minutes on weekdays. "
    "Funding comes from the existing transit levy. Riders can see the timetable on the city "
    "website next month.",
    "Demo C — local civic report",
)

# Paste a headline and an article here, then rerun this cell.
USER_TITLE = ""
USER_ARTICLE = ""
if USER_ARTICLE.strip():
    print_screening(USER_TITLE, USER_ARTICLE, "Your article")

artifact = {
    "model": BEST["model"],
    "vectorizer": BEST["vectorizer"],
    "threshold": BEST["threshold"],
    "score_kind": BEST["kind"],
    "variant": BEST["variant"],
    "model_name": BEST["model_name"],
    "vectorizer_name": BEST["vectorizer_name"],
    "hyperparameters": BEST["hyperparameters"],
    "label_map": {0: "Real", 1: "Fake"},
    "test_metrics": TEST["metrics"],
    "max_tokens": MAX_TOKENS,
    "title_cap": TITLE_CAP,
    "title_repeats": TITLE_REPEATS,
}
dump(artifact, "best_fake_news_pipeline.joblib", compress=3)
print("Saved best_fake_news_pipeline.joblib")
print("Screening a saved file still needs the preprocessing functions in this notebook.")
```
