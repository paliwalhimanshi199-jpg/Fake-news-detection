"""Preprocessing and inference that mirror the frozen pipeline in Fake_News_Detection_NLP.ipynb.

Keep the constants and regexes here in sync with sections 1, 5 and 12 of the notebook.
"""
from __future__ import annotations

import html
import os
import re
from functools import lru_cache
from pathlib import Path

import joblib
import nltk
import numpy as np

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get("FAKE_NEWS_MODEL", APP_DIR / "artifacts" / "best_fake_news_pipeline.joblib")
)

MAX_TOKENS = 500
TITLE_CAP = 40
TITLE_REPEATS = 2
NEGATION = {"no", "not", "nor", "never"}
REUTERS_TOKENS = {"reuters", "reuter"}

URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+", re.IGNORECASE)
DATELINE_RE = re.compile(
    r"\b[A-Za-z][A-Za-z .'\-]{0,60}\(\s*reuters\s*\)\s*[-:\u2013\u2014]?\s*",
    re.IGNORECASE,
)
NON_LETTER_RE = re.compile(r"[^a-z\s]+")


def _ensure_nltk(resource: str, path: str) -> None:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    _ensure_nltk("stopwords", "corpora/stopwords")
    return frozenset(nltk.corpus.stopwords.words("english")) - NEGATION


@lru_cache(maxsize=1)
def _lemmatizer():
    from nltk.stem import WordNetLemmatizer

    _ensure_nltk("wordnet", "corpora/wordnet")
    _ensure_nltk("omw-1.4", "corpora/omw-1.4")
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem import PorterStemmer

    return PorterStemmer()


def base_tokens(text) -> list[str]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = html.unescape(text)
    text = URL_RE.sub(" ", text)
    text = DATELINE_RE.sub(" ", text)
    text = text.lower()
    text = NON_LETTER_RE.sub(" ", text)
    stop = stopwords()
    return [
        token
        for token in text.split()
        if len(token) >= 2 and token not in stop and token not in REUTERS_TOKENS
    ]


def tokens_from_raw(title, body) -> list[str]:
    title_tokens = base_tokens(title)[:TITLE_CAP]
    body_tokens = base_tokens(body)
    return (title_tokens * TITLE_REPEATS + body_tokens)[:MAX_TOKENS]


def render(tokens: list[str], variant: str) -> str:
    if variant == "lemma":
        lemmatizer = _lemmatizer()
        produced = [
            lemmatizer.lemmatize(lemmatizer.lemmatize(token, pos="n"), pos="v") for token in tokens
        ]
    elif variant == "stem":
        stemmer = _stemmer()
        produced = [stemmer.stem(token) for token in tokens]
    elif variant == "stopwords":
        produced = list(tokens)
    else:
        raise ValueError(f"Unknown representation: {variant}")
    stop = stopwords()
    return " ".join(
        token
        for token in produced
        if len(token) >= 2 and token not in stop and token not in REUTERS_TOKENS
    )


def positive_scores(model, matrix) -> np.ndarray:
    # Higher score means the article looks more fake.
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
    return np.asarray(model.decision_function(matrix), dtype=float)


class FakeNewsClassifier:
    """Loads the joblib bundle saved by the notebook and classifies articles."""

    def __init__(self, path: Path | str = MODEL_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.vectorizer = bundle["vectorizer"]
        self.threshold = float(bundle["threshold"])
        self.variant = bundle["variant"]
        self.score_kind = bundle["score_kind"]
        self.model_name = bundle["model_name"]
        self.vectorizer_name = bundle["vectorizer_name"]
        self.hyperparameters = bundle["hyperparameters"]

    def predict_many(self, articles) -> list[dict]:
        """Classify (title, body) pairs. Returns one dict per input, in order."""
        cleaned = [render(tokens_from_raw(title, body), self.variant) for title, body in articles]
        results = [{"label": "Unusable", "score": None} for _ in cleaned]
        usable = [i for i, text in enumerate(cleaned) if text.strip()]
        if usable:
            matrix = self.vectorizer.transform([cleaned[i] for i in usable])
            scores = positive_scores(self.model, matrix)
            for i, score in zip(usable, scores):
                results[i] = {
                    "label": "Fake" if score >= self.threshold else "Real",
                    "score": float(score),
                }
        return results

    def predict(self, title, body) -> dict:
        return self.predict_many([(title, body)])[0]
