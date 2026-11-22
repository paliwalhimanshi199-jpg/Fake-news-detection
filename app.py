"""Streamlit front end: live news filter, saved articles, and pipeline accuracy."""
import json

import pandas as pd
import streamlit as st

import news_store
from news_feed import fetch_feed
from pipeline import APP_DIR, MODEL_PATH, FakeNewsClassifier
from ui import bar_cards, esc, hero, inject_css, news_item, render, stat_cards, verdict

METRICS_PATH = APP_DIR / "artifacts" / "metrics.json"
COMPARISON_PATH = APP_DIR / "artifacts" / "test_model_comparison.csv"
LABEL_OPTIONS = ["All", "Fake", "Real", "Unusable"]
PAGES = {"📡  Live news": "live", "✍️  Add news": "add", "📊  Accuracy": "accuracy"}

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
inject_css()
news_store.init_db()


@st.cache_resource(show_spinner="Loading the pipeline...")
def load_classifier() -> FakeNewsClassifier:
    return FakeNewsClassifier()


@st.cache_data(ttl=300, show_spinner="Fetching live headlines...")
def load_feed(query: str) -> list[dict]:
    return fetch_feed(query)


def get_classifier() -> FakeNewsClassifier:
    try:
        return load_classifier()
    except FileNotFoundError:
        st.error(
            f"Model file not found: `{MODEL_PATH}`. Copy `best_fake_news_pipeline.joblib` "
            "from the notebook into `fake_news_app/artifacts/`. See README.md."
        )
        st.stop()


def live_page() -> None:
    hero("Live news screening", "Latest headlines from Google News, scored by the fake news pipeline.")
    clf = get_classifier()

    col_query, col_label, col_keyword, col_refresh = st.columns([3, 1.5, 2, 1])
    query = col_query.text_input("Search topic", value="world news")
    label = col_label.selectbox("Prediction", LABEL_OPTIONS, key="live_label")
    keyword = col_keyword.text_input("Filter headlines", key="live_keyword")
    if col_refresh.button("↻ Refresh"):
        load_feed.clear()

    try:
        items = load_feed(query.strip() or "world news")
    except Exception as exc:  # network errors, bad XML, and so on
        st.error(f"Could not load the live feed: {exc}")
        return
    if not items:
        st.info("The feed returned no headlines for this topic.")
        return

    predictions = clf.predict_many((item["headline"], "") for item in items)
    rows = [{**item, **pred} for item, pred in zip(items, predictions)]
    if keyword.strip():
        rows = [r for r in rows if keyword.strip().lower() in r["headline"].lower()]
    if label != "All":
        rows = [r for r in rows if r["label"] == label]

    stat_cards([
        ("Headlines shown", str(len(rows))),
        ("Predicted Fake", str(sum(r["label"] == "Fake" for r in rows))),
        ("Predicted Real", str(sum(r["label"] == "Real" for r in rows))),
    ])
    if not rows:
        st.info("No headlines match these filters.")
    else:
        render("".join(
            news_item(r["headline"], r["label"], r["score"], r["source"], r["published"], r["link"])
            for r in rows[:50]
        ))
    st.caption(
        "Predictions use headlines only. The model was trained on full article text, "
        "so treat these as a screening signal, not a verdict."
    )


def add_page() -> None:
    hero("Add and screen news", "Paste an article to get a verdict, then filter your saved list.")
    clf = get_classifier()

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### New article")
        with st.form("add_article", clear_on_submit=True):
            headline = st.text_input("Headline", placeholder="e.g. Council approves new bus route")
            source = st.text_input("Source (optional)", placeholder="e.g. City Gazette")
            body = st.text_area("Article text", height=200, placeholder="Paste the article body here")
            submitted = st.form_submit_button("Classify and save", type="primary")
        if submitted:
            if not headline.strip() and not body.strip():
                st.warning("Enter a headline or article text first.")
            else:
                result = clf.predict(headline, body)
                news_store.add_article(
                    headline.strip(), body.strip(), source.strip(), result["label"], result["score"]
                )
                st.session_state["last_result"] = {"result": result, "headline": headline.strip()}

    with right:
        st.markdown("#### Verdict")
        last = st.session_state.get("last_result")
        if not last:
            render('<div class="card muted">Submit an article to see its verdict here.</div>')
        elif last["result"]["label"] == "Unusable":
            verdict("Unusable", "Nothing usable is left after cleaning. Add more text.", last["headline"])
        else:
            res = last["result"]
            verdict(
                res["label"],
                f"Score {res['score']:.4f} against a threshold of {clf.threshold:.4f}. "
                "Scores at or above the threshold are Fake.",
                last["headline"],
            )

    st.markdown("#### Saved articles")
    counts = news_store.counts()
    stat_cards([
        ("Total saved", str(sum(counts.values()))),
        ("Fake", str(counts.get("Fake", 0))),
        ("Real", str(counts.get("Real", 0))),
        ("Unusable", str(counts.get("Unusable", 0))),
    ])
    col_label, col_keyword = st.columns([1, 3])
    label = col_label.selectbox("Prediction", LABEL_OPTIONS, key="saved_label")
    keyword = col_keyword.text_input("Search headline, text, or source", key="saved_keyword")

    rows = news_store.list_articles(label, keyword.strip())
    if not rows:
        st.info("No saved articles match these filters.")
        return
    render("".join(
        news_item(
            r["title"], r["label"], r["score"], r["source"], r["created_at"],
            preview=r["body"][:180],
        )
        for r in rows[:100]
    ))


def accuracy_page() -> None:
    if not METRICS_PATH.exists():
        st.error(f"Missing `{METRICS_PATH.name}` in the artifacts folder.")
        return
    report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    m = report["test_metrics"]
    n_fake, n_real = report["test_fake"], report["test_real"]

    hero(
        "Pipeline accuracy",
        f"{report['dataset']}. Locked test set of {report['test_size']:,} articles, "
        "chosen on validation data only.",
    )

    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    bar_cards([
        ("Accuracy", m["accuracy"], pct(m["accuracy"])),
        ("Precision (Fake)", m["precision_fake"], pct(m["precision_fake"])),
        ("Recall (Fake)", m["recall_fake"], pct(m["recall_fake"])),
        ("F1 (Fake)", m["f1_fake"], pct(m["f1_fake"])),
    ])
    bar_cards([
        ("Precision (Real)", m["precision_real"], pct(m["precision_real"])),
        ("Recall (Real)", m["recall_real"], pct(m["recall_real"])),
        ("F1 (Real)", m["f1_real"], pct(m["f1_real"])),
        ("Macro F1", m["f1_macro"], pct(m["f1_macro"])),
    ])
    stat_cards([
        ("Matthews correlation", f"{m['mcc']:.4f}"),
        ("ROC-AUC", f"{m['roc_auc']:.4f}"),
        ("Selection score S", f"{m['selection_score']:.4f}"),
        ("Majority baseline", pct(report["majority_baseline_accuracy"])),
    ])

    # Derived from recall and class support. Counts round to whole articles.
    tp = round(m["recall_fake"] * n_fake)
    fn = n_fake - tp
    tn = round(m["recall_real"] * n_real)
    fp = n_real - tn

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Confusion matrix")
        render(
            '<div class="cm">'
            '<div class="head"></div><div class="head">Predicted Real</div><div class="head">Predicted Fake</div>'
            '<div class="head">Actual Real</div>'
            f'<div class="cell ok"><div class="num">{tn:,}</div>correct</div>'
            f'<div class="cell bad"><div class="num">{fp:,}</div>false alarms</div>'
            '<div class="head">Actual Fake</div>'
            f'<div class="cell bad"><div class="num">{fn:,}</div>missed</div>'
            f'<div class="cell ok"><div class="num">{tp:,}</div>caught</div>'
            '</div>'
        )
        st.caption("Counts are derived from the reported recall and class support.")
    with right:
        st.markdown("#### Pipeline")
        p = report["pipeline"]
        settings = [
            ("Text representation", p["representation"]),
            ("Features", p["vectorizer"]),
            ("Model", p["model"]),
            ("Hyperparameters", p["hyperparameters"]),
            ("Decision threshold", f"{p['threshold']:.4f}"),
            ("Score type", p["score_kind"]),
        ]
        kv = "".join(f'<div class="kv"><span class="muted">{esc(k)}</span><span>{esc(v)}</span></div>'
                     for k, v in settings)
        render(f'<div class="card">{kv}</div>')

    if COMPARISON_PATH.exists():
        st.markdown("#### Frozen models on the test set")
        st.dataframe(pd.read_csv(COMPARISON_PATH), hide_index=True)

    render(
        '<div class="note"><b>Read these numbers with care.</b> ISOT is a legacy corpus. Its real '
        "articles are Reuters wire copy and its fake articles come from other outlets, so the model "
        "can learn that house style. These scores can overstate performance on new news from other sources.</div>"
    )


with st.sidebar:
    st.markdown("### 📰 Fake News Detector")
    choice = st.radio("Section", list(PAGES), label_visibility="collapsed")
    render(
        f'<div class="card" style="margin-top:0.8rem"><div class="label">Active model</div>'
        f'<div class="value" style="font-size:0.95rem;word-break:break-all">{esc(MODEL_PATH.name)}</div></div>'
    )

page = PAGES[choice]
if page == "live":
    live_page()
elif page == "add":
    add_page()
else:
    accuracy_page()
