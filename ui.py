"""Light-mode styling and small HTML components for the Streamlit app."""
import html

import streamlit as st

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1180px;}
#MainMenu, footer {visibility: hidden;}
.hero {background: linear-gradient(135deg, #6C5CE7 0%, #0984E3 100%); color: #ffffff; border-radius: 18px; padding: 1.5rem 2rem; margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(108, 92, 231, 0.22);}
.hero h1 {color: #ffffff; margin: 0; font-size: 1.9rem; font-weight: 800;}
.hero p {color: #ffffff; margin: 0.3rem 0 0 0; opacity: 0.92;}
.card {background: #ffffff; border: 1px solid #E4E7EF; border-radius: 14px; padding: 1rem 1.15rem; box-shadow: 0 1px 3px rgba(20, 20, 60, 0.05); color: #1F2937;}
.grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.75rem; margin-bottom: 0.9rem;}
.label {font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6B7280; font-weight: 600;}
.value {font-size: 1.6rem; font-weight: 700; line-height: 1.2; margin-top: 0.25rem; color: #111827;}
.bar {height: 6px; border-radius: 99px; background: #EEF0F6; margin-top: 0.6rem; overflow: hidden;}
.bar > span {display: block; height: 100%; background: linear-gradient(90deg, #6C5CE7, #0984E3); border-radius: 99px;}
.badge {display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.03em;}
.badge-fake {background: #FDE8EA; color: #B42335;}
.badge-real {background: #E3F6EE; color: #1B7F5A;}
.badge-unusable {background: #FFF3D6; color: #8A5A00;}
.muted {color: #6B7280; font-size: 0.82rem;}
.news-item {background: #ffffff; border: 1px solid #E4E7EF; border-radius: 12px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; box-shadow: 0 1px 2px rgba(20, 20, 60, 0.04);}
.news-item:hover {border-color: #6C5CE7;}
.news-meta {display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.35rem;}
.score {margin-left: auto; font-family: monospace; color: #6B7280; font-size: 0.82rem;}
.news-title {font-weight: 600; font-size: 1.02rem; color: #111827 !important; text-decoration: none;}
.news-title:hover {text-decoration: underline;}
.preview {color: #4B5563; font-size: 0.88rem; margin-top: 0.3rem;}
.verdict {border-radius: 16px; padding: 1.2rem 1.4rem; color: #ffffff; box-shadow: 0 6px 18px rgba(20, 20, 60, 0.12);}
.verdict-fake {background: linear-gradient(135deg, #E63946, #FF7A59);}
.verdict-real {background: linear-gradient(135deg, #14946A, #2ECC8F);}
.verdict-unusable {background: linear-gradient(135deg, #C98A00, #F5A623);}
.verdict .label {color: #ffffff; opacity: 0.85;}
.verdict h3 {color: #ffffff; margin: 0.1rem 0 0.2rem 0; font-size: 1.7rem;}
.verdict p {color: #ffffff; margin: 0; opacity: 0.95;}
.cm {display: grid; grid-template-columns: 130px 1fr 1fr; gap: 0.5rem; text-align: center;}
.cm .head {font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.06em; color: #6B7280; font-weight: 600; display: flex; align-items: center; justify-content: center;}
.cm .cell {border-radius: 12px; padding: 1rem 0.5rem; color: #1F2937; font-size: 0.8rem;}
.cm .ok {background: #E3F6EE;}
.cm .bad {background: #FDE8EA;}
.cm .num {font-size: 1.6rem; font-weight: 700; color: #111827;}
.kv {display: flex; justify-content: space-between; gap: 1rem; padding: 0.5rem 0; border-bottom: 1px solid #EEF0F6;}
.kv:last-child {border-bottom: none;}
.note {background: #FFF8E6; border-left: 4px solid #F5A623; color: #5C4300; padding: 0.8rem 1rem; border-radius: 8px; font-size: 0.9rem; margin-top: 0.8rem;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value))


def render(markup: str) -> None:
    # Build markup without newlines, so Markdown does not treat it as a code block.
    st.markdown(markup, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    render(f'<div class="hero"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>')


def badge(label: str) -> str:
    css = {"Fake": "badge-fake", "Real": "badge-real"}.get(label, "badge-unusable")
    return f'<span class="badge {css}">{esc(label)}</span>'


def stat_cards(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="card"><div class="label">{esc(name)}</div><div class="value">{esc(value)}</div></div>'
        for name, value in items
    )
    render(f'<div class="grid">{cells}</div>')


def bar_cards(items: list[tuple[str, float, str]]) -> None:
    cells = "".join(
        f'<div class="card"><div class="label">{esc(name)}</div><div class="value">{esc(text)}</div>'
        f'<div class="bar"><span style="width:{max(0.0, min(value, 1.0)) * 100:.2f}%"></span></div></div>'
        for name, value, text in items
    )
    render(f'<div class="grid">{cells}</div>')


def news_item(headline: str, label: str, score, source: str = "", when: str = "",
              link: str = "", preview: str = "") -> str:
    title = esc(headline)
    if link.startswith("https://"):
        title = (
            f'<a class="news-title" href="{esc(link)}" target="_blank" '
            f'rel="noopener noreferrer">{title}</a>'
        )
    else:
        title = f'<span class="news-title">{title}</span>'
    score_text = "" if score is None else f"{score:.3f}"
    meta = " · ".join(esc(part) for part in (source, when) if part)
    preview_html = f'<div class="preview">{esc(preview)}</div>' if preview else ""
    return (
        '<div class="news-item"><div class="news-meta">'
        f'{badge(label)}<span class="muted">{meta}</span><span class="score">{score_text}</span>'
        f'</div>{title}{preview_html}</div>'
    )


def verdict(label: str, detail: str, headline: str = "") -> None:
    css = {"Fake": "verdict-fake", "Real": "verdict-real"}.get(label, "verdict-unusable")
    headline_html = f'<p class="muted" style="color:#ffffff;opacity:.85">{esc(headline)}</p>' if headline else ""
    render(
        f'<div class="verdict {css}"><div class="label">Prediction</div>'
        f'<h3>{esc(label)}</h3><p>{esc(detail)}</p>{headline_html}</div>'
    )
