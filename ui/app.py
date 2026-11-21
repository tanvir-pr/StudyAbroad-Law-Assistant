"""Streamlit UI (day 21 / v0.5): ask + summarize/translate/changes with disclaimer."""

from __future__ import annotations

import os

import streamlit as st

from t5.rouge_eval import extractive_summary
from t5.translate_bn import gloss_translate
from t5.what_changed import diff_changes

DISCLAIMER = (
    "Not legal advice. Answers cite public allow-list pages only. "
    "Confirm with OIF, Tempus, or your university before you act."
)

st.set_page_config(page_title="StudyAbroad Law Assistant", layout="centered")
st.title("StudyAbroad Law Assistant")
st.caption(DISCLAIMER)

tab_ask, tab_sum, tab_tr, tab_ch = st.tabs(["Ask", "Summarize", "Translate → BN", "What changed"])

with tab_ask:
    st.subheader("Ask (via API or local refuse demo)")
    question = st.text_area("Question", placeholder="How many hours may a student work during studies?")
    api_base = st.text_input("API base URL", value=os.environ.get("API_BASE", "http://127.0.0.1:8000"))
    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            try:
                import httpx

                resp = httpx.post(f"{api_base.rstrip('/')}/ask", json={"question": question}, timeout=120.0)
                data = resp.json()
                if resp.status_code >= 400:
                    st.error(data)
                else:
                    st.write(data.get("answer", ""))
                    st.write(f"**Refused:** {data.get('refused')}")
                    cites = data.get("citations") or []
                    if cites:
                        st.write("**Citations**")
                        for cite in cites:
                            st.markdown(f"- `{cite.get('source_id')}` — {cite.get('source_url')}")
                    st.write("**Last-updated:** not stated on the source metadata (see retrieved chunk metadata in API logs / RAG hits).")
                    st.info(data.get("disclaimer", DISCLAIMER))
            except Exception as exc:
                st.error(f"API call failed: {exc}. Start uvicorn api.main:app first.")

with tab_sum:
    text = st.text_area("Document text to summarize", height=200)
    source_id = st.text_input("source_id", value="oif-student-residence-en")
    source_url = st.text_input("source_url", value="https://oif.gov.hu/factsheets/residence-of-the-student-pupil")
    last_updated = st.text_input("last_updated", value="not stated on the source metadata")
    if st.button("Summarize"):
        if len(text.strip()) < 20:
            st.warning("Paste more text.")
        else:
            st.write(extractive_summary(text, n_lines=5))
            st.markdown(f"**Citation:** [{source_id}]({source_url})")
            st.write(f"**Last-updated:** {last_updated}")
            st.info(DISCLAIMER)

with tab_tr:
    src = st.text_area("English or Hungarian text", height=150)
    lang = st.selectbox("source_lang", ["en", "hu"])
    if st.button("Translate to Bangla"):
        st.write(gloss_translate(src))
        st.info(DISCLAIMER)

with tab_ch:
    old = st.text_area("Old text", height=120)
    new = st.text_area("New text", height=120)
    sid = st.text_input("change source_id", value="sh-operational-regulations")
    surl = st.text_input(
        "change source_url",
        value="https://stipendiumhungaricum.hu/documents/stipendium_hungaricum/OR/SH_OR.pdf",
    )
    if st.button("Show changes"):
        for item in diff_changes(old, new):
            st.write(f"- {item}")
        st.markdown(f"**Citation:** [{sid}]({surl})")
        st.write("**Last-updated:** not stated on the source metadata")
        st.info(DISCLAIMER)

st.divider()
st.write(DISCLAIMER)
