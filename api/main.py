"""FastAPI app: /health and /ask skeleton (day 17). Not legal advice."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ingestion.sources import ROOT
from rag.ask import ask
from rag.bm25 import Bm25Index
from rag.generate import EchoGenerator, GenerationConfig, QwenGenerator
from rag.index_faiss import load_rag_config
from rag.rerank import OverlapReranker

DISCLAIMER = (
    "Not legal advice. Answers cite public allow-list pages only. "
    "Confirm with OIF, Tempus, or your university before you act."
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    country: str | None = None
    institution: str | None = None
    mode: str | None = Field(default=None, description="hybrid | dense | bm25")


class Citation(BaseModel):
    source_id: str
    source_url: str


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    citations: list[Citation]
    model_name: str
    generation_mode: str
    prompt_version: str
    disclaimer: str = DISCLAIMER


class AppState:
    embedder: Any = None
    generator: Any = None
    bm25: Bm25Index | None = None
    reranker: Any = None
    config: dict = {}
    index_dir: Path = ROOT / "data" / "index"
    model_name: str = "echo"
    generation_mode: str = "echo"


STATE = AppState()


def _use_echo() -> bool:
    return os.environ.get("ASK_ECHO", "1") not in {"0", "false", "False"}


@lru_cache(maxsize=1)
def _load_embedder(model_name: str, batch_size: int):
    from rag.embed import BgeM3Embedder

    return BgeM3Embedder(model_name, batch_size=batch_size)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_rag_config()
    STATE.config = cfg
    STATE.index_dir = ROOT / cfg["index_dir"]
    if _use_echo():
        # Fast local smoke: hash-sized stand-in is not used; BGE still needed for real index.
        # For API unit tests without an index, ask() may refuse — that is OK for skeleton demos.
        try:
            STATE.embedder = _load_embedder(cfg["embedding_model"], int(cfg.get("batch_size", 8)))
        except Exception:
            from rag.index_faiss import HashEmbedder

            STATE.embedder = HashEmbedder(dim=32)
        STATE.generator = EchoGenerator()
        STATE.model_name = "echo"
        STATE.generation_mode = "echo"
        STATE.reranker = OverlapReranker()
    else:
        STATE.embedder = _load_embedder(cfg["embedding_model"], int(cfg.get("batch_size", 8)))
        gen_cfg = GenerationConfig(
            model_name=cfg["generation"]["model"],
            load_in_4bit=bool(cfg["generation"].get("load_in_4bit", True)),
            max_new_tokens=int(cfg["generation"]["max_new_tokens"]),
            temperature=float(cfg["generation"]["temperature"]),
        )
        STATE.generator = QwenGenerator(gen_cfg)
        STATE.model_name = gen_cfg.model_name
        STATE.generation_mode = STATE.generator.mode
        STATE.reranker = OverlapReranker()
    if (STATE.index_dir / "chunks.jsonl").exists():
        STATE.bm25 = Bm25Index.from_index_dir(STATE.index_dir)
    yield


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=20000)
    source_id: str | None = None
    source_url: str | None = None
    last_updated: str | None = None


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    source_lang: str = Field(default="en", description="en or hu")


class ChangesRequest(BaseModel):
    old_text: str = Field(..., min_length=10)
    new_text: str = Field(..., min_length=10)
    source_id: str | None = None
    source_url: str | None = None
    last_updated: str | None = None


app = FastAPI(
    title="StudyAbroad Law Assistant API",
    version="0.5.0",
    description=DISCLAIMER,
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "echo": _use_echo(),
        "index_dir": str(STATE.index_dir),
        "index_present": (STATE.index_dir / "chunks.faiss").exists(),
        "disclaimer": DISCLAIMER,
    }


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest) -> AskResponse:
    if STATE.embedder is None or STATE.generator is None:
        raise HTTPException(status_code=503, detail="Model stack not loaded")
    cfg = STATE.config or load_rag_config()
    hybrid = cfg.get("hybrid", {})
    rerank_cfg = cfg.get("rerank", {})
    try:
        result = ask(
            body.question,
            STATE.embedder,
            STATE.generator,
            STATE.index_dir,
            top_k=int(cfg.get("top_k", 5)),
            min_score=float(cfg.get("generation", {}).get("min_score", 0.0)),
            prompt_version=str(cfg.get("generation", {}).get("prompt_version", "v1")),
            model_name=STATE.model_name,
            generation_mode=STATE.generation_mode,
            retrieve_mode=body.mode or str(hybrid.get("mode", "hybrid")),
            country=body.country,
            institution=body.institution,
            candidate_k=int(hybrid.get("candidate_k", 30)),
            rrf_k=int(hybrid.get("rrf_k", 60)),
            dense_weight=float(hybrid.get("dense_weight", 1.0)),
            bm25_weight=float(hybrid.get("bm25_weight", 1.0)),
            bm25_index=STATE.bm25,
            reranker=STATE.reranker,
            rerank_candidates=int(rerank_cfg.get("candidates", 20)),
        )
    except Exception as exc:  # index missing, dim mismatch, etc.
        raise HTTPException(status_code=500, detail=f"ask failed: {exc}") from exc
    return AskResponse(
        question=result.question,
        answer=result.answer,
        refused=result.refused,
        citations=[Citation(**c) for c in result.citations],
        model_name=result.model_name,
        generation_mode=result.generation_mode,
        prompt_version=result.prompt_version,
    )


@app.post("/summarize")
def summarize_endpoint(body: SummarizeRequest) -> dict:
    from t5.rouge_eval import extractive_summary

    summary = extractive_summary(body.text, n_lines=5)
    return {
        "summary": summary,
        "source_id": body.source_id,
        "source_url": body.source_url,
        "last_updated": body.last_updated or "not stated on the source metadata",
        "disclaimer": DISCLAIMER,
    }


@app.post("/translate")
def translate_endpoint(body: TranslateRequest) -> dict:
    from t5.translate_bn import gloss_translate

    if body.source_lang.lower() not in {"en", "hu"}:
        raise HTTPException(status_code=400, detail="source_lang must be en or hu")
    return {
        "source_lang": body.source_lang.lower(),
        "target_lang": "bn",
        "translation": gloss_translate(body.text),
        "disclaimer": DISCLAIMER,
    }


@app.post("/changes")
def changes_endpoint(body: ChangesRequest) -> dict:
    from t5.what_changed import diff_changes

    return {
        "changes": diff_changes(body.old_text, body.new_text),
        "source_id": body.source_id,
        "source_url": body.source_url,
        "last_updated": body.last_updated or "not stated on the source metadata",
        "disclaimer": DISCLAIMER,
    }


@app.get("/")
def root() -> dict:
    return {
        "service": "StudyAbroad Law Assistant",
        "endpoints": ["/health", "/ask", "/summarize", "/translate", "/changes"],
        "disclaimer": DISCLAIMER,
    }


def create_app() -> FastAPI:
    return app
