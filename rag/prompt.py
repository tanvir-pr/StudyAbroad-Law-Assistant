"""Versioned generation prompts. Change the file, bump the version in rag.yaml."""

from pathlib import Path

from ingestion.sources import ROOT

PROMPTS_DIR = ROOT / "rag" / "prompts"

SYSTEM_V1 = """You are StudyAbroad Law Assistant. You explain public official rules about studying in Hungary. This is not legal advice.

Rules:
- Use only the CONTEXT passages. Do not invent fees, deadlines, hours, or institutions.
- Every factual sentence must end with an inline citation like [source_id](source_url) using values from CONTEXT.
- If a CONTEXT passage answers the question, you must answer. Do not refuse when the fact is present.
- If CONTEXT does not contain the answer, reply with exactly: REFUSE: I cannot cite an official source from the locked allow-list for this question. Check OIF, Tempus Public Foundation, or the university.
- Keep the answer short. End with: Not legal advice."""

USER_V1 = """CONTEXT:
{context}

QUESTION: {question}

Write the answer now."""

# On-disk v1.txt keeps a single template for inspection; runtime uses system+user chat turns.
PROMPT_V1 = SYSTEM_V1 + "\n\n" + USER_V1


def prompt_path(version: str) -> Path:
    return PROMPTS_DIR / f"{version}.txt"


def load_prompt(version: str = "v1") -> str:
    path = prompt_path(version)
    if path.exists():
        return path.read_text(encoding="utf-8")
    if version == "v1":
        return PROMPT_V1
    raise FileNotFoundError(f"Unknown prompt version: {version}")


def format_context(hits: list[dict]) -> str:
    if not hits:
        return "(no passages retrieved)"
    blocks = []
    for i, hit in enumerate(hits, start=1):
        updated = hit.get("last_updated") or "not stated on the source metadata"
        blocks.append(
            f"[{i}] source_id={hit['source_id']}\n"
            f"source_url={hit['source_url']}\n"
            f"title={hit.get('title')}\n"
            f"last_updated={updated}\n"
            f"score={hit.get('score', 0):.3f}\n"
            f"text={hit['text']}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[dict], version: str = "v1") -> list[dict]:
    if version != "v1":
        template = load_prompt(version)
        user = template.format(context=format_context(hits), question=question.strip())
        return [{"role": "user", "content": user}]
    return [
        {"role": "system", "content": SYSTEM_V1},
        {
            "role": "user",
            "content": USER_V1.format(context=format_context(hits), question=question.strip()),
        },
    ]


def ensure_prompt_file(version: str = "v1") -> Path:
    path = prompt_path(version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PROMPT_V1 if version == "v1" else load_prompt(version), encoding="utf-8")
    return path
