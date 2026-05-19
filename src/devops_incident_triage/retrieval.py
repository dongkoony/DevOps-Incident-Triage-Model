from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_RUNBOOK_DIR = Path("docs/runbooks")
EMBEDDING_MODEL = "scikit-learn-tfidf-preview"
INDEX_TYPE = "in_memory_sparse_vector_index"
DOMAIN_BOOST = 0.15

RUNBOOK_DOMAIN_BY_FILENAME = {
    "aws-iam-network.md": "aws_iam_network",
    "cicd.md": "cicd_pipeline",
    "container-runtime.md": "container_runtime",
    "database.md": "database_state",
    "deployment-release.md": "deployment_release",
    "kubernetes.md": "k8s_cluster",
    "observability.md": "observability_alerting",
}


@dataclass(frozen=True)
class RunbookSection:
    document_id: str
    domain: str
    title: str
    section: str
    text: str
    citation: str


@dataclass(frozen=True)
class RetrievedEvidence:
    document_id: str
    domain: str
    title: str
    section: str
    score: float
    citation: str
    excerpt: str


@dataclass(frozen=True)
class RetrievalResponse:
    predicted_domain: str
    retrieval_query: str
    evidence: list[RetrievedEvidence]
    metadata: dict[str, Any]


class RetrievalConfigurationError(RuntimeError):
    """Raised when the local preview corpus cannot be loaded."""


def _slugify_heading(heading: str) -> str:
    slug = re.sub(r"[^a-z0-9\s-]", "", heading.lower())
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug


def _document_id_from_path(path: Path) -> str:
    return f"runbook-{path.stem}"


def _domain_from_path(path: Path) -> str:
    return RUNBOOK_DOMAIN_BY_FILENAME.get(path.name, "unknown")


def _clean_excerpt(text: str, max_length: int = 220) -> str:
    normalized = " ".join(line.strip("- ").strip() for line in text.splitlines() if line.strip())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3].rstrip()}..."


def _parse_markdown_sections(path: Path) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Overview"
    current_lines: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = line.removeprefix("## ").strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    return [
        (heading, "\n".join(lines).strip())
        for heading, lines in sections
        if "\n".join(lines).strip()
    ]


def load_runbook_corpus(runbook_dir: Path = DEFAULT_RUNBOOK_DIR) -> list[RunbookSection]:
    if not runbook_dir.exists():
        raise RetrievalConfigurationError(f"Runbook directory not found: {runbook_dir}")

    corpus: list[RunbookSection] = []
    for path in sorted(runbook_dir.glob("*.md")):
        domain = _domain_from_path(path)
        document_id = _document_id_from_path(path)
        title = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line.removeprefix("# ").strip()
                break
        if not title:
            title = path.stem.replace("-", " ").title()

        for heading, text in _parse_markdown_sections(path):
            corpus.append(
                RunbookSection(
                    document_id=document_id,
                    domain=domain,
                    title=title,
                    section=heading,
                    text=text,
                    citation=f"{path.as_posix()}#{_slugify_heading(heading)}",
                )
            )

    if not corpus:
        raise RetrievalConfigurationError(f"No runbook sections found in {runbook_dir}")
    return corpus


class RunbookRetriever:
    def __init__(self, corpus: list[RunbookSection]) -> None:
        if not corpus:
            raise RetrievalConfigurationError("RunbookRetriever requires at least one corpus item.")
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self._index_text(item) for item in corpus)

    @classmethod
    def from_runbook_dir(cls, runbook_dir: Path = DEFAULT_RUNBOOK_DIR) -> RunbookRetriever:
        return cls(load_runbook_corpus(runbook_dir))

    def retrieve(self, text: str, predicted_domain: str, top_k: int = 5) -> RetrievalResponse:
        started_at = time.perf_counter()
        safe_top_k = max(1, min(top_k, len(self.corpus)))
        query_vector = self.vectorizer.transform([text])
        similarities = cosine_similarity(query_vector, self.matrix)[0]

        ranked: list[tuple[float, RunbookSection]] = []
        for similarity, section in zip(similarities, self.corpus, strict=True):
            score = float(similarity)
            if section.domain == predicted_domain:
                score += DOMAIN_BOOST
            ranked.append((score, section))

        ranked.sort(key=lambda item: item[0], reverse=True)
        evidence = [
            RetrievedEvidence(
                document_id=section.document_id,
                domain=section.domain,
                title=section.title,
                section=section.section,
                score=round(score, 6),
                citation=section.citation,
                excerpt=_clean_excerpt(section.text),
            )
            for score, section in ranked[:safe_top_k]
        ]

        return RetrievalResponse(
            predicted_domain=predicted_domain,
            retrieval_query=text,
            evidence=evidence,
            metadata={
                "embedding_model": EMBEDDING_MODEL,
                "index_type": INDEX_TYPE,
                "rag_enabled": True,
                "retrieval_latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
            },
        )

    @staticmethod
    def _index_text(section: RunbookSection) -> str:
        return " ".join(
            [
                section.domain,
                section.title,
                section.section,
                section.text,
            ]
        )
