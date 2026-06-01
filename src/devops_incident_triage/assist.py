from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

from devops_incident_triage.retrieval import RetrievalResponse, RetrievedEvidence

ASSISTANT_MODE = "deterministic_beta"
LLM_ENABLED = False


@dataclass(frozen=True)
class AssistIncident:
    text: str
    predicted_domain: str
    classifier_confidence: float
    needs_human_review: bool
    recommended_queue: str


@dataclass(frozen=True)
class AssistRetrieval:
    query: str
    evidence: list[RetrievedEvidence]


@dataclass(frozen=True)
class RecommendedAction:
    action: str
    citation: str


@dataclass(frozen=True)
class AssistAssistantResponse:
    summary: str
    root_cause_candidates: list[str]
    recommended_actions: list[RecommendedAction]
    citations: list[str]
    safety_notes: list[str]


@dataclass(frozen=True)
class AssistMetadata:
    assistant_mode: str
    rag_enabled: bool
    llm_enabled: bool
    retrieval_latency_ms: float
    generation_latency_ms: float


@dataclass(frozen=True)
class AssistResponse:
    incident: AssistIncident
    retrieval: AssistRetrieval
    assistant_response: AssistAssistantResponse
    metadata: AssistMetadata


def build_assist_response(
    *,
    incident_text: str,
    predicted_domain: str,
    classifier_confidence: float,
    needs_human_review: bool,
    recommended_queue: str,
    retrieval: RetrievalResponse,
) -> AssistResponse:
    started_at = time.perf_counter()
    evidence = retrieval.evidence
    citations = _dedupe_citations(item.citation for item in evidence)
    route_note = (
        "Human review is required before routing this incident."
        if needs_human_review
        else "Review the cited runbook evidence before taking action."
    )
    summary = f"Initial triage points to the {predicted_domain} domain. {route_note}"

    root_cause_candidates = [
        f"{item.title} / {item.section} may explain the incident symptoms."
        for item in evidence
    ]
    recommended_actions = [
        RecommendedAction(
            action=(
                f"Review {item.title} / {item.section} and compare it with the "
                "current incident timeline."
            ),
            citation=item.citation,
        )
        for item in evidence
    ]

    safety_notes = [
        "This beta assistant does not execute remediation actions.",
        "Validate guidance with an operator before changing production systems.",
    ]

    generation_latency_ms = round((time.perf_counter() - started_at) * 1000, 3)

    return AssistResponse(
        incident=AssistIncident(
            text=incident_text,
            predicted_domain=predicted_domain,
            classifier_confidence=classifier_confidence,
            needs_human_review=needs_human_review,
            recommended_queue=recommended_queue,
        ),
        retrieval=AssistRetrieval(
            query=retrieval.retrieval_query,
            evidence=evidence,
        ),
        assistant_response=AssistAssistantResponse(
            summary=summary,
            root_cause_candidates=root_cause_candidates,
            recommended_actions=recommended_actions,
            citations=citations,
            safety_notes=safety_notes,
        ),
        metadata=AssistMetadata(
            assistant_mode=ASSISTANT_MODE,
            rag_enabled=bool(retrieval.metadata.get("rag_enabled", True)),
            llm_enabled=LLM_ENABLED,
            retrieval_latency_ms=float(retrieval.metadata.get("retrieval_latency_ms", 0.0)),
            generation_latency_ms=generation_latency_ms,
        ),
    )


def _dedupe_citations(citations: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for citation in citations:
        if not isinstance(citation, str) or citation in seen:
            continue
        seen.add(citation)
        deduped.append(citation)
    return deduped
