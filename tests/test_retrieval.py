from pathlib import Path

from devops_incident_triage.retrieval import (
    DEFAULT_RUNBOOK_DIR,
    RunbookRetriever,
    load_runbook_corpus,
    parse_wiki_metadata,
)


def test_load_runbook_corpus_reads_section_level_evidence() -> None:
    corpus = load_runbook_corpus(DEFAULT_RUNBOOK_DIR)

    kubernetes = [
        item
        for item in corpus
        if item.document_id == "runbook-kubernetes" and item.section == "First Checks"
    ]

    assert kubernetes
    assert kubernetes[0].domain == "k8s_cluster"
    assert kubernetes[0].citation == "docs/runbooks/kubernetes.md#first-checks"
    assert "node readiness" in kubernetes[0].text.lower()


def test_retriever_prioritizes_predicted_domain_evidence() -> None:
    retriever = RunbookRetriever.from_runbook_dir(DEFAULT_RUNBOOK_DIR)

    response = retriever.retrieve(
        text="EKS worker nodes became NotReady after a CNI upgrade.",
        predicted_domain="k8s_cluster",
        top_k=3,
    )

    assert response.evidence
    assert response.evidence[0].domain == "k8s_cluster"
    assert response.evidence[0].document_id in {
        "runbook-kubernetes",
        "wiki-k8s-node-readiness",
    }
    assert response.evidence[0].score > 0


def test_retriever_limits_top_k() -> None:
    retriever = RunbookRetriever.from_runbook_dir(DEFAULT_RUNBOOK_DIR)

    response = retriever.retrieve(
        text="GitHub Actions runner cannot assume the production IAM role.",
        predicted_domain="aws_iam_network",
        top_k=2,
    )

    assert len(response.evidence) == 2
    assert response.evidence[0].domain == "aws_iam_network"


def test_retriever_response_includes_preview_metadata() -> None:
    retriever = RunbookRetriever.from_runbook_dir(DEFAULT_RUNBOOK_DIR)

    response = retriever.retrieve(
        text="Database writes are timing out because locks are waiting.",
        predicted_domain="database_state",
        top_k=1,
    )

    assert response.metadata["embedding_model"] == "scikit-learn-tfidf-preview"
    assert response.metadata["index_type"] == "in_memory_sparse_vector_index"
    assert response.metadata["rag_enabled"] is True


def test_parse_wiki_metadata_reads_required_fields(tmp_path: Path) -> None:
    wiki_page = tmp_path / "node-readiness.md"
    wiki_page.write_text(
        """# Kubernetes Node Readiness Wiki

## Wiki Metadata

- Wiki ID: `wiki-k8s-node-readiness`
- Source type: `runbook`
- Domain label: `k8s_cluster`
- Service: `kubernetes-platform`
- Severity: `medium`
- Owner: `platform-team`
- Last reviewed: `2026-06-02`
- Confidence level: `preview`

## Scope

Portfolio example for Kubernetes node readiness incidents.
""",
        encoding="utf-8",
    )

    metadata = parse_wiki_metadata(wiki_page)

    assert metadata.wiki_id == "wiki-k8s-node-readiness"
    assert metadata.source_type == "runbook"
    assert metadata.domain == "k8s_cluster"
    assert metadata.service == "kubernetes-platform"
    assert metadata.severity == "medium"
    assert metadata.owner == "platform-team"
    assert metadata.last_reviewed == "2026-06-02"
    assert metadata.confidence_level == "preview"


def test_load_runbook_corpus_includes_wiki_pages() -> None:
    corpus = load_runbook_corpus(DEFAULT_RUNBOOK_DIR)

    wiki_sections = [item for item in corpus if item.wiki_id == "wiki-k8s-node-readiness"]

    assert wiki_sections
    assert wiki_sections[0].document_id == "wiki-k8s-node-readiness"
    assert wiki_sections[0].source_type == "runbook"
    assert wiki_sections[0].service == "kubernetes-platform"
    assert wiki_sections[0].confidence_level == "preview"
    assert wiki_sections[0].citation.startswith("docs/wiki/kubernetes/node-readiness.md#")


def test_existing_runbooks_get_backward_compatible_wiki_defaults() -> None:
    corpus = load_runbook_corpus(DEFAULT_RUNBOOK_DIR)

    runbook = next(
        item
        for item in corpus
        if item.document_id == "runbook-kubernetes" and item.section == "First Checks"
    )

    assert runbook.wiki_id == "runbook-kubernetes"
    assert runbook.source_type == "runbook"
    assert runbook.service == "general"
    assert runbook.severity == "unknown"
    assert runbook.owner == "portfolio"
    assert runbook.last_reviewed == "unknown"
    assert runbook.confidence_level == "placeholder"
