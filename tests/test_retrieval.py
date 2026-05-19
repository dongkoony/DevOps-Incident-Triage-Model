from devops_incident_triage.retrieval import (
    DEFAULT_RUNBOOK_DIR,
    RunbookRetriever,
    load_runbook_corpus,
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
    assert response.evidence[0].document_id == "runbook-kubernetes"
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
