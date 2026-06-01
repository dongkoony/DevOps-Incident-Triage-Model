from devops_incident_triage.assist import build_assist_response
from devops_incident_triage.retrieval import RetrievalResponse, RetrievedEvidence


def test_build_assist_response_grounds_actions_in_evidence() -> None:
    retrieval = RetrievalResponse(
        predicted_domain="aws_iam_network",
        retrieval_query="GitHub Actions cannot assume production role.",
        evidence=[
            RetrievedEvidence(
                document_id="runbook-aws-iam-network",
                domain="aws_iam_network",
                title="AWS IAM And Network Runbook",
                section="First Checks",
                score=0.91,
                citation="docs/runbooks/aws-iam-network.md#first-checks",
                excerpt="Verify trust policy, OIDC provider, role ARN, and sts:AssumeRole.",
            )
        ],
        metadata={
            "embedding_model": "scikit-learn-tfidf-preview",
            "index_type": "in_memory_sparse_vector_index",
            "rag_enabled": True,
            "retrieval_latency_ms": 12.3,
        },
    )

    response = build_assist_response(
        incident_text="GitHub Actions cannot assume production role.",
        predicted_domain="aws_iam_network",
        classifier_confidence=0.82,
        needs_human_review=False,
        recommended_queue="aws_iam_network",
        retrieval=retrieval,
    )

    assert response.incident.predicted_domain == "aws_iam_network"
    assert response.incident.needs_human_review is False
    assert response.retrieval.evidence[0].citation == (
        "docs/runbooks/aws-iam-network.md#first-checks"
    )
    assert response.assistant_response.citations == [
        "docs/runbooks/aws-iam-network.md#first-checks"
    ]
    assert response.assistant_response.recommended_actions[0].citation == (
        "docs/runbooks/aws-iam-network.md#first-checks"
    )
    assert response.metadata.assistant_mode == "deterministic_beta"
    assert response.metadata.llm_enabled is False


def test_build_assist_response_deduplicates_citations() -> None:
    retrieval = RetrievalResponse(
        predicted_domain="k8s_cluster",
        retrieval_query="Nodes became NotReady after CNI upgrade.",
        evidence=[
            RetrievedEvidence(
                document_id="runbook-kubernetes",
                domain="k8s_cluster",
                title="Kubernetes Cluster Runbook",
                section="First Checks",
                score=0.92,
                citation="docs/runbooks/kubernetes.md#first-checks",
                excerpt="Check node readiness, CNI status, and recent events.",
            ),
            RetrievedEvidence(
                document_id="runbook-kubernetes",
                domain="k8s_cluster",
                title="Kubernetes Cluster Runbook",
                section="Useful Commands",
                score=0.88,
                citation="docs/runbooks/kubernetes.md#first-checks",
                excerpt="Use kubectl get nodes and kubectl describe node.",
            ),
        ],
        metadata={
            "embedding_model": "scikit-learn-tfidf-preview",
            "index_type": "in_memory_sparse_vector_index",
            "rag_enabled": True,
            "retrieval_latency_ms": 9.8,
        },
    )

    response = build_assist_response(
        incident_text="Nodes became NotReady after CNI upgrade.",
        predicted_domain="k8s_cluster",
        classifier_confidence=0.91,
        needs_human_review=False,
        recommended_queue="k8s_cluster",
        retrieval=retrieval,
    )

    assert response.assistant_response.citations == [
        "docs/runbooks/kubernetes.md#first-checks"
    ]
    assert len(response.assistant_response.recommended_actions) == 2
