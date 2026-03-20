from devops_incident_triage.labels import INCIDENT_LABELS, id_to_label_map, label_to_id_map


def test_labels_are_unique() -> None:
    assert len(INCIDENT_LABELS) == len(set(INCIDENT_LABELS))


def test_label_id_roundtrip() -> None:
    label2id = label_to_id_map()
    id2label = id_to_label_map()
    assert len(label2id) == len(id2label)
    for label, idx in label2id.items():
        assert id2label[idx] == label
