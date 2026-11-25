from copilot.corpus import enforcement_to_doc, label_to_doc, shortage_to_doc


def test_label_becomes_document_with_storage_text():
    rec = {"set_id": "abc-123", "openfda": {"generic_name": ["ADALIMUMAB"]},
           "storage_and_handling": ["Store refrigerated at 2°C to 8°C."],
           "inactive_ingredient": ["mannitol, polysorbate 80"]}
    doc = label_to_doc(rec)
    assert doc["id"] == "label-abc-123"
    assert "2°C to 8°C" in doc["text"] and "mannitol" in doc["text"]


def test_label_without_text_is_skipped():
    assert label_to_doc({"set_id": "x", "openfda": {}}) is None


def test_shortage_and_recall_documents():
    s = shortage_to_doc({"generic_name": "CISPLATIN", "status": "Currently in Shortage"})
    assert "Currently in Shortage" in s["text"]
    r = enforcement_to_doc({"recall_number": "D-0001-2026", "classification": "Class I",
                            "reason_for_recall": "temperature abuse"})
    assert r["id"] == "recall-D-0001-2026" and "temperature abuse" in r["text"]


def test_fetch_labels_saves_raw_with_hash_and_builds_docs(monkeypatch, tmp_path):
    import copilot.corpus as C
    monkeypatch.setattr(C, "RAW", tmp_path)
    sample = {"results": [{"set_id": "s1", "openfda": {"generic_name": ["CISPLATIN"]},
                           "storage_and_handling": ["Store at 15 to 25 C. Do not refrigerate."]}]}
    monkeypatch.setattr(C, "_get", lambda path, params, retries=3: sample)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    docs = C.fetch_labels(["cisplatin"], per_name=1)
    assert docs and docs[0]["id"] == "label-s1"
    manifest = __import__("json").loads((tmp_path / "manifest.json").read_text())
    assert any("sha256" in v for v in manifest.values())


def test_long_document_is_split_into_titled_passages():
    from copilot.corpus import chunk_document
    doc = {"id": "label-x", "text": "ADALIMUMAB\n" + "Store refrigerated. " * 200, "source": "s", "url": "u"}
    chunks = chunk_document(doc, chunk_chars=300)
    assert len(chunks) > 1
    assert all(c["text"].startswith("ADALIMUMAB\n") and len(c["text"]) <= 300 + len("ADALIMUMAB\n") for c in chunks)
    assert [c["id"] for c in chunks[:2]] == ["label-x#c0", "label-x#c1"] and chunks[0]["parent"] == "label-x"
    assert chunk_document(chunks[0], 300) == [chunks[0]]          # re-running does not split again
