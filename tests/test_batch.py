import json
import math

from proseprobe import (
    BANDS,
    CAUTIONS,
    SUMMARY_FILENAME,
    score_directory,
    score_documents,
    score_paths,
    write_reports,
)

EVEN = (
    "The system works well. The system runs fast. The system stays calm. "
    "The system holds up. The system scales out."
)
BURSTY = (
    "Yes. The argument, which had been rehearsed for weeks in rooms that smelled "
    "of old coffee and older ambition, collapsed the moment anyone asked it a "
    "plain question about the data. Odd. Nobody minded much."
)


def _booleans(node):
    if isinstance(node, bool):
        return [node]
    if isinstance(node, dict):
        return [found for value in node.values() for found in _booleans(value)]
    if isinstance(node, list):
        return [found for value in node for found in _booleans(value)]
    return []


def test_documents_keep_their_names_and_their_scores():
    batch = score_documents([("even.txt", EVEN), ("bursty.txt", BURSTY)])
    assert [document.name for document in batch.documents] == ["even.txt", "bursty.txt"]
    assert batch.unread == ()
    for document in batch.documents:
        result = document.score
        assert 0.0 <= result.low <= result.value <= result.high <= 1.0
        assert document.words > 0
    assert "2 documents" in batch.headline()
    assert [item.name for item in score_documents({"a": EVEN}).documents] == ["a"]


def test_a_document_report_carries_the_whole_score():
    document = score_documents([("even.txt", EVEN)]).documents[0]
    payload = json.loads(document.to_json())
    assert payload["name"] == "even.txt"
    assert payload["score"]["band"]["name"] in [band.name for band in BANDS]
    assert payload["score"]["notes"]
    assert payload["score"]["guard"]["words"] == document.score.guard.words
    assert payload["explanation"].startswith(document.score.summary())

    contributions = payload["score"]["contributions"]
    shifts = [row["logit_shift"] for row in contributions]
    value = payload["score"]["value"]
    assert shifts
    assert math.isclose(sum(shifts), math.log(value / (1.0 - value)), abs_tol=1e-4)
    assert any(row["evidence"] for row in contributions)


def test_the_summary_counts_how_the_scores_fall():
    batch = score_documents([("even.txt", EVEN), ("bursty.txt", BURSTY)])
    summary = json.loads(batch.to_json())
    assert summary["documents"] == 2
    assert summary["words"] > 0
    assert set(summary["bands"]) == {band.name for band in BANDS}
    assert sum(summary["bands"].values()) == 2
    assert set(summary["cautions"]) == {caution.name for caution in CAUTIONS}
    assert sum(summary["cautions"].values()) == 2
    scale = summary["scale"]
    assert scale["lowest"] <= scale["mean"] <= scale["highest"]
    movers = summary["features_moved"]
    assert movers
    assert movers[0]["mean_absolute_shift"] >= movers[-1]["mean_absolute_shift"]
    assert any("labels no" in note for note in summary["notes"])


def test_no_report_carries_a_boolean_verdict():
    batch = score_documents([("even.txt", EVEN)])
    assert not _booleans(batch.summary())
    assert not _booleans(batch.documents[0].to_dict())


def test_one_file_is_written_per_document_plus_a_summary(tmp_path):
    batch = score_documents([("even.txt", EVEN), ("bursty.txt", BURSTY)])
    target = tmp_path / "reports"
    written = write_reports(batch, target)
    assert len(written) == 3
    assert sorted(path.name for path in target.iterdir()) == [
        "bursty.txt.json",
        "even.txt.json",
        SUMMARY_FILENAME,
    ]

    summary = json.loads((target / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert len(summary["reports"]) == 2
    for entry in summary["reports"]:
        payload = json.loads((target / entry["file"]).read_text(encoding="utf-8"))
        assert payload["name"] == entry["name"]
        assert payload["score"]["value"] == entry["value"]


def test_report_filenames_are_unique_and_spare_the_summary():
    batch = score_documents([("a/b.txt", EVEN), ("a-b.txt", EVEN), ("summary", EVEN)])
    names = batch.filenames()
    assert len(set(names)) == 3
    assert SUMMARY_FILENAME not in names


def test_a_directory_is_read_recursively_and_unreadable_files_are_named(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_text(EVEN, encoding="utf-8")
    (tmp_path / "nested" / "b.md").write_text(BURSTY, encoding="utf-8")
    (tmp_path / "c.rst").write_text(EVEN, encoding="utf-8")
    (tmp_path / "broken.txt").write_bytes(b"\xff\xfe\x00bad")

    batch = score_directory(tmp_path)
    assert [document.name for document in batch.documents] == ["a.txt", "nested/b.md"]
    assert [record.name for record in batch.unread] == ["broken.txt"]
    assert batch.unread[0].reason
    assert "could not be read" in batch.headline()
    assert any("broken.txt" in note for note in batch.notes())
    assert batch.summary()["unread"][0]["name"] == "broken.txt"


def test_paths_can_be_given_as_a_list(tmp_path):
    first = tmp_path / "one.txt"
    second = tmp_path / "two.md"
    first.write_text(EVEN, encoding="utf-8")
    second.write_text(BURSTY, encoding="utf-8")

    batch = score_paths([first, second])
    assert [document.name for document in batch.documents] == ["one.txt", "two.md"]
    assert [document.path for document in batch.documents] == [str(first), str(second)]


def test_an_empty_batch_says_so_instead_of_failing(tmp_path):
    batch = score_documents([])
    summary = batch.summary()
    assert summary["documents"] == 0
    assert summary["scale"]["mean"] is None
    assert "No document" in summary["headline"]
    assert [path.name for path in write_reports(batch, tmp_path / "out")] == [
        SUMMARY_FILENAME
    ]


def test_the_requested_confidence_travels_into_the_reports():
    batch = score_documents([("even.txt", EVEN)], confidence=0.8)
    assert batch.confidence == 0.8
    assert batch.summary()["confidence"] == 0.8
    assert batch.documents[0].to_dict()["score"]["interval"]["confidence"] == 0.8
