import json

from proseprobe import REPORT_VERSION, report, report_score, score, score_documents

EVEN = (
    "The system works well. The system runs fast. The system stays calm. "
    "The system holds up. The system scales out."
)
BURSTY = (
    "Yes. The argument, which had been rehearsed for weeks in rooms that smelled "
    "of old coffee and older ambition, collapsed the moment anyone asked it a "
    "plain question about the data. Odd. Nobody minded much."
)


def test_one_call_renders_the_whole_score():
    payload = report(EVEN, name="even.txt", path="corpus/even.txt")
    assert payload["report_version"] == REPORT_VERSION
    assert payload["name"] == "even.txt"
    assert payload["path"] == "corpus/even.txt"
    assert payload["words"] > 0
    assert payload["score"]["band"]["name"]
    assert payload["score"]["contributions"]
    assert payload["score"]["guard"]["words"] > 0
    assert payload["score"]["notes"]
    assert payload["explanation"].startswith(payload["score"]["summary"])
    assert json.loads(json.dumps(payload)) == payload


def test_the_library_report_and_a_batch_document_report_have_one_shape():
    document = score_documents([("even.txt", EVEN)]).documents[0]
    assert report(EVEN, name="even.txt") == document.to_dict()


def test_a_score_already_in_hand_renders_the_same_way():
    rendered = report_score(score(EVEN), name="even.txt", words=20)
    assert rendered["score"] == report(EVEN, name="even.txt")["score"]
    assert rendered["words"] == 20


def test_the_requested_confidence_reaches_the_interval():
    payload = report(EVEN, confidence=0.8)
    assert payload["score"]["interval"]["confidence"] == 0.8
    assert payload["score"]["interval"]["width"] < report(EVEN)["score"]["interval"]["width"]


def test_a_limit_trims_the_explanation_and_leaves_the_score_whole():
    full = report(BURSTY)
    trimmed = report(BURSTY, limit=2)
    assert trimmed["score"] == full["score"]
    assert "smaller shifts" in trimmed["explanation"]
    assert "smaller shifts" not in full["explanation"]


def test_a_text_below_every_minimum_reports_the_prior():
    payload = report("Hello there.")
    assert payload["score"]["value"] == 0.5
    assert payload["score"]["contributions"] == []
    assert any("prior" in note for note in payload["score"]["notes"])
