import io
import json

import pytest

from proseprobe import __version__, report
from proseprobe.cli import main

EVEN = (
    "The system works well. The system runs fast. The system stays calm. "
    "The system holds up. The system scales out."
)
BURSTY = (
    "Yes. The argument, which had been rehearsed for weeks in rooms that smelled "
    "of old coffee and older ambition, collapsed the moment anyone asked it a "
    "plain question about the data. Odd. Nobody minded much."
)


def _written(directory, name, text):
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_file_is_reported_as_sentences(tmp_path, capsys):
    path = _written(tmp_path, "even.txt", EVEN)
    assert main([str(path)]) == 0
    printed = capsys.readouterr().out
    assert "of 1.00" in printed
    assert "log-odds" in printed
    assert "Non-native English guard" in printed
    assert "even.txt:" not in printed


def test_json_prints_the_library_report(tmp_path, capsys):
    path = _written(tmp_path, "even.txt", EVEN)
    assert main([str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "even.txt"
    assert payload["path"] == str(path)
    assert payload["score"] == report(EVEN)["score"]


def test_several_files_each_keep_their_own_report(tmp_path, capsys):
    first = _written(tmp_path, "a.txt", EVEN)
    second = _written(tmp_path, "b.txt", BURSTY)
    assert main([str(first), str(second), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["name"] for item in payload] == ["a.txt", "b.txt"]

    assert main([str(first), str(second)]) == 0
    printed = capsys.readouterr().out
    assert "a.txt:" in printed
    assert "b.txt:" in printed


def test_standard_input_is_read_when_no_file_is_named(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(EVEN))
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "-"
    assert payload["path"] == ""


def test_an_unreadable_file_is_named_and_the_rest_is_scored(tmp_path, capsys):
    path = _written(tmp_path, "even.txt", EVEN)
    assert main([str(tmp_path / "missing.txt"), str(path), "--json"]) == 1
    captured = capsys.readouterr()
    assert "missing.txt" in captured.err
    assert json.loads(captured.out)["name"] == "even.txt"


def test_the_requested_confidence_travels_into_the_report(tmp_path, capsys):
    path = _written(tmp_path, "even.txt", EVEN)
    assert main([str(path), "--json", "--confidence", "0.8"]) == 0
    assert json.loads(capsys.readouterr().out)["score"]["interval"]["confidence"] == 0.8


def test_a_limit_keeps_only_the_largest_shifts(tmp_path, capsys):
    path = _written(tmp_path, "bursty.txt", BURSTY)
    assert main([str(path), "--limit", "2"]) == 0
    assert "smaller shifts" in capsys.readouterr().out


def test_zero_indentation_prints_one_line(tmp_path, capsys):
    path = _written(tmp_path, "even.txt", EVEN)
    assert main([str(path), "--json", "--indent", "0"]) == 0
    printed = capsys.readouterr().out
    assert printed.count("\n") == 1
    assert json.loads(printed)["name"] == "even.txt"


def test_a_confidence_outside_the_scale_is_refused(tmp_path, capsys):
    path = _written(tmp_path, "even.txt", EVEN)
    with pytest.raises(SystemExit) as refused:
        main([str(path), "--confidence", "1.5"])
    assert refused.value.code == 2
    assert "between 0 and 1" in capsys.readouterr().err


def test_the_version_is_printed(capsys):
    with pytest.raises(SystemExit) as done:
        main(["--version"])
    assert done.value.code == 0
    assert __version__ in capsys.readouterr().out
