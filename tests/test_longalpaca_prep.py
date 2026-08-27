import json
import sys

from scripts.probe.prepare_longalpaca_decode_calib import main


def test_prepare_longalpaca_decode_calib_converts_and_limits_rows(tmp_path, monkeypatch):
    source = tmp_path / "LongAlpaca-12k.json"
    output = tmp_path / "alpaca12k_decode_calib.jsonl"
    source.write_text(
        json.dumps(
            [
                {"instruction": "  explain sparse routing  ", "output": " answer one "},
                {"instruction": "second prompt", "output": "answer two"},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_longalpaca_decode_calib.py", "--input", str(source), "--output", str(output), "--limit", "1"],
    )

    main()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "input": "",
            "context": "explain sparse routing",
            "answers": ["answer one"],
            "all_classes": None,
            "length": 3,
            "dataset": "alpaca12k_decode_calib",
            "_id": "0",
        }
    ]
