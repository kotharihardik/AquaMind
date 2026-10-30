"""The local reader's disk cache: a repeated read is answered from disk, a failure is never cached, a different model never reuses a read."""
import json

import aquamind.reader.local_reader as lr


def fake(monkeypatch, tmp_path, answers, mid="m1"):
    monkeypatch.setattr(lr, "CACHE_DIR", str(tmp_path))
    r = object.__new__(lr.LocalReader)
    r.calls = r.errors = r.cached = 0
    r._mid = mid
    seen = []

    def raw(system, user):
        seen.append(user)
        return answers.pop(0)
    r._gen_raw = raw
    return r, seen


def test_second_read_comes_from_disk_without_calling_the_model(monkeypatch, tmp_path):
    r, seen = fake(monkeypatch, tmp_path, [{"category": "leak"}])
    assert r._gen("sys", "text A") == {"category": "leak"} and r._gen("sys", "text A") == {"category": "leak"}
    assert len(seen) == 1 and r.cached == 1 and r.calls == 2


def test_a_failed_read_is_not_cached(monkeypatch, tmp_path):
    r, seen = fake(monkeypatch, tmp_path, [None, {"category": "other"}])
    assert r._gen("sys", "text B") is None
    assert r._gen("sys", "text B") == {"category": "other"} and len(seen) == 2 and r.errors == 1          # asked again, then stored


def test_other_text_prompt_or_model_is_a_miss(monkeypatch, tmp_path):
    r, seen = fake(monkeypatch, tmp_path, [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}])
    r._gen("sys", "t1"); r._gen("sys", "t2"); r._gen("other prompt", "t1")
    assert len(seen) == 3
    r2, seen2 = fake(monkeypatch, tmp_path, [{"a": 9}], mid="a re-trained model")
    assert r2._gen("sys", "t1") == {"a": 9} and len(seen2) == 1


def test_cache_files_are_plain_json(monkeypatch, tmp_path):
    r, _ = fake(monkeypatch, tmp_path, [{"places": [{"text": "Station Rd", "type": "street"}]}])
    r._gen("s", "u")
    files = list(tmp_path.glob("*/*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["places"][0]["text"] == "Station Rd" and not list(tmp_path.glob("**/*.tmp"))
