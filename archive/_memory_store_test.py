"""MemoryStore persists profile/summary/session and recalls them.
Run: python -m archive._memory_store_test"""
import json
import tempfile
from pathlib import Path

from local_services.avatar_memory import MemoryStore


def main() -> None:
    d = tempfile.mkdtemp()
    m = MemoryStore(base_dir=d, enabled=True)

    # fresh store: empty recall, no greeting hint
    assert m.recall() == "" or isinstance(m.recall(), str)
    assert m.greeting_hint() is None

    # seed a profile + summary, persist, reload
    m.profile["default_city"] = "台北市"  # Taipei
    m.profile["name"] = "Ann"
    m.summary = "使用者常問台北天氣"
    m._save_profile()
    m._save_summary()

    m2 = MemoryStore(base_dir=d, enabled=True)
    assert m2.profile.get("default_city") == "台北市"
    assert "台北市" in m2.recall()
    assert m2.greeting_hint() is not None  # has a default_city -> personalized greeting

    # record_turn appends to session + jsonl
    m2.record_turn("今天天氣", "晴天")
    assert len(m2.session) == 1
    lines = Path(d, "session.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["user"] == "今天天氣"

    # reset clears both
    m2.reset_session()
    assert m2.session == []
    assert Path(d, "session.jsonl").read_text(encoding="utf-8") == ""

    # disabled store is inert
    off = MemoryStore(base_dir=tempfile.mkdtemp(), enabled=False)
    off.record_turn("a", "b")
    assert off.session == []
    print("PASS _memory_store_test")


if __name__ == "__main__":
    main()
