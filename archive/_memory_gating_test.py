"""needs_rewrite fires on follow-ups / location-less asks, skips self-contained ones.
Run: python -m archive._memory_gating_test"""
from local_services.avatar_memory import needs_rewrite


def main() -> None:
    prof_city = {"default_city": "台北市"}
    prof_none = {"default_city": ""}

    # follow-up markers -> rewrite
    assert needs_rewrite("那台中呢？", prof_city) is True
    assert needs_rewrite("後天呢？", prof_city) is True
    # no location named + we know their city -> rewrite (fill it in)
    assert needs_rewrite("明天會下雨嗎？", prof_city) is True
    # self-contained (names a city) + nothing to add -> skip
    assert needs_rewrite("明天台南會下雨嗎？", prof_none) is False
    # location-less but no profile city to inject -> skip (nothing to add)
    assert needs_rewrite("明天會下雨嗎？", prof_none) is False
    print("PASS _memory_gating_test")


if __name__ == "__main__":
    main()
