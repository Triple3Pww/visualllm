"""Avatar memory harness (fully local). Persists the virtual human's growing memory --
a durable profile, a rolling Chinese summary, and the live session log -- and rewrites
utterances into self-contained queries + distills conversations via local qwen.

Storage layout under base_dir (default state/avatar_memory/, gitignored):
  profile.json   durable facts {name, default_city, preferences[], notes}
  summary.txt    rolling zh summary of past conversations
  session.jsonl  current conversation turns ({user, bot, ts}), written live

Hardening: memory NEVER breaks a turn -- callers wrap these in try/except, and a
disabled store (enabled=False) is fully inert. ASCII-only logging (the console is cp1252).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

_DEFAULT_PROFILE = {"name": "", "default_city": "", "preferences": [], "notes": ""}

# Follow-up / ellipsis markers that signal the utterance leans on prior context.
_FOLLOWUP_MARKERS = (
    "那",      # na -- "what about..."
    "呢",      # ne -- trailing question particle
    "還有",  # also
    "同樣",  # likewise
    "一樣",  # same
    "剛",      # just now
    "這個",  # this one
    "那個",  # that one
    "它",      # it
)
# Taiwan city/county tokens; if none appears, the ask has no explicit location.
_TW_LOCATIONS = (
    "台北", "新北", "桃園", "台中", "台南",
    "高雄", "基隆", "新竹", "苗栗", "彰化",
    "南投", "雲林", "嘉義", "屏東", "宜蘭",
    "花蓮", "台東", "澎湖", "金門", "馬祖",
)


def needs_rewrite(text: str, profile: dict) -> bool:
    """Gate the (latency-costing) rewrite: only rewrite a context-dependent ask.

    True if the utterance has a follow-up marker, OR it names no location while we
    know the user's default_city (so the rewrite can fill it in). Otherwise skip --
    a self-contained query goes straight to the chain.
    """
    if any(mark in text for mark in _FOLLOWUP_MARKERS):
        return True
    has_location = any(loc in text for loc in _TW_LOCATIONS)
    if not has_location and profile.get("default_city"):
        return True
    return False


def _extract_json(text: str) -> Optional[dict]:
    """First JSON object in a model reply (handles ```json fences / prose around it)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


class MemoryStore:
    def __init__(
        self,
        *,
        base_dir: str,
        llm_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        gated: bool = True,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.base = Path(base_dir)
        self._llm_url = llm_url
        self._llm_model = llm_model
        self._gated = gated
        self._http: Optional[httpx.AsyncClient] = None
        self.profile = dict(_DEFAULT_PROFILE)
        self.summary = ""
        self.session: list[dict] = []
        if self.enabled:
            self.base.mkdir(parents=True, exist_ok=True)
            self._load()

    # ---- paths ----
    @property
    def _profile_path(self) -> Path:
        return self.base / "profile.json"

    @property
    def _summary_path(self) -> Path:
        return self.base / "summary.txt"

    @property
    def _session_path(self) -> Path:
        return self.base / "session.jsonl"

    # ---- load / save ----
    def _load(self) -> None:
        try:
            if self._profile_path.exists():
                loaded = json.loads(self._profile_path.read_text(encoding="utf-8"))
                self.profile = {**_DEFAULT_PROFILE, **loaded}
            if self._summary_path.exists():
                self.summary = self._summary_path.read_text(encoding="utf-8").strip()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"memory load failed ({type(e).__name__}); starting empty")

    def _save_profile(self) -> None:
        self._profile_path.write_text(
            json.dumps(self.profile, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _save_summary(self) -> None:
        self._summary_path.write_text(self.summary, encoding="utf-8")

    # ---- recall (context engineering input) ----
    def recall(self) -> str:
        """Compact zh context block fed to the rewrite/distill prompts."""
        bits = []
        city = self.profile.get("default_city")
        name = self.profile.get("name")
        prefs = self.profile.get("preferences") or []
        if name:
            bits.append(f"使用者名稱：{name}")            # user name
        if city:
            bits.append(f"使用者住在：{city}")            # lives in
        if prefs:
            bits.append("偏好：" + "、".join(map(str, prefs)))  # preferences
        if self.summary:
            bits.append(f"過往摘要：{self.summary}")          # past summary
        return "\n".join(bits)

    def greeting_hint(self) -> Optional[str]:
        """A zh greeting tail personalized from the profile, or None if nothing known."""
        city = self.profile.get("default_city")
        if city:
            return f"還是想看{city}的天氣嗎？"  # "Still want <city>'s weather?"
        return None

    # ---- session log ----
    def record_turn(self, user: str, bot: str) -> None:
        if not self.enabled:
            return
        turn = {"user": user, "bot": bot, "ts": time.time()}
        self.session.append(turn)
        with self._session_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")

    def reset_session(self) -> None:
        self.session = []
        if self.enabled:
            self._session_path.write_text("", encoding="utf-8")

    # ---- local-LLM client (Ollama, OpenAI-compatible) ----
    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=3.0))
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _chat(self, prompt: str, max_tokens: int) -> str:
        """One non-streaming completion from the local model. Raises on failure."""
        payload = {
            "model": self._llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        r = await self._client().post(self._llm_url + "/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    # ---- context engineering: rewrite the utterance into a self-contained query ----
    async def build_query(self, raw: str) -> str:
        if not self.enabled or not self._llm_url or not raw:
            return raw
        if self._gated and not needs_rewrite(raw, self.profile):
            return raw
        last = self.session[-1]["user"] if self.session else "（無）"  # "(none)"
        prompt = (
            "任務：把「目前問題」改寫成一句完整、可獨立查詢的繁體中文天氣問題。"
            "利用記憶與上一輪補上缺少的地點或時間。只輸出改寫後的問題，不要解釋。\n\n"
            f"記憶：{self.recall() or '（無）'}\n"
            f"上一輪：{last}\n"
            f"目前問題：{raw}\n改寫："
        )
        try:
            out = await self._chat(prompt, max_tokens=48)
        except Exception as e:  # noqa: BLE001 -- memory must never break a turn
            logger.warning(f"rewrite failed ({type(e).__name__}); using raw utterance")
            return raw
        out = out.splitlines()[0].strip().strip('"「」') if out else ""
        return out or raw

    # ---- harness: distill the conversation into durable memory ----
    async def _distill_turns(self, turns: list[dict]) -> None:
        """Fold a list of turns into the profile + summary via local qwen, then save.
        Shared by end-of-conversation (distill_and_save) and startup recovery
        (distill_pending). Never raises -- memory must not break the app."""
        if not self.enabled or not self._llm_url or not turns:
            return
        convo = "\n".join(f"使用者：{t['user']}\n助理：{t['bot']}" for t in turns)
        prompt = (
            "你是記憶整理助手。讀下面的對話，更新使用者記憶。"
            "只輸出一個 JSON，欄位：name、default_city、preferences(陣列)、summary(繁體中文一段話)。"
            "不確定的欄位保留舊值。\n\n"
            f"舊資料：name={self.profile.get('name')}, default_city={self.profile.get('default_city')}, "
            f"preferences={self.profile.get('preferences')}\n舊摘要：{self.summary or '（無）'}\n\n"
            f"對話：\n{convo}\n\nJSON："
        )
        try:
            out = await self._chat(prompt, max_tokens=400)
            data = _extract_json(out)
            if data:
                self.profile["name"] = data.get("name") or self.profile.get("name", "")
                self.profile["default_city"] = data.get("default_city") or self.profile.get("default_city", "")
                if isinstance(data.get("preferences"), list):
                    self.profile["preferences"] = data["preferences"]
                if data.get("summary"):
                    self.summary = str(data["summary"]).strip()
                self._save_profile()
                self._save_summary()
                logger.info("memory distilled + saved")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"distill failed ({type(e).__name__}); memory unchanged")

    async def distill_and_save(self) -> None:
        """End of conversation: distill the live session, then clear it + close."""
        try:
            await self._distill_turns(self.session)
        finally:
            self.reset_session()
            await self.aclose()

    def _read_session_file(self) -> list[dict]:
        """Leftover turns persisted live by record_turn (survive a hard crash)."""
        if not self.enabled or not self._session_path.exists():
            return []
        turns = []
        for line in self._session_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                turns.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return turns

    async def distill_pending(self) -> None:
        """Startup recovery: if a prior run crashed without a clean disconnect,
        session.jsonl still holds its turns. Fold them in + clear so the next
        conversation starts clean. Instant no-op when nothing is pending."""
        pending = self._read_session_file()
        if not pending:
            return
        logger.info(f"recovering {len(pending)} pending memory turn(s) from a prior session")
        try:
            await self._distill_turns(pending)
        finally:
            self.reset_session()
            await self.aclose()
