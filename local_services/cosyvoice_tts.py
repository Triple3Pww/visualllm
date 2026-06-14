"""CosyVoice2 streaming TTS as a Pipecat service.

This is a thin HTTP client: it streams text to a local CosyVoice2 server (see
local_services/cosyvoice_server/) and yields audio chunks as soon as they
arrive, so the avatar can start lip-syncing on the first chunk (~150 ms).

The server returns raw 16-bit PCM mono at `sample_rate` (default 24 kHz, which
is CosyVoice2's native rate). Pipecat resamples downstream as needed.
"""
from __future__ import annotations

from typing import AsyncGenerator

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


class CosyVoiceTTSService(TTSService):
    def __init__(
        self,
        *,
        base_url: str,
        voice: str = "default",          # a preset speaker, or a cloned voice id
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._base_url = base_url.rstrip("/")
        self._voice = voice
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def stop(self, frame):  # close the session on pipeline shutdown
        await super().stop(frame)
        if self._session and not self._session.closed:
            await self._session.close()

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"CosyVoice TTS: {text!r}")
        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            session = await self._get_session()
            payload = {
                "text": text,
                "voice": self._voice,
                "sample_rate": self.sample_rate,
                "stream": True,
            }
            async with session.post(f"{self._base_url}/tts", json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    yield ErrorFrame(f"CosyVoice server {resp.status}: {body}")
                    return

                first = True
                # Server streams raw PCM; read fixed-size chunks (~20ms frames).
                chunk_bytes = int(self.sample_rate * 2 * 0.02)
                async for chunk in resp.content.iter_chunked(chunk_bytes):
                    if not chunk:
                        continue
                    if first:
                        await self.stop_ttfb_metrics()
                        first = False
                    yield TTSAudioRawFrame(chunk, self.sample_rate, 1)

            yield TTSStoppedFrame()
        except Exception as e:  # noqa: BLE001
            logger.exception("CosyVoice TTS failed")
            yield ErrorFrame(f"CosyVoice TTS error: {e}")
