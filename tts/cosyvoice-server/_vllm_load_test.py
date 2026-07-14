import os, time
os.environ["COSYVOICE_VLLM"] = "1"
import tts_engine  # sets up CosyVoice sys.path + reads COSYVOICE_VLLM

t0 = time.perf_counter()
eng = tts_engine.get_engine()
print("ENGINE_LOADED in %.1fs" % (time.perf_counter() - t0), flush=True)

# Warm once (first call compiles kernels), then measure TTFB on a real reply.
for _ in eng.synthesize_stream("Warming up."):
    pass
print("WARMED", flush=True)

t1 = time.perf_counter()
ttfb = None
n = 0
sr = 24000
for wav, sr in eng.synthesize_stream("Hi there! How can I help you today?"):
    if ttfb is None:
        ttfb = time.perf_counter() - t1
    n += wav.shape[-1]
total = time.perf_counter() - t1
print("RESULT TTFB=%.2fs total=%.2fs audio=%.2fs" % (ttfb, total, n / sr), flush=True)
print("VLLM_TEST_OK", flush=True)
