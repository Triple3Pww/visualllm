"""Module 1 toy -- Sound is just numbers.

A from-scratch voice-activity detector. Two blanks are yours; the rest is given.

Run it:
    python learn/m1_vad.py output/q_ai.wav
    python learn/m1_vad.py output/q_ai.wav --drop-byte

The second command drops ONE byte from the front of the audio. Predict what happens
BEFORE you run it -- that is the whole point of the exercise.
"""
import argparse
import math
import struct
import sys
import wave

FRAME_MS = 20          # one VAD frame
SILENCE_RMS = 500      # below this (in int16 units) we call it silence
HANGOVER_FRAMES = 8    # keep speech "on" this many quiet frames before ending it


def read_pcm(path):
    """Return (raw_pcm_bytes, sample_rate, sample_width_bytes)."""
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            sys.exit("this toy wants mono int16 audio")
        # NOTE: we ASK the file for its rate. q_ai.wav is 24000, not 16000.
        # Assuming 16k here is the classic bug -- every timestamp would be 1.5x wrong.
        return w.readframes(w.getnframes()), w.getframerate(), w.getsampwidth()


def frame_size_bytes(rate, sample_width, frame_ms):
    """How many BYTES is one frame_ms frame of audio?"""
    # TODO(you) #1
    #   samples in the frame = rate * frame_ms / 1000
    #   each sample costs `sample_width` bytes
    #   return an int (number of BYTES)
    raise NotImplementedError("TODO(you) #1 -- see the hint above")


def rms(frame):
    """Root-mean-square loudness of one frame of int16 PCM."""
    n = len(frame) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack("<%dh" % n, frame[: n * 2])
    # TODO(you) #2
    #   square each sample, take the mean, take the square root, return a float.
    raise NotImplementedError("TODO(you) #2 -- see the hint above")


def find_speech(pcm, rate, sample_width):
    """Walk the audio frame by frame and return [(start_s, end_s), ...]."""
    fsize = frame_size_bytes(rate, sample_width, FRAME_MS)
    bytes_per_sec = rate * sample_width
    segments, quiet, start = [], 0, None

    for i in range(0, len(pcm) - fsize + 1, fsize):
        loud = rms(pcm[i:i + fsize]) >= SILENCE_RMS
        t = i / bytes_per_sec          # byte offset -> SECONDS. this is the whole trick.
        if loud:
            if start is None:
                start = t
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= HANGOVER_FRAMES:
                segments.append((start, t))
                start, quiet = None, 0
    if start is not None:
        segments.append((start, len(pcm) / bytes_per_sec))
    return segments


def mean_rms(pcm, rate, sample_width):
    fsize = frame_size_bytes(rate, sample_width, FRAME_MS)
    frames = [rms(pcm[i:i + fsize]) for i in range(0, len(pcm) - fsize + 1, fsize)]
    return sum(frames) / len(frames) if frames else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--drop-byte", action="store_true",
                    help="drop ONE byte from the front. predict the result first.")
    args = ap.parse_args()

    pcm, rate, sw = read_pcm(args.wav)
    print("file        : %s" % args.wav)
    print("sample rate : %d Hz   (did you assume 16000?)" % rate)
    print("sample width: %d bytes/sample" % sw)
    print("audio bytes : %d" % len(pcm))
    print("duration    : %.2f s   = bytes / (rate * width)" % (len(pcm) / (rate * sw)))

    if args.drop_byte:
        pcm = pcm[1:]
        print("\n!! dropped 1 byte. every int16 is now assembled from the WRONG two bytes.")

    print("\nmean RMS    : %.0f" % mean_rms(pcm, rate, sw))
    segs = find_speech(pcm, rate, sw)
    print("speech segments: %d" % len(segs))
    for s, e in segs:
        print("  %.2fs -> %.2fs  (%.2fs)" % (s, e, e - s))

    print("\n--- what you just proved ---")
    print("Audio is a list of int16 numbers. Each one is EXACTLY two bytes.")
    print("Shift by one byte and every sample becomes garbage -- yet it is still")
    print("a perfectly valid WAV, and the loudness stays plausible. That is P40:")
    print("the avatar's lip-sync model was fed exactly this noise for three sessions,")
    print("while the voice a human heard stayed perfect. Read P40 only after this runs.")


if __name__ == "__main__":
    main()
