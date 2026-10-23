"""Microphone and speaker for the terminal, at one sample rate: 16-bit mono PCM, the format the realtime API takes in and sends back."""
import asyncio
import threading

import sounddevice as sd

RATE = 24000
FRAME = RATE // 50  # 20 ms


class Microphone:
    """Puts 20 ms chunks of PCM on an asyncio queue."""

    def __init__(self):
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def callback(indata, frames, time_info, status):
            loop.call_soon_threadsafe(self.queue.put_nowait, bytes(indata))

        self.stream = sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=FRAME, callback=callback)

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()


class Speaker:
    """Plays PCM as it arrives. clear() drops whatever hasn't played yet, for barge-in."""

    def __init__(self):
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.stream = sd.RawOutputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=FRAME, callback=self._fill)

    def _fill(self, outdata, frames, time_info, status):
        with self.lock:
            n = min(len(outdata), len(self.buffer))
            outdata[:n] = self.buffer[:n]
            del self.buffer[:n]
        outdata[n:] = b"\0" * (len(outdata) - n)

    def play(self, pcm: bytes):
        with self.lock:
            self.buffer.extend(pcm)

    def clear(self):
        with self.lock:
            self.buffer.clear()

    async def drain(self):
        while self.buffer:
            await asyncio.sleep(0.05)

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()
