"""
GenderDetector — parallel, zero-latency gender inference for LiveKit Agents.

Design
------
Frames are pushed in real-time from the VAD pipeline (on_vad_inference_done).
Once the buffer reaches TRIGGER_SECONDS (2 s) OR speech ends (whichever comes
first), inference runs in a thread-pool executor so it never blocks the event
loop.  The result is stored on the instance and read synchronously inside
on_end_of_turn — by that point the ONNX call (~30–60 ms) has long finished.

Labels: "female" | "male" | "child"  (mirrors the model's class order)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Literal

import numpy as np

from .log import logger

# How many seconds of audio to accumulate before firing the first inference.
# 2 s gives reliable gender classification; shorter clips work but are noisier.
TRIGGER_SECONDS = 2.0

# Model expects 16 kHz mono float32
MODEL_SAMPLE_RATE = 16000

GenderLabel = Literal["female", "male", "child"]
_LABELS: list[GenderLabel] = ["female", "male", "child"]

# Minimum confidence margin (difference between top-1 and top-2 softmax prob)
# below which we treat the result as uncertain and keep None.
MIN_CONFIDENCE_MARGIN = 0.15


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class GenderDetector:
    """
    Accumulates raw PCM frames from the VAD pipeline and runs ONNX gender
    inference in a background thread.  Designed to be reused across turns.

    Typical lifecycle per user turn
    --------------------------------
    1. ``begin_turn()``             called from on_start_of_speech
    2. ``push_frames(frames)``      called from on_vad_inference_done  (many times)
    3. ``finalize()``               called from on_end_of_speech
    4. ``result`` property          read from on_end_of_turn
    """

    def __init__(self, model_root: str, *, executor: concurrent.futures.ThreadPoolExecutor | None = None) -> None:
        """
        Parameters
        ----------
        model_root:
            Directory containing the audonnx model (model.onnx + model.yaml).
            Pass the same path used in your notebook.
        executor:
            Optional shared ThreadPoolExecutor.  If None a dedicated single-
            thread executor is created (ONNX is not thread-safe by default).
        """
        import audonnx  # imported lazily so the plugin is importable without audonnx

        self._model = audonnx.load(model_root)
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gender_det"
        )

        # per-turn state
        self._buffer: list[np.ndarray] = []
        self._buffered_samples: int = 0
        self._result: GenderLabel | None = None
        self._confidence: float = 0.0
        self._inference_task: asyncio.Task[None] | None = None
        self._triggered: bool = False  # True once the 2-s threshold task has been launched

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, model_root: str, **kwargs: object) -> "GenderDetector":
        """Convenience constructor.  Blocks while loading the ONNX model."""
        return cls(model_root, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def load_from_hf(
        cls,
        repo_id: str = "Yousof10/GenderDetection",
        *,
        cache_dir: str | None = None,
        **kwargs: object,
    ) -> "GenderDetector":
        """Download model files from HuggingFace Hub and load the detector.

        The files are cached locally by ``huggingface_hub`` so subsequent
        starts are instant (no re-download unless the revision changes).

        Parameters
        ----------
        repo_id:
            HuggingFace repo in ``owner/name`` format.
            Defaults to ``"Yousof10/GenderDetection"``.
        cache_dir:
            Optional override for the HF cache directory.
            Defaults to the standard HF_HOME cache (``~/.cache/huggingface``).
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to use load_from_hf(). "
                "Install it with: pip install huggingface-hub"
            ) from exc

        model_root = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            ignore_patterns=["*.gitattributes", ".gitattributes"],
        )
        return cls(model_root, **kwargs)  # type: ignore[arg-type]

    def begin_turn(self) -> None:
        """Reset all per-turn state.  Call at on_start_of_speech."""
        if self._inference_task is not None and not self._inference_task.done():
            self._inference_task.cancel()
        self._buffer = []
        self._buffered_samples = 0
        self._result = None
        self._confidence = 0.0
        self._inference_task = None
        self._triggered = False

    def push_frames(self, frames: list) -> None:
        """
        Append raw audio frames (livekit.rtc.AudioFrame) to the buffer.
        Automatically resamples if the frame's sample_rate != 16000.

        Once the buffer crosses TRIGGER_SECONDS, launches background inference
        *once*.  Subsequent frames keep accumulating for finalize().
        """
        for frame in frames:
            pcm = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0

            # handle multi-channel: take mean across channels
            if frame.num_channels > 1:
                pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1)

            # resample if needed
            if frame.sample_rate != MODEL_SAMPLE_RATE:
                pcm = _resample(pcm, frame.sample_rate, MODEL_SAMPLE_RATE)

            self._buffer.append(pcm)
            self._buffered_samples += len(pcm)

        # fire early inference once we have enough audio
        if not self._triggered and self._buffered_samples >= int(TRIGGER_SECONDS * MODEL_SAMPLE_RATE):
            self._triggered = True
            self._launch_inference(snapshot=list(self._buffer))

    def finalize(self) -> None:
        """
        Called from on_end_of_speech.  If the 2-s threshold was never reached
        (very short utterance) we fire inference now with whatever we have.
        If inference is already running/done, we re-run with the full buffer
        to get a more accurate result.
        """
        if not self._buffer:
            return  # nothing was spoken

        # Always run a final inference on the full buffer
        self._launch_inference(snapshot=list(self._buffer))

    @property
    def result(self) -> GenderLabel | None:
        """
        Latest gender prediction, or None if inference hasn't completed yet
        (very unlikely by on_end_of_turn time).
        """
        return self._result

    @property
    def confidence(self) -> float:
        """Softmax confidence of the predicted class (0–1)."""
        return self._confidence

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _launch_inference(self, snapshot: list[np.ndarray]) -> None:
        """Schedule inference in the thread-pool, cancel any previous task."""
        if self._inference_task is not None and not self._inference_task.done():
            self._inference_task.cancel()

        loop = asyncio.get_event_loop()
        self._inference_task = loop.create_task(
            self._run_inference(snapshot), name="gender_det_inference"
        )

    async def _run_inference(self, buffer: list[np.ndarray]) -> None:
        signal = np.concatenate(buffer, axis=0)
        loop = asyncio.get_event_loop()
        try:
            label, confidence = await loop.run_in_executor(
                self._executor, self._infer, signal
            )
            self._result = label
            self._confidence = confidence
            logger.debug(
                "gender inference done | result=%s confidence=%.3f duration=%.2fs",
                label,
                confidence,
                len(signal) / MODEL_SAMPLE_RATE,
            )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("gender inference failed")

    def _infer(self, signal: np.ndarray) -> tuple[GenderLabel, float]:
        """Blocking ONNX call — runs in executor thread."""
        output = self._model(signal, MODEL_SAMPLE_RATE)
        logits = output["logits_gender"][0]
        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        label = _LABELS[idx]

        # confidence margin: difference between top-1 and runner-up
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1])

        if margin < MIN_CONFIDENCE_MARGIN:
            # too close to call — treat as uncertain (keep previous result if any)
            raise ValueError(
                f"gender uncertain (margin={margin:.3f} < {MIN_CONFIDENCE_MARGIN})"
            )

        return label, float(probs[idx])


def _resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resample (no extra dependency required).
    For production quality, install resampy and replace with resampy.resample."""
    try:
        import resampy  # type: ignore
        return resampy.resample(signal, orig_sr, target_sr)
    except ImportError:
        pass

    # Fallback: numpy linear interpolation (acceptable for gender — low-freq features)
    duration = len(signal) / orig_sr
    target_len = int(duration * target_sr)
    old_idx = np.linspace(0, len(signal) - 1, target_len)
    return np.interp(old_idx, np.arange(len(signal)), signal).astype(np.float32)
