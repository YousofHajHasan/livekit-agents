"""LiveKit Agents — Gender Detection Plugin

Detects speaker gender (female / male / child) in parallel with the live
audio stream using a wav2vec2-based ONNX model, with zero added latency to
the LLM pipeline.
"""

from .detector import GenderDetector, GenderLabel
from .version import __version__

__all__ = ["GenderDetector", "GenderLabel", "__version__"]

from livekit.agents import Plugin

from .log import logger


class GenderDetectionPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(GenderDetectionPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
