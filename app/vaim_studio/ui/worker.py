"""Background thread that owns the GPU: model loading and rendering."""
import sys
import traceback

from PySide6.QtCore import QObject, Signal, Slot


def _keep_awake(on):
    """Stop Windows from sleeping mid-render (the display may still turn off)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


class EngineWorker(QObject):
    log = Signal(str)
    ready = Signal(str, float)              # gpu name, vram GB
    download = Signal(str, float, float)    # file, done bytes, total bytes (float: >2 GB safe)
    preview = Signal(object)                # engine Preview
    finished = Signal(object, object)       # final Preview, job dict
    failed = Signal(str, object)            # message, job
    stage = Signal(str, float, str)         # "prepare" | "paint" | "upscale", fraction (-1 = unknown), detail

    def __init__(self):
        super().__init__()
        self._stop = False
        self.dreamer = None

    def _log(self, text):
        print(text)
        self.log.emit(text)

    def _fail(self, text, job=None):
        print("ERROR:", text, file=sys.stderr)
        self.failed.emit(text, job)

    def stop(self):
        self._stop = True

    def _stopped(self):
        return self._stop

    @Slot()
    def init_engine(self):
        try:
            self._log("Starting PyTorch ...")
            from ..engine import Dreamer
            self.dreamer = Dreamer(log=self._log)
            self.dreamer.stage = lambda key, frac=-1.0, detail="": self.stage.emit(key, float(frac), detail)
            self.ready.emit(self.dreamer.gpu_name(), self.dreamer.vram_gb())
        except Exception as e:
            self._fail(f"Could not start PyTorch:\n{e}\n\n{traceback.format_exc()}")

    @Slot(object)
    def dream(self, job):
        self._stop = False
        s = job["settings"]
        progress = lambda f, d, t: self.download.emit(f, float(d), float(t))  # noqa: E731
        _keep_awake(True)
        self.stage.emit("prepare", -1.0, "Getting the models ready")
        try:
            self.dreamer.ensure_models(s.image_size, s.clip_model, progress=progress, cancelled=self._stopped)
            if s.output_size and s.output_size > s.image_size and s.upscale_strength > 0:
                self.dreamer.ensure_upscaler(progress=progress, cancelled=self._stopped)
            final = self.dreamer.run(s, on_preview=self.preview.emit, should_stop=self._stopped,
                                     progress=progress)
            self.finished.emit(final, job)
        except Exception as e:
            if type(e).__name__ == "Cancelled":
                self._fail("Download cancelled.", job)
            else:
                self._fail(f"{e}\n\n{traceback.format_exc()}", job)
        finally:
            _keep_awake(False)
