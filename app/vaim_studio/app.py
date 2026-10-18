import faulthandler
import os
import sys
import time
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"


class _Tee:
    """Write to the log file and (if there is one) the console."""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, text):
        for s in self.streams:
            try:
                s.write(text)
                s.flush()
            except Exception:
                pass
        return len(text)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def _setup_logging():
    from .engine.weights import app_home
    home = app_home()
    home.mkdir(parents=True, exist_ok=True)
    log_path = home / "log.txt"
    if log_path.exists():  # keep the previous session's log (e.g. the one that crashed)
        try:
            os.replace(log_path, home / "log-previous.txt")
        except OSError:
            pass
    log = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(log, sys.stdout)
    sys.stderr = _Tee(log, sys.stderr)
    # native crashes (access violations, heap corruption) still get a Python traceback in the log
    faulthandler.enable(file=log, all_threads=True)
    print(f"VAIM Studio log - {time.strftime('%Y-%m-%d %H:%M:%S')} - Python {sys.version.split()[0]}")
    return log_path


def main():
    # Windows: own taskbar identity (so our icon shows instead of python's)
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("VAIMStudio.App")
        except Exception:
            pass
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
    _setup_logging()

    # IMPORTANT: load PyTorch (and its C++ runtime DLLs) on the main thread BEFORE Qt.
    # PySide6 bundles its own msvcp140.dll; if Qt loads first, PyTorch is forced to share
    # that copy, and a version mismatch corrupts the heap (Windows error 0xc0000374).
    try:
        import torch
        print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    except Exception as e:  # reported in the UI by the engine thread
        print(f"PyTorch failed to load: {e!r}")

    def excepthook(t, v, tb):
        import traceback
        text = "".join(traceback.format_exception(t, v, tb))
        print(text, file=sys.stderr)
        try:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "VAIM Studio", f"Unexpected error:\n\n{v}\n\n{text[-1500:]}")
        except Exception:
            pass
    sys.excepthook = excepthook

    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from .ui import theme
    from .ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("VAIM Studio")
    app.setOrganizationName("VAIMStudio")
    app.setStyle("Fusion")
    app.setStyleSheet(theme.QSS)
    icon = QIcon(str(ASSETS / "icon.ico"))
    app.setWindowIcon(icon)
    w = MainWindow(icon)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
