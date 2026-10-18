import random
import time
from dataclasses import replace
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, QSettings, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QStackedWidget, QToolButton, QVBoxLayout, QWidget,
)

from .. import __version__, storage
from .worker import EngineWorker

BASES = [(128, "128 · lo-fi"), (256, "256 · soft"), (512, "512 · full")]
SIZES = [("1080", 1080), ("2K", 2048), ("4K", 4096), ("512", 0), ("Custom", -1)]
QUALITY = [("Draft", 32), ("Balanced", 64), ("Best", 128)]
CLIPS = ["ViT-B/32", "ViT-B/16", "ViT-L/14"]
CUSTOM = len(SIZES) - 1

# name -> settings. "size" (optional) is an index into SIZES.
PRESETS = {
    "Sharpest": dict(sub="Max detail · slowest", base=512, cutouts=128, steps=500, maxc=0, center=False,
                     interp="bilinear", sharpen=85, size=1,
                     tip="128 cutouts, 500 steps, smooth crops, 2K output with strong but natural upscaling."),
    "Balanced": dict(sub="Good everyday default", base=512, cutouts=64, steps=350, maxc=0, center=False,
                     interp="bilinear", sharpen=75,
                     tip="64 cutouts and smooth crops: clean results at about half the time of Sharpest."),
    "Portrait": dict(sub="Faces & figures", base=512, cutouts=64, steps=400, maxc=15, center=True,
                     interp="bilinear", sharpen=60,
                     tip="Centre-focused, fewer blended classes. Add a face photo as image prompt for real faces."),
    "Landscape": dict(sub="Scenes & places", base=512, cutouts=64, steps=300, maxc=0, center=False,
                      interp="nearest", sharpen=80,
                      tip="Whole-frame composition with a slightly softer upscale."),
    "Classic 2021": dict(sub="Original Big Sleep look", base=512, cutouts=128, steps=400, maxc=0, center=False,
                         interp="nearest", sharpen=100,
                         tip="The original settings: 128 cutouts, all classes, crunchy nearest-neighbour crops."),
    "Fast draft": dict(sub="Quick preview", base=512, cutouts=32, steps=150, maxc=0, center=False,
                       interp="nearest", sharpen=100,
                       tip="32 cutouts, 150 steps: a quick look (about 3× faster) before a full render."),
}
EXAMPLES = [
    "a lighthouse in a storm at night",
    "a cathedral made of glass",
    "an old sailor, oil painting",
    "a lantern in a foggy forest",
    "a burning city skyline at dusk",
    "a jellyfish drifting through space",
]


def pil_to_qimage(im: Image.Image) -> QImage:
    im = im.convert("RGB")
    return QImage(im.tobytes("raw", "RGB"), im.width, im.height, im.width * 3, QImage.Format_RGB888).copy()


def fmt_time(sec):
    sec = max(0, int(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def fmt_minutes(sec):
    m = sec / 60
    return "under a minute" if m < 1 else f"~{m:.0f} min" if m < 90 else f"~{m / 60:.1f} h"


def upscale_passes(native, out):
    passes, w = 0, native
    while True:
        passes += 1
        w *= 2
        if w * 1.5 >= out:
            return passes


# --------------------------------------------------------------------------- widgets

class ImageView(QWidget):
    """Fit-to-window or 100% view with drag-to-pan. Enlarging uses whole-number zoom with
    nearest-neighbour pixels so small BigGAN canvases stay crisp."""
    fit_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.img = None
        self.fit = True
        self.offset = QPointF(0, 0)
        self._drag = None
        self.setMinimumSize(256, 256)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, im, keep_view=True):
        self.img = pil_to_qimage(im) if im is not None else None
        if not keep_view:
            self.offset = QPointF(0, 0)
        self.update()

    def set_fit(self, fit):
        self.fit = fit
        self.offset = QPointF(0, 0)
        self.update()
        self.fit_changed.emit(fit)

    def _geometry(self):
        w, h = self.width(), self.height()
        iw, ih = self.img.width(), self.img.height()
        if self.fit:
            s = min(w / iw, h / ih)
            if s >= 1:
                s = float(int(s))
        else:
            s = 1.0
        tw, th = iw * s, ih * s
        mx, my = max(0.0, (tw - w) / 2), max(0.0, (th - h) / 2)
        self.offset = QPointF(min(mx, max(-mx, self.offset.x())), min(my, max(-my, self.offset.y())))
        return QRectF((w - tw) / 2 + self.offset.x(), (h - th) / 2 + self.offset.y(), tw, th), s

    def paintEvent(self, e):
        if self.img is None:
            return
        p = QPainter(self)
        r, s = self._geometry()
        p.setRenderHint(QPainter.SmoothPixmapTransform, s < 1)
        p.drawImage(r, self.img)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.img is not None and not self.fit:
            self._drag = (e.position(), QPointF(self.offset))
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._drag:
            start, off = self._drag
            self.offset = off + (e.position() - start)
            self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None
        self.unsetCursor()

    def mouseDoubleClickEvent(self, e):
        if self.img is not None:
            self.set_fit(not self.fit)


class Segmented(QWidget):
    changed = Signal(int)

    def __init__(self, labels, name="seg"):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.group = QButtonGroup(self)
        for i, t in enumerate(labels):
            b = QPushButton(t)
            b.setObjectName(name)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            lay.addWidget(b)
            self.group.addButton(b, i)
        self.group.idClicked.connect(self.changed.emit)

    def index(self):
        return max(0, self.group.checkedId())

    def set_index(self, i):
        b = self.group.button(i)
        if b:
            b.setChecked(True)

    def set_all_enabled(self, on):
        for b in self.group.buttons():
            b.setEnabled(on)


def label(text, name="hint", wrap=False):
    lab = QLabel(text)
    lab.setObjectName(name)
    lab.setWordWrap(wrap)
    return lab


def restyle(w, name):
    if w.objectName() != name:
        w.setObjectName(name)
        w.style().unpolish(w)
        w.style().polish(w)


# --------------------------------------------------------------------------- window

class MainWindow(QMainWindow):
    start_dream = Signal(object)

    def __init__(self, icon: QIcon = None, start_engine=True, settings: QSettings = None):
        super().__init__()
        self.setWindowTitle("VAIM Studio")
        if icon:
            self.setWindowIcon(icon)
        self.resize(1440, 920)
        self.setMinimumSize(1120, 740)
        self.qs = settings or QSettings("VAIMStudio", "VAIMStudio")
        self.out_dir = Path(self.qs.value("out_dir", str(storage.default_output_dir())))
        self.running = False
        self.engine_ready = False
        self.vram = 0.0
        self.gpu_label = ""
        self.init_from = None   # (path, latents dict, native size, meta)
        self.last = None        # last final Preview
        self.last_job = None
        self.last_paths = []
        self.current = []
        self.focus = 0
        self.run_started = 0.0
        self.queue = []
        self.queue_paused = False
        self.job = None
        self._stopping = False
        self._applying = False
        self.active_preset = None

        self._build()
        self._load_settings()
        self.refresh_gallery()
        self.refresh_queue()

        self.worker = None
        if start_engine:
            self.engine_thread = QThread(self)
            self.worker = EngineWorker()
            self.worker.moveToThread(self.engine_thread)
            self.engine_thread.started.connect(self.worker.init_engine)
            self.worker.log.connect(self.on_log)
            self.worker.ready.connect(self.on_ready)
            self.worker.download.connect(self.on_download)
            self.worker.preview.connect(self.on_preview)
            self.worker.finished.connect(self.on_finished)
            self.worker.failed.connect(self.on_failed)
            self.start_dream.connect(self.worker.dream)
            self.engine_thread.start()

    # ================================================================== layout
    def _build(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_sidebar())
        outer.addWidget(self._build_main(), 1)
        self._build_statusbar()

        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.dream_or_stop)
        QShortcut(QKeySequence("Ctrl+Shift+Return"), self, activated=self.add_to_queue)
        QShortcut(QKeySequence("Escape"), self, activated=self.stop)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=lambda: self.view.set_fit(not self.view.fit))

    def _heading(self, lay, title, extra=None):
        row = QHBoxLayout()
        row.setContentsMargins(0, 14, 0, 2)
        row.addWidget(label(title, "heading"))
        row.addStretch(1)
        if extra is not None:
            row.addWidget(extra)
        lay.addLayout(row)

    def _build_sidebar(self):
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(384)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setObjectName("sidebar")
        f = QVBoxLayout(inner)
        f.setContentsMargins(22, 18, 22, 16)
        f.setSpacing(8)
        scroll.setWidget(inner)
        sv.addWidget(scroll, 1)

        head = QHBoxLayout()
        wm = QVBoxLayout()
        wm.setSpacing(0)
        wm.addWidget(label("vaim studio", "wordmark"))
        wm.addWidget(label("vintage ai model studio", "tagline"))
        head.addLayout(wm, 1)
        about = QToolButton()
        about.setObjectName("ghost")
        about.setText("?")
        about.setToolTip("About VAIM Studio")
        about.clicked.connect(self.show_about)
        head.addWidget(about, 0, Qt.AlignTop)
        f.addLayout(head)

        # ---- prompt card
        self._heading(f, "Describe your image")
        self.prompt = QPlainTextEdit()
        self.prompt.setObjectName("prompt")
        self.prompt.setPlaceholderText("a lantern dangling from a tree in a foggy graveyard")
        self.prompt.setFixedHeight(96)
        f.addWidget(self.prompt)
        links = QHBoxLayout()
        links.setSpacing(2)
        self.avoid_link = QPushButton("+ Avoid words")
        self.avoid_link.setObjectName("link")
        self.avoid_link.setCursor(Qt.PointingHandCursor)
        self.avoid_link.clicked.connect(lambda: self._toggle_avoid(True))
        self.img_btn = QPushButton("+ Image prompt")
        self.img_btn.setObjectName("link")
        self.img_btn.setCursor(Qt.PointingHandCursor)
        self.img_btn.setToolTip("Steer toward a picture (alone, or blended with the text)")
        self.img_btn.clicked.connect(self.pick_image_prompt)
        self.img_clear = QToolButton()
        self.img_clear.setObjectName("ghost")
        self.img_clear.setText("✕")
        self.img_clear.setToolTip("Remove the image prompt")
        self.img_clear.clicked.connect(lambda: self.set_image_prompt(None))
        links.addWidget(self.avoid_link)
        links.addWidget(self.img_btn)
        links.addWidget(self.img_clear)
        links.addStretch(1)
        links.addWidget(label("use  |  to blend ideas"))
        f.addLayout(links)
        self.negative = QLineEdit()
        self.negative.setPlaceholderText("Avoid:  text | people | frames")
        f.addWidget(self.negative)
        self.image_prompt = None
        self.img_clear.hide()
        self.negative.hide()
        self.init_chip = label("", "chip", wrap=True)
        self.init_chip_clear = QPushButton("Start fresh instead")
        self.init_chip_clear.setObjectName("link")
        self.init_chip_clear.clicked.connect(self.clear_init)
        f.addWidget(self.init_chip)
        f.addWidget(self.init_chip_clear)
        self.init_chip.hide()
        self.init_chip_clear.hide()

        # ---- style cards
        self._heading(f, "Style")
        grid = QGridLayout()
        grid.setSpacing(6)
        self.preset_group = QButtonGroup(self)
        self.preset_buttons = {}
        for i, (name, p) in enumerate(PRESETS.items()):
            b = QPushButton(f"{name}\n{p['sub']}")
            b.setObjectName("preset")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(p["tip"])
            b.setMinimumHeight(52)
            b.clicked.connect(lambda _=False, n=name: self.apply_preset(n))
            self.preset_group.addButton(b, i)
            self.preset_buttons[name] = b
            grid.addWidget(b, i // 2, i % 2)
        f.addLayout(grid)
        self.preset_hint = label("", wrap=True)
        f.addWidget(self.preset_hint)

        # ---- size
        self._heading(f, "Size")
        srow = QHBoxLayout()
        srow.setSpacing(6)
        self.size_seg = Segmented([t for t, _ in SIZES], "pill")
        self.size_seg.changed.connect(lambda _: self.update_hints())
        self.custom_size = QSpinBox()
        self.custom_size.setRange(256, 4096)
        self.custom_size.setSingleStep(64)
        self.custom_size.setSuffix(" px")
        self.custom_size.setFixedWidth(92)
        self.custom_size.valueChanged.connect(lambda _: self.update_hints())
        srow.addWidget(self.size_seg, 1)
        srow.addWidget(self.custom_size)
        f.addLayout(srow)
        self.size_hint = label("", wrap=True)
        f.addWidget(self.size_hint)
        self.sharpen_row = QWidget()
        sh = QGridLayout(self.sharpen_row)
        sh.setContentsMargins(0, 2, 0, 0)
        sh.setVerticalSpacing(2)
        sh.addWidget(label("Upscale sharpness", "field"), 0, 0)
        self.sharpen_val = label("", "value")
        sh.addWidget(self.sharpen_val, 0, 1, Qt.AlignRight)
        self.sharpen = QSlider(Qt.Horizontal)
        self.sharpen.setRange(0, 100)
        self.sharpen.setToolTip("100% = full Real-ESRGAN detail. 60-85% looks most natural. "
                                "Lower keeps more of the soft, vintage GAN texture.")
        self.sharpen.valueChanged.connect(lambda v: self.sharpen_val.setText(f"{v}%"))
        sh.addWidget(self.sharpen, 1, 0, 1, 2)
        f.addWidget(self.sharpen_row)

        # ---- quality
        self._heading(f, "Quality")
        self.quality_seg = Segmented([q for q, _ in QUALITY], "pill")
        self.quality_seg.changed.connect(lambda i: self.cutouts.setValue(QUALITY[i][1]))
        f.addWidget(self.quality_seg)
        self.quality_hint = label("", wrap=True)
        f.addWidget(self.quality_hint)

        # ---- save
        self._heading(f, "Save to")
        frow = QHBoxLayout()
        frow.setSpacing(6)
        self.folder_btn = QPushButton()
        self.folder_btn.setObjectName("folder")
        self.folder_btn.setCursor(Qt.PointingHandCursor)
        self.folder_btn.clicked.connect(self.pick_out_dir)
        self.fmt = QComboBox()
        self.fmt.addItems(list(storage.FORMATS))
        self.fmt.setToolTip("PNG = lossless. JPEG / WebP = much smaller files at 95% quality.")
        self.fmt.setFixedWidth(82)
        open_dir = QToolButton()
        open_dir.setObjectName("ghost")
        open_dir.setText("↗")
        open_dir.setToolTip("Open the render folder")
        open_dir.clicked.connect(lambda: storage.open_in_explorer(self.out_dir))
        frow.addWidget(self.folder_btn, 1)
        frow.addWidget(self.fmt)
        frow.addWidget(open_dir)
        f.addLayout(frow)

        # ---- fine-tune drawer
        self.adv_btn = QToolButton()
        self.adv_btn.setObjectName("disclosure")
        self.adv_btn.setText("▸  Fine-tune")
        self.adv_btn.setCheckable(True)
        self.adv_btn.setCursor(Qt.PointingHandCursor)
        self.adv_btn.toggled.connect(self.toggle_advanced)
        f.addSpacing(10)
        f.addWidget(self.adv_btn)
        self.adv = QFrame()
        self.adv.setObjectName("drawer")
        a = QGridLayout(self.adv)
        a.setContentsMargins(12, 12, 12, 12)
        a.setHorizontalSpacing(10)
        a.setVerticalSpacing(7)
        r = 0

        def add(text, w, tip=None):
            nonlocal r
            lab = label(text)
            if tip:
                lab.setToolTip(tip)
                w.setToolTip(tip)
            a.addWidget(lab, r, 0)
            a.addWidget(w, r, 1)
            r += 1

        def sub(text):
            nonlocal r
            a.addWidget(label(text, "subheading"), r, 0, 1, 2)
            r += 1

        sub("Run")
        steps_box = QWidget()
        sb = QHBoxLayout(steps_box)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(8)
        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setRange(1, 40)  # x25
        self.steps_slider.valueChanged.connect(lambda v: self.steps_spin.setValue(v * 25))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 5000)
        self.steps_spin.setSingleStep(50)
        self.steps_spin.setFixedWidth(70)
        self.steps_spin.valueChanged.connect(self._steps_changed)
        sb.addWidget(self.steps_slider, 1)
        sb.addWidget(self.steps_spin)
        add("Steps", steps_box, "Optimisation steps. 300-500 is plenty; auto-finish usually stops earlier.")
        self.auto_stop = QCheckBox("Finish early once the image settles")
        self.auto_stop.setToolTip("Stops when the picture has barely changed for 30 steps (never before step 150).")
        a.addWidget(self.auto_stop, r, 0, 1, 2); r += 1
        self.base_seg = Segmented([t for _, t in BASES], "pill")
        self.base_seg.changed.connect(lambda _: self.update_hints())
        add("Base model", self.base_seg, "BigGAN canvas size. 512 is recommended: CLIP does most of the work, "
                                         "so smaller bases are barely faster but much blurrier.")
        self.batch = QSpinBox()
        self.batch.setRange(1, 8)
        add("Variations", self.batch, "Several images optimised together in one batched pass")
        seed_box = QWidget()
        sr = QHBoxLayout(seed_box)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.setSpacing(4)
        self.seed = QLineEdit()
        self.seed.setPlaceholderText("random")
        dice = QToolButton()
        dice.setObjectName("ghost")
        dice.setText("↺")
        dice.setToolTip("Reuse the last seed")
        dice.clicked.connect(self.reuse_seed)
        sr.addWidget(self.seed, 1)
        sr.addWidget(dice)
        add("Seed", seed_box, "Same seed + same settings = same image")

        sub("Look")
        self.cutouts = QSpinBox(); self.cutouts.setRange(4, 256)
        self.cutouts.valueChanged.connect(lambda _: self.update_hints())
        add("Cutouts", self.cutouts, "Random crops CLIP looks at each step. More = more coherent, slower.")
        self.max_classes = QSpinBox(); self.max_classes.setRange(0, 1000); self.max_classes.setSpecialValueText("all")
        add("Max classes", self.max_classes, "How many ImageNet classes BigGAN may blend (e.g. 15 = more coherent shapes).")
        self.class_temp = QDoubleSpinBox(); self.class_temp.setRange(0.1, 10); self.class_temp.setSingleStep(0.5)
        add("Class temperature", self.class_temp, "Only used with Max classes.")
        self.interp = QComboBox(); self.interp.addItems(["bilinear", "nearest"])
        add("Crop sampling", self.interp, "bilinear = smoother, cleaner surfaces · nearest = the original crunchy look")
        self.center_bias = QCheckBox("Focus crops on the centre")
        a.addWidget(self.center_bias, r, 0, 1, 2); r += 1
        self.lr = QDoubleSpinBox(); self.lr.setRange(0.001, 0.5); self.lr.setDecimals(3); self.lr.setSingleStep(0.01)
        add("Learning rate", self.lr, "How fast the latents move. Original: 0.07")
        self.ema = QDoubleSpinBox(); self.ema.setRange(0.5, 0.999); self.ema.setDecimals(3); self.ema.setSingleStep(0.005)
        add("Smoothing (EMA)", self.ema, "Smoothing of the displayed image. Original: 0.99")

        sub("Performance")
        self.clip_model = QComboBox(); self.clip_model.addItems(CLIPS)
        add("CLIP model", self.clip_model, "ViT-B/32 is the original and by far the fastest. Larger models need much more VRAM.")
        self.precision = QComboBox(); self.precision.addItems(["auto", "bf16", "fp16", "fp32"])
        add("Precision", self.precision, "auto = the tuner measures what is fastest and safe on your GPU.")
        self.ckpt = QComboBox(); self.ckpt.addItems(["auto", "on", "off"])
        add("Checkpointing", self.ckpt, "Trades speed for less VRAM. auto = the tuner decides.")
        self.chunk = QSpinBox(); self.chunk.setRange(0, 256); self.chunk.setSpecialValueText("auto")
        add("Crops per pass", self.chunk, "How many cutouts go through CLIP at once. auto = the tuner decides.")
        self.preview_every = QSpinBox(); self.preview_every.setRange(1, 200)
        add("Preview every", self.preview_every, "Steps between preview refreshes.")
        self.compile = QCheckBox("torch.compile (experimental)")
        self.compile.setToolTip("Needs the triton-windows package; falls back automatically.")
        a.addWidget(self.compile, r, 0, 1, 2); r += 1
        retune = QPushButton("Re-tune for my GPU")
        retune.setToolTip("Forget the measured speed settings (e.g. after a driver update)")
        retune.clicked.connect(self.reset_tuning)
        a.addWidget(retune, r, 0, 1, 2); r += 1
        a.setColumnStretch(1, 1)
        f.addWidget(self.adv)
        self.adv.hide()
        f.addStretch(1)

        # anything changed by hand after picking a style makes it "custom"
        for sig in (self.quality_seg.changed, self.steps_spin.valueChanged, self.cutouts.valueChanged,
                    self.max_classes.valueChanged, self.class_temp.valueChanged, self.interp.currentIndexChanged,
                    self.center_bias.toggled, self.sharpen.valueChanged, self.base_seg.changed, self.lr.valueChanged):
            sig.connect(self._mark_custom)

        # ---- dock: estimate, actions, queue
        bottom = QWidget()
        bottom.setObjectName("dock")
        b = QVBoxLayout(bottom)
        b.setContentsMargins(22, 12, 22, 16)
        b.setSpacing(8)
        self.estimate = label("", "estimate")
        b.addWidget(self.estimate)
        arow = QHBoxLayout()
        arow.setSpacing(8)
        self.dream_btn = QPushButton("Dream")
        self.dream_btn.setObjectName("dream")
        self.dream_btn.setCursor(Qt.PointingHandCursor)
        self.dream_btn.setToolTip("Render now  (Ctrl+Enter)")
        self.dream_btn.setEnabled(False)
        self.dream_btn.clicked.connect(self.dream_or_stop)
        self.queue_btn = QPushButton("+ Queue")
        self.queue_btn.setObjectName("secondary")
        self.queue_btn.setCursor(Qt.PointingHandCursor)
        self.queue_btn.setToolTip("Add these settings to the render queue  (Ctrl+Shift+Enter)")
        self.queue_btn.clicked.connect(self.add_to_queue)
        arow.addWidget(self.dream_btn, 1)
        arow.addWidget(self.queue_btn)
        b.addLayout(arow)
        self.queue_box = QWidget()
        q = QVBoxLayout(self.queue_box)
        q.setContentsMargins(0, 2, 0, 0)
        q.setSpacing(4)
        qh = QHBoxLayout()
        self.queue_title = label("Queue", "field")
        qh.addWidget(self.queue_title, 1)
        self.queue_start = QPushButton("Start")
        self.queue_start.setObjectName("link")
        self.queue_start.clicked.connect(self.resume_queue)
        qclear = QPushButton("Clear")
        qclear.setObjectName("link")
        qclear.clicked.connect(self.clear_queue)
        qh.addWidget(self.queue_start)
        qh.addWidget(qclear)
        q.addLayout(qh)
        self.queue_list = QListWidget()
        self.queue_list.setObjectName("queue")
        self.queue_list.setMaximumHeight(84)
        self.queue_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.queue_list.setTextElideMode(Qt.ElideRight)
        self.queue_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.queue_list.customContextMenuRequested.connect(self.queue_menu)
        q.addWidget(self.queue_list)
        b.addWidget(self.queue_box)
        sv.addWidget(bottom)
        return side

    def _build_welcome(self):
        page = QWidget()
        page.setObjectName("welcome")
        v = QVBoxLayout(page)
        v.setContentsMargins(40, 40, 40, 40)
        v.addStretch(2)
        v.addWidget(label("What should we dream?", "hero"), 0, Qt.AlignHCenter)
        v.addSpacing(6)
        v.addWidget(label("Type a prompt on the left, pick a style, press Dream.  Or try one of these:",
                          "herosub"), 0, Qt.AlignHCenter)
        v.addSpacing(18)
        grid = QGridLayout()
        grid.setSpacing(10)
        for i, ex in enumerate(EXAMPLES):
            b = QPushButton(ex)
            b.setObjectName("example")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, t=ex: self.use_example(t))
            grid.addWidget(b, i // 2, i % 2)
        wrap = QHBoxLayout()
        wrap.addStretch(1)
        wrap.addLayout(grid)
        wrap.addStretch(1)
        v.addLayout(wrap)
        v.addSpacing(22)
        v.addWidget(label("Tip: “Sharpest” gives the most detail · faces work best with “Portrait” "
                          "plus a face photo as image prompt", "herotip"), 0, Qt.AlignHCenter)
        v.addStretch(3)
        return page

    def _build_main(self):
        main = QWidget()
        mv = QVBoxLayout(main)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(0)

        stage = QWidget()
        st = QVBoxLayout(stage)
        st.setContentsMargins(26, 16, 26, 12)
        st.setSpacing(10)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.title = label("", "imgtitle")
        self.title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        bar.addWidget(self.title, 1)
        self.zoom_btn = QPushButton("Fit")
        self.zoom_btn.setObjectName("tool")
        self.zoom_btn.setToolTip("Toggle Fit / 100%  (Ctrl+0, or double-click the image)")
        self.zoom_btn.clicked.connect(lambda: self.view.set_fit(not self.view.fit))
        self.continue_btn = QPushButton("Keep dreaming +200")
        self.continue_btn.setObjectName("tool")
        self.continue_btn.setToolTip("Continue the last result for 200 more steps (uses the prompt box)")
        self.continue_btn.clicked.connect(self.continue_dream)
        self.reveal_btn = QPushButton("Show in folder")
        self.reveal_btn.setObjectName("tool")
        self.reveal_btn.clicked.connect(self.reveal_current)
        for w in (self.zoom_btn, self.continue_btn, self.reveal_btn):
            w.setCursor(Qt.PointingHandCursor)
            bar.addWidget(w)
        self.continue_btn.setEnabled(False)
        self.reveal_btn.setEnabled(False)
        st.addLayout(bar)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_welcome())
        canvas = QFrame()
        canvas.setObjectName("canvas")
        cv = QVBoxLayout(canvas)
        cv.setContentsMargins(0, 0, 0, 0)
        self.view = ImageView()
        self.view.fit_changed.connect(lambda fit: self.zoom_btn.setText("Fit" if fit else "100%"))
        cv.addWidget(self.view)
        self.stack.addWidget(canvas)
        st.addWidget(self.stack, 1)

        self.var_strip = QListWidget()
        self.var_strip.setViewMode(QListWidget.IconMode)
        self.var_strip.setFlow(QListWidget.LeftToRight)
        self.var_strip.setWrapping(False)
        self.var_strip.setIconSize(QSize(64, 64))
        self.var_strip.setFixedHeight(84)
        self.var_strip.setMovement(QListWidget.Static)
        self.var_strip.currentRowChanged.connect(self.set_focus)
        self.var_strip.hide()
        st.addWidget(self.var_strip)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        st.addWidget(self.progress)
        mv.addWidget(stage, 1)

        gp = QWidget()
        gp.setObjectName("gallery_panel")
        gl = QVBoxLayout(gp)
        gl.setContentsMargins(26, 10, 26, 12)
        gl.setSpacing(6)
        gh = QHBoxLayout()
        self.gallery_title = label("Gallery", "heading")
        gh.addWidget(self.gallery_title)
        gh.addStretch(1)
        gh.addWidget(label("double-click to view · right-click for more"))
        refresh = QToolButton()
        refresh.setObjectName("ghost")
        refresh.setText("⟳")
        refresh.setToolTip("Refresh")
        refresh.clicked.connect(self.refresh_gallery)
        gh.addWidget(refresh)
        gl.addLayout(gh)
        self.gallery = QListWidget()
        self.gallery.setObjectName("gallery")
        self.gallery.setViewMode(QListWidget.IconMode)
        self.gallery.setFlow(QListWidget.LeftToRight)
        self.gallery.setWrapping(False)
        self.gallery.setIconSize(QSize(104, 104))
        self.gallery.setSpacing(5)
        self.gallery.setFixedHeight(134)
        self.gallery.setMovement(QListWidget.Static)
        self.gallery.setSelectionMode(QListWidget.ExtendedSelection)
        self.gallery.setContextMenuPolicy(Qt.CustomContextMenu)
        self.gallery.customContextMenuRequested.connect(self.gallery_menu)
        self.gallery.itemDoubleClicked.connect(self.show_gallery_item)
        gl.addWidget(self.gallery)
        self.gallery_empty = label("Your renders will appear here.", "hint")
        gl.addWidget(self.gallery_empty)
        mv.addWidget(gp)
        return main

    def _build_statusbar(self):
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        self.status = label("Starting…", "status")
        sb.addWidget(self.status, 1)
        self.speed_label = label("", "status")
        self.vram_label = label("", "status")
        self.gpu_w = label("", "status")
        for w in (self.speed_label, self.vram_label, self.gpu_w):
            sb.addPermanentWidget(w)

    # ================================================================== settings
    def _load_settings(self):
        q = self.qs
        self._applying = True
        self.first_launch = q.value("v11/initialized") is None
        self.prompt.setPlainText(q.value("prompt", ""))
        neg = q.value("negative", "")
        self.negative.setText(neg)
        self._toggle_avoid(bool(neg))
        base_idx = int(q.value("v11/base_idx", 2))
        if q.value("v13/base_reset") is None:  # older versions wrongly pitched 128/256 as "fast"
            base_idx = 2
            q.setValue("v13/base_reset", 1)
        self.base_seg.set_index(base_idx)
        self.size_seg.set_index(int(q.value("v14/size_idx", 0)))
        self.custom_size.setValue(int(q.value("v11/custom_size", 1536)))
        self.sharpen.setValue(int(q.value("sharpen", 75)))
        self.sharpen_val.setText(f"{self.sharpen.value()}%")
        self.fmt.setCurrentText(q.value("v11/format", "PNG"))
        self.steps_spin.setValue(int(q.value("steps", 350)))
        self.batch.setValue(int(q.value("batch", 1)))
        self.cutouts.setValue(int(q.value("v11/cutouts", 64)))
        self.quality_seg.set_index(next((i for i, (_, n) in enumerate(QUALITY) if n == self.cutouts.value()), 1))
        self.lr.setValue(float(q.value("lr", 0.07)))
        self.max_classes.setValue(int(q.value("max_classes", 0)))
        self.class_temp.setValue(float(q.value("class_temp", 2.0)))
        self.ema.setValue(float(q.value("ema", 0.99)))
        self.interp.setCurrentText(q.value("v14/interp", "bilinear"))
        self.clip_model.setCurrentText(q.value("clip", "ViT-B/32"))
        self.precision.setCurrentText(q.value("precision", "auto"))
        self.ckpt.setCurrentText(q.value("ckpt", "auto"))
        self.chunk.setValue(int(q.value("chunk", 0)))
        self.preview_every.setValue(int(q.value("preview_every", 10)))
        self.center_bias.setChecked(q.value("center_bias", "false") == "true")
        self.compile.setChecked(q.value("compile", "false") == "true")
        self.auto_stop.setChecked(q.value("v12/auto_stop", "true") == "true")
        self._applying = False
        preset = q.value("v14/preset", "Balanced")
        if preset in PRESETS and q.value("v14/preset") is None:
            self.apply_preset(preset, announce=False)  # first run of this version: start from Balanced
        else:
            self._show_preset(preset if preset in PRESETS else None)
        self._update_folder_button()
        self.update_hints()

    def _save_settings(self):
        q = self.qs
        q.setValue("v11/initialized", 1)
        q.setValue("prompt", self.prompt.toPlainText())
        q.setValue("negative", self.negative.text())
        q.setValue("v11/base_idx", self.base_seg.index())
        q.setValue("v14/size_idx", self.size_seg.index())
        q.setValue("v11/custom_size", self.custom_size.value())
        q.setValue("sharpen", self.sharpen.value())
        q.setValue("v11/format", self.fmt.currentText())
        q.setValue("steps", self.steps_spin.value())
        q.setValue("batch", self.batch.value())
        q.setValue("v11/cutouts", self.cutouts.value())
        q.setValue("lr", self.lr.value())
        q.setValue("max_classes", self.max_classes.value())
        q.setValue("class_temp", self.class_temp.value())
        q.setValue("ema", self.ema.value())
        q.setValue("v14/interp", self.interp.currentText())
        q.setValue("clip", self.clip_model.currentText())
        q.setValue("precision", self.precision.currentText())
        q.setValue("ckpt", self.ckpt.currentText())
        q.setValue("chunk", self.chunk.value())
        q.setValue("preview_every", self.preview_every.value())
        q.setValue("center_bias", "true" if self.center_bias.isChecked() else "false")
        q.setValue("compile", "true" if self.compile.isChecked() else "false")
        q.setValue("v12/auto_stop", "true" if self.auto_stop.isChecked() else "false")
        q.setValue("v14/preset", self.active_preset or "")
        q.setValue("out_dir", str(self.out_dir))

    # ---- presets
    def apply_preset(self, name, announce=True):
        p = PRESETS.get(name)
        if not p:
            return
        self._applying = True
        if self.init_from is None:
            self.base_seg.set_index(next(i for i, (n, _) in enumerate(BASES) if n == p["base"]))
        self.cutouts.setValue(p["cutouts"])
        q = next((i for i, (_, n) in enumerate(QUALITY) if n == p["cutouts"]), None)
        if q is not None:
            self.quality_seg.set_index(q)
        self.steps_spin.setValue(p["steps"])
        self.max_classes.setValue(p["maxc"])
        self.class_temp.setValue(2.0)
        self.center_bias.setChecked(p["center"])
        self.interp.setCurrentText(p["interp"])
        self.sharpen.setValue(p["sharpen"])
        if "size" in p:
            self.size_seg.set_index(p["size"])
        self._applying = False
        self._show_preset(name)
        self.update_hints()
        if announce:
            self.status.setText(f"{name}: {p['tip']}")

    def _show_preset(self, name):
        self.active_preset = name
        self.preset_group.setExclusive(False)
        for n, b in self.preset_buttons.items():
            b.setChecked(n == name)
        self.preset_group.setExclusive(True)
        self.preset_hint.setText(PRESETS[name]["tip"] if name else "Custom settings (changed in Fine-tune).")

    def _mark_custom(self, *_):
        if not self._applying and self.active_preset is not None:
            self._show_preset(None)

    def use_example(self, text):
        self.prompt.setPlainText(text)
        self.prompt.setFocus()
        self.status.setText("Example loaded - press Dream (or Ctrl+Enter)")

    def _toggle_avoid(self, show):
        self.negative.setVisible(show)
        self.avoid_link.setVisible(not show)
        if show and self.isVisible():
            self.negative.setFocus()

    def apply_meta(self, m):
        if not m:
            return
        self._applying = True
        self.prompt.setPlainText(m.get("prompt", ""))
        self.negative.setText(m.get("negative", ""))
        self._toggle_avoid(bool(m.get("negative")))
        native = int(m.get("image_size", 512))
        self.base_seg.set_index(next((i for i, (n, _) in enumerate(BASES) if n == native), 2))
        out = int(m.get("output_size") or 0)
        idx = next((i for i, (_, v) in enumerate(SIZES) if v == out), None)
        if idx is None:
            idx = CUSTOM
            self.custom_size.setValue(out)
        self.size_seg.set_index(idx)
        self.sharpen.setValue(int(round(float(m.get("upscale_strength", 1.0)) * 100)))
        self.steps_spin.setValue(int(m.get("steps", 300)))
        self.seed.setText(str(m.get("seed", "")))
        self.cutouts.setValue(int(m.get("cutouts", 64)))
        q = next((i for i, (_, n) in enumerate(QUALITY) if n == self.cutouts.value()), None)
        if q is not None:
            self.quality_seg.set_index(q)
        self.lr.setValue(float(m.get("lr", 0.07)))
        self.max_classes.setValue(int(m.get("max_classes") or 0))
        self.class_temp.setValue(float(m.get("class_temperature", 2.0)))
        self.ema.setValue(float(m.get("ema_decay", 0.99)))
        self.interp.setCurrentText(m.get("interp", "nearest"))
        self.clip_model.setCurrentText(m.get("clip_model", "ViT-B/32"))
        self.center_bias.setChecked(bool(m.get("center_bias")))
        self._applying = False
        self._show_preset(None)
        self.update_hints()

    def output_size(self):
        v = SIZES[self.size_seg.index()][1]
        return self.custom_size.value() if v == -1 else v

    def native_size(self):
        return self.init_from[2] if self.init_from else BASES[self.base_seg.index()][0]

    def build_settings(self):
        from ..engine import DreamSettings
        txt = self.seed.text().strip()
        seed = int(txt) if txt.lstrip("-").isdigit() else random.randrange(2**31)
        size, out = self.native_size(), self.output_size()
        return DreamSettings(
            prompt=self.prompt.toPlainText().strip(),
            negative=self.negative.text().strip(),
            image_prompt=self.image_prompt,
            image_size=size,
            output_size=out if out and out != size else 0,
            upscale_strength=self.sharpen.value() / 100,
            steps=self.steps_spin.value(),
            lr=self.lr.value(),
            cutouts=self.cutouts.value(),
            max_classes=self.max_classes.value() or None,
            class_temperature=self.class_temp.value(),
            ema_decay=self.ema.value(),
            seed=seed,
            batch=1 if self.init_from else self.batch.value(),
            preview_every=self.preview_every.value(),
            interp=self.interp.currentText(),
            center_bias=self.center_bias.isChecked(),
            clip_model=self.clip_model.currentText(),
            precision=self.precision.currentText(),
            compile=self.compile.isChecked(),
            checkpointing=self.ckpt.currentText(),
            cutout_chunk=self.chunk.value(),
            auto_stop=self.auto_stop.isChecked(),
            init_latents=[self.init_from[1]] if self.init_from else None,
        )

    def make_job(self):
        s = self.build_settings()
        job = {"settings": s, "out_dir": self.out_dir, "fmt": self.fmt.currentText()}
        if self.init_from:
            job["started_from"] = self.init_from[0].name
        return job

    def _steps_changed(self, v):
        self.steps_slider.blockSignals(True)
        self.steps_slider.setValue(max(1, min(40, round(v / 25))))
        self.steps_slider.blockSignals(False)
        self.update_hints()

    # ---- hints & estimate
    def _speed_key(self):
        return f"speed/{self.native_size()}/{self.cutouts.value()}/{self.batch.value()}"

    def update_hints(self):
        if not hasattr(self, "estimate") or not hasattr(self, "auto_stop"):
            return  # still building
        n = self.cutouts.value()
        names = {32: "Draft", 64: "Balanced", 128: "Best"}
        self.quality_hint.setText(f"{n} CLIP cutouts per step" +
                                  ("" if n in names else "  ·  custom") +
                                  ("  ·  the original 2021 setting" if n == 128 else ""))
        native = self.native_size()
        out = self.output_size()
        self.custom_size.setVisible(SIZES[self.size_seg.index()][1] == -1)
        upscale = bool(out) and out > native
        self.sharpen_row.setVisible(upscale)
        warn = False
        if not out or out == native:
            text = f"{native} × {native}, straight from BigGAN (no upscaling)."
        elif upscale:
            p = upscale_passes(native, out)
            text = f"BigGAN paints at {native}px → {p} AI upscale pass{'es' if p > 1 else ''} → {out} × {out}."
            if out / native > 4.5:
                warn = True
                text += f"  Enlarging a {native}px canvas {out // native}× looks blurry - set Base model to 512."
        else:
            text = f"Downscaled from {native}px to {out} × {out}."
        self.size_hint.setText(text)
        restyle(self.size_hint, "warn" if warn else "hint")
        self.base_seg.set_all_enabled(self.init_from is None)
        self._update_estimate()

    def _update_estimate(self):
        if not hasattr(self, "estimate"):
            return
        out = self.output_size() or self.native_size()
        its = float(self.qs.value(self._speed_key(), 0) or 0)
        steps = self.steps_spin.value()
        if its > 0:
            upscale = {0: 0, 1: 3, 2: 8, 3: 40}.get(upscale_passes(self.native_size(), out)
                                                     if out > self.native_size() else 0, 60)
            t = steps / its + upscale
            prefix = "up to " if self.auto_stop.isChecked() else ""
            self.estimate.setText(f"{out} × {out}  ·  {prefix}{fmt_minutes(t)} on your GPU")
        elif self.engine_ready:
            self.estimate.setText(f"{out} × {out}  ·  first render at these settings also tunes your GPU (~30 s)")
        else:
            self.estimate.setText(f"{out} × {out}")

    def toggle_advanced(self, on):
        self.adv.setVisible(on)
        self.adv_btn.setText(("▾" if on else "▸") + "  Fine-tune")

    def _update_folder_button(self):
        p = str(self.out_dir)
        fm = self.folder_btn.fontMetrics()
        self.folder_btn.setText("📁  " + fm.elidedText(p, Qt.ElideMiddle, 200))
        self.folder_btn.setToolTip(f"Render folder:\n{p}\n\nClick to change")
        if hasattr(self, "gallery_title"):
            self.gallery_title.setText(f"Gallery  ·  {self.out_dir.name}")

    # ================================================================== engine events
    def on_log(self, text):
        self.status.setText(text)

    def on_ready(self, gpu, vram):
        self.engine_ready = True
        self.vram = vram
        self.gpu_label = gpu
        self.gpu_w.setText(gpu)
        self.dream_btn.setEnabled(True)
        self.status.setText("Ready")
        if self.first_launch:
            self.apply_preset("Fast draft" if vram and vram < 3.5 else "Balanced", announce=False)
            self.size_seg.set_index(0)
            self._save_settings()
            self.first_launch = False
        self.update_hints()
        if self.queue and not self.queue_paused:
            self.start_next()

    def on_download(self, name, done, total):
        mb = 1024 * 1024
        self.progress.setRange(0, 1000)
        self.progress.setValue(int(1000 * done / total) if total else 0)
        self.status.setText(f"Downloading {name}  ·  {done / mb:.0f} / {total / mb:.0f} MB  (first time only)")

    def on_preview(self, p):
        if p.final:
            return
        self.current = p.images
        self.show_current()
        self.progress.setRange(0, p.total)
        self.progress.setValue(p.step)
        eta = (p.total - p.step) / p.its if p.its > 0 else 0
        settle = f"  ·  change {p.change:.4f}" if p.change >= 0 else ""
        self.status.setText(f"Dreaming  ·  step {p.step} / {p.total}  ·  loss {p.loss:.2f}{settle}")
        self.speed_label.setText(f"{p.its:.2f} it/s  ·  {fmt_time(eta)} left")
        if p.vram:
            self.vram_label.setText(f"VRAM {p.vram:.1f} GB")

    def show_current(self, keep_view=True):
        imgs = self.current
        if not imgs:
            return
        self.stack.setCurrentIndex(1)
        self.focus = min(self.focus, len(imgs) - 1)
        self.view.set_image(imgs[self.focus], keep_view)
        if len(imgs) > 1:
            self.var_strip.show()
            self.var_strip.blockSignals(True)
            if self.var_strip.count() != len(imgs):
                self.var_strip.clear()
                for _ in imgs:
                    self.var_strip.addItem(QListWidgetItem())
            for i, im in enumerate(imgs):
                t = im.copy()
                t.thumbnail((64, 64))
                self.var_strip.item(i).setIcon(QIcon(QPixmap.fromImage(pil_to_qimage(t))))
            self.var_strip.setCurrentRow(self.focus)
            self.var_strip.blockSignals(False)
        else:
            self.var_strip.hide()

    def set_focus(self, i):
        if i >= 0:
            self.focus = i
            self.show_current()
            if self.last_paths and i < len(self.last_paths):
                self.title.setText(self.last_paths[i].name)

    def on_finished(self, p, job):
        s = job["settings"]
        self.set_running(False)
        if p.step == 0:
            self.status.setText("Stopped before the first step - nothing saved")
            self._after_job()
            return
        self.current = p.images
        meta = s.to_meta()
        meta.update(steps_done=job.get("steps_before", 0) + p.step, seconds=round(time.time() - self.run_started, 1),
                    started_from=job.get("started_from"), app_version=__version__)
        out_dir, fmt = Path(job["out_dir"]), job["fmt"]
        try:
            self.last_paths = [storage.save_dream(out_dir, im, meta, p.latents[i], i, fmt)
                               for i, im in enumerate(p.images)]
        except Exception as e:
            QMessageBox.critical(self, "VAIM Studio", f"Could not save to {out_dir}:\n{e}")
            self.last_paths = []
        self.last, self.last_job = p, job
        if p.its > 0 and not s.init_latents:
            self.qs.setValue(f"speed/{s.image_size}/{s.cutouts}/{s.batch}", round(p.its, 3))
        self.show_current()
        took = time.time() - self.run_started
        size = p.images[0].width
        self.title.setText(self.last_paths[min(self.focus, len(self.last_paths) - 1)].name if self.last_paths else "")
        self.status.setText(f"Saved {len(self.last_paths)} image(s) at {size} × {size}  ·  "
                            f"{p.step} steps in {fmt_time(took)}" + (f"  ·  {p.note} (auto-finish)" if p.note else ""))
        self.speed_label.setText(f"{p.its:.2f} it/s")
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.continue_btn.setEnabled(True)
        self.reveal_btn.setEnabled(bool(self.last_paths))
        if out_dir == self.out_dir:
            self.refresh_gallery()
        self._update_estimate()
        if not self.isActiveWindow():
            QApplication.alert(self, 0)
        self._after_job()

    def on_failed(self, msg, job=None):
        self.set_running(False)
        short = msg.split("\n\n")[0]
        if short == "Download cancelled." or self._stopping:
            self.status.setText("Stopped")
        else:
            self.status.setText((short.splitlines() or ["Error"])[0][:160])
            self.queue_paused = True  # don't plough through the queue after an error
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("VAIM Studio")
            box.setText(short)
            box.setDetailedText(msg)
            box.exec()
        self._after_job()

    def _after_job(self):
        self._stopping = False
        self.refresh_queue()
        if self.queue and not self.queue_paused and self.engine_ready:
            QTimer.singleShot(400, self.start_next)

    # ================================================================== actions
    def set_running(self, on):
        self.running = on
        self.dream_btn.setText("Stop" if on else "Dream")
        self.dream_btn.setToolTip("Stop the current render  (Esc)" if on else "Render now  (Ctrl+Enter)")
        self.dream_btn.setProperty("running", "true" if on else "false")
        self.dream_btn.style().unpolish(self.dream_btn)
        self.dream_btn.style().polish(self.dream_btn)
        self.continue_btn.setEnabled(not on and self.last is not None)
        if on:
            self.speed_label.setText("")
        self.refresh_queue()

    def stop(self):
        if self.running and self.worker:
            self._stopping = True
            self.queue_paused = True
            self.worker.stop()
            self.status.setText("Stopping after this step…")

    def dream_or_stop(self):
        if self.running:
            self.stop()
            return
        if not self.engine_ready:
            return
        job = self.make_job()
        if not job["settings"].prompt and not job["settings"].image_prompt:
            self.prompt.setFocus()
            self.status.setText("Type a prompt first")
            return
        self._save_settings()
        self.launch(job)

    def launch(self, job):
        self.job = job
        self.current, self.focus = [], 0
        self.var_strip.clear()
        self.var_strip.hide()
        self.run_started = time.time()
        self.set_running(True)
        self.progress.setRange(0, job["settings"].steps)
        self.progress.setValue(0)
        self.title.setText(job["settings"].prompt[:120])
        self.status.setText("Loading models…")
        self.start_dream.emit(job)

    def continue_dream(self):
        if not self.last or self.running or not self.last_job:
            return
        prev = self.last_job["settings"]
        s = replace(prev, steps=200, init_latents=self.last.latents, batch=len(self.last.latents),
                    prompt=self.prompt.toPlainText().strip() or prev.prompt, negative=self.negative.text().strip())
        job = {"settings": s, "out_dir": self.out_dir, "fmt": self.fmt.currentText(),
               "steps_before": self.last_job.get("steps_before", 0) + self.last.step,
               "started_from": self.last_paths[0].name if self.last_paths else None}
        self.launch(job)

    # ---- queue
    def add_to_queue(self):
        job = self.make_job()
        if not job["settings"].prompt and not job["settings"].image_prompt:
            self.prompt.setFocus()
            self.status.setText("Type a prompt first")
            return
        self._save_settings()
        self.queue.append(job)
        self.status.setText(f"Added to queue  ·  {len(self.queue)} waiting")
        if not self.running and self.engine_ready:
            self.queue_paused = False
            self.start_next()
        else:
            self.refresh_queue()

    def start_next(self):
        if self.running or not self.queue or not self.engine_ready:
            self.refresh_queue()
            return
        self.launch(self.queue.pop(0))

    def resume_queue(self):
        self.queue_paused = False
        self.start_next()

    def clear_queue(self):
        self.queue.clear()
        self.refresh_queue()

    def refresh_queue(self):
        if not hasattr(self, "queue_list"):
            return
        self.queue_list.clear()
        for job in self.queue:
            s = job["settings"]
            size = s.output_size or s.image_size
            it = QListWidgetItem(f"{size}px · {s.prompt[:70] or '(image prompt)'}")
            it.setToolTip(f"{s.prompt}\nBase {s.image_size}px → {size}px, {s.cutouts} cutouts, {s.steps} steps, "
                          f"seed {s.seed}\n→ {job['out_dir']}  ({job['fmt']})")
            self.queue_list.addItem(it)
        n = len(self.queue)
        self.queue_box.setVisible(n > 0)
        paused = self.queue_paused and not self.running
        self.queue_title.setText(f"Queue  ·  {n} waiting" + ("  ·  paused" if paused else ""))
        self.queue_start.setVisible(paused and n > 0)

    def queue_menu(self, pos):
        item = self.queue_list.itemAt(pos)
        if item is None:
            return
        row = self.queue_list.row(item)
        menu = QMenu(self)
        menu.addAction("Remove", lambda: (self.queue.pop(row), self.refresh_queue()))
        if row > 0:
            menu.addAction("Move up", lambda: (self.queue.insert(row - 1, self.queue.pop(row)), self.refresh_queue()))
        menu.addAction("Load its settings", lambda: self.apply_meta(self.queue[row]["settings"].to_meta()))
        menu.exec(self.queue_list.mapToGlobal(pos))

    # ---- misc
    def reuse_seed(self):
        if self.last_job:
            self.seed.setText(str(self.last_job["settings"].seed))

    def pick_image_prompt(self):
        p, _ = QFileDialog.getOpenFileName(self, "Image prompt", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if p:
            self.set_image_prompt(p)

    def set_image_prompt(self, p):
        self.image_prompt = p
        self.img_btn.setText(f"🖼  {Path(p).name[:28]}" if p else "+ Image prompt")
        self.img_clear.setVisible(bool(p))

    def pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Render to folder", str(self.out_dir))
        if d:
            self.out_dir = Path(d)
            self._update_folder_button()
            self._save_settings()
            self.refresh_gallery()

    def clear_init(self):
        self.init_from = None
        self.init_chip.hide()
        self.init_chip_clear.hide()
        self.batch.setEnabled(True)
        self.update_hints()

    def reveal_current(self):
        if self.last_paths:
            storage.open_in_explorer(self.last_paths[min(self.focus, len(self.last_paths) - 1)])
        else:
            storage.open_in_explorer(self.out_dir)

    def reset_tuning(self):
        from ..engine.weights import app_home
        try:
            (app_home() / "autotune.json").unlink()
        except FileNotFoundError:
            pass
        self.status.setText("GPU tuning reset - the next render re-measures your card")

    def show_about(self):
        QMessageBox.about(
            self, "About VAIM Studio",
            f"<h3>VAIM Studio {__version__}</h3>"
            "<p>Vintage AI Model Studio: BigGAN-deep steered by CLIP, the 2021 “Big Sleep” method, "
            "rebuilt for speed on consumer GPUs, with Real-ESRGAN upscaling.</p>"
            f"<p>GPU: {self.gpu_label or 'not started yet'}</p>"
            "<p style='color:#9a98a8'>big-sleep by Phil Wang &amp; Ryan Murdock (MIT) · BigGAN port by "
            "Hugging Face · CLIP by OpenAI · Real-ESRGAN by Xintao Wang et al.</p>")

    # ================================================================== gallery
    def refresh_gallery(self):
        self._update_folder_button()
        self.gallery.clear()
        for p in storage.list_images(self.out_dir)[:400]:
            tp = storage.thumbnail(p)
            if tp is None:
                continue
            it = QListWidgetItem(QIcon(str(tp)), "")
            it.setData(Qt.UserRole, str(p))
            m = storage.read_meta(p) or {}
            it.setToolTip(f"{m.get('prompt', p.name)}\n{p.name}")
            self.gallery.addItem(it)
        empty = self.gallery.count() == 0
        self.gallery.setVisible(not empty)
        self.gallery_empty.setVisible(empty)

    def _item_path(self, it):
        return Path(it.data(Qt.UserRole))

    def show_gallery_item(self, it):
        p = self._item_path(it)
        try:
            self.current = [Image.open(p).convert("RGB")]
        except Exception as e:
            self.status.setText(f"Could not open {p.name}: {e}")
            return
        self.focus = 0
        self.var_strip.hide()
        self.stack.setCurrentIndex(1)
        self.view.set_image(self.current[0], keep_view=False)
        self.title.setText(p.name)
        self.last_paths = [p]
        self.reveal_btn.setEnabled(True)
        m = storage.read_meta(p)
        if m:
            w = self.current[0].width
            self.status.setText(f"{w} × {w}  ·  {m.get('steps_done', m.get('steps'))} steps  ·  seed {m.get('seed')}"
                                f"  ·  {m.get('prompt', '')[:80]}")

    def gallery_menu(self, pos):
        items = self.gallery.selectedItems()
        if not items:
            it = self.gallery.itemAt(pos)
            if not it:
                return
            it.setSelected(True)
            items = [it]
        menu = QMenu(self)
        if len(items) == 1:
            p = self._item_path(items[0])
            m = storage.read_meta(p)
            a = menu.addAction("Re-dream from this (new prompt)", lambda: self.use_as_start(p))
            a.setEnabled(storage.latents_path(p).is_file() and m is not None)
            a = menu.addAction("Load its settings", lambda: self.apply_meta(m))
            a.setEnabled(m is not None)
            menu.addAction("Copy prompt", lambda: QApplication.clipboard().setText((m or {}).get("prompt", "")))
            menu.addSeparator()
            menu.addAction("Open", lambda: storage.open_file(p))
            menu.addAction("Show in folder", lambda: storage.open_in_explorer(p))
        menu.addSeparator()
        menu.addAction(f"Delete {len(items)} images…" if len(items) > 1 else "Delete…", lambda: self.delete_items(items))
        menu.exec(self.gallery.mapToGlobal(pos))

    def use_as_start(self, p):
        m = storage.read_meta(p)
        lat = storage.read_latents(p)
        if not m or lat is None:
            return
        self.init_from = (p, lat, int(m["image_size"]), m)
        self.apply_meta(m)
        self.seed.setText("")
        self.batch.setValue(1)
        self.batch.setEnabled(False)
        self.init_chip.setText(f"Starting from: {m.get('prompt', p.name)[:80]}")
        self.init_chip.show()
        self.init_chip_clear.show()
        self.update_hints()
        self.prompt.setFocus()
        self.prompt.selectAll()
        self.status.setText("Type the new prompt and press Dream - the image will drift toward it")

    def delete_items(self, items):
        if QMessageBox.question(self, "Delete", f"Delete {len(items)} image(s) from disk?") != QMessageBox.Yes:
            return
        for it in items:
            storage.delete_dream(self._item_path(it))
        self.refresh_gallery()

    # ================================================================== shutdown
    def closeEvent(self, e):
        if self.running and QMessageBox.question(
                self, "VAIM Studio", "A render is in progress. Quit anyway?") != QMessageBox.Yes:
            e.ignore()
            return
        self._save_settings()
        if self.worker:
            self.worker.stop()
            self.engine_thread.quit()
            self.engine_thread.wait(3000)
        super().closeEvent(e)
