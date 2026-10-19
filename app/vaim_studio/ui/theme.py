from pathlib import Path

_CHEVRON = (Path(__file__).resolve().parent.parent / "assets" / "chevron.svg").as_posix()

# A soft, warm dark palette: charcoal instead of pure black, lavender accent, cream text.
BG = "#121216"
PANEL = "#18181e"
CARD = "#202028"
CARD_HI = "#272731"
BORDER = "#2c2c37"
RULE = "#23232c"
TEXT = "#ecebf2"
MUTED = "#9a98a8"
FAINT = "#62606f"
ACCENT = "#b9a4ff"
ACCENT_HI = "#cfc1ff"
ACCENT_2 = "#e9a7d8"
ACCENT_LO = "#2e2747"
WARN = "#e8b86b"
STOP = "#ff8f80"

QSS = f"""
* {{ font-family: "Segoe UI Variable Text", "Segoe UI", "Inter", sans-serif; font-size: 10pt; color: {TEXT}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QWidget#sidebar {{ background: {PANEL}; }}
QWidget#dock {{ background: {PANEL}; border-top: 1px solid {RULE}; }}
QWidget#gallery_panel {{ background: {PANEL}; border-top: 1px solid {RULE}; }}
QWidget#welcome {{ background: transparent; }}
QFrame#canvas {{ background: #0e0e12; border-radius: 12px; }}
QFrame#drawer {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; }}

QLabel#wordmark {{ font-family: Georgia, "Times New Roman", serif; font-style: italic; font-size: 22pt; color: {TEXT}; }}
QLabel#tagline {{ color: {MUTED}; font-size: 8.5pt; letter-spacing: 1px; }}
QLabel#heading {{ color: {TEXT}; font-size: 10.5pt; font-weight: 600; }}
QLabel#subheading {{ color: {ACCENT}; font-size: 8pt; font-weight: 700; letter-spacing: 1.5px; margin-top: 6px; }}
QLabel#field {{ color: {TEXT}; font-size: 9pt; font-weight: 600; }}
QLabel#value {{ color: {ACCENT_HI}; font-size: 9pt; font-family: Consolas, "Cascadia Mono", monospace; }}
QLabel#hint {{ color: {MUTED}; font-size: 8.5pt; }}
QLabel#warn {{ color: {WARN}; font-size: 8.5pt; }}
QLabel#estimate {{ color: {MUTED}; font-size: 9pt; }}
QLabel#imgtitle {{ color: {MUTED}; font-size: 9.5pt; }}
QLabel#status {{ font-family: Consolas, "Cascadia Mono", monospace; font-size: 8.5pt; color: {MUTED}; padding: 0 10px; }}
QLabel#chip {{ background: {ACCENT_LO}; border: 1px solid {ACCENT}; border-radius: 8px; padding: 6px 10px; font-size: 9pt; }}
QLabel#hero {{ font-family: Georgia, "Times New Roman", serif; font-style: italic; font-size: 26pt; color: {TEXT}; }}
QLabel#herosub {{ color: {MUTED}; font-size: 10.5pt; }}
QLabel#herotip {{ color: {FAINT}; font-size: 9pt; }}

QStatusBar {{ background: {PANEL}; border-top: 1px solid {RULE}; min-height: 26px; }}
QStatusBar::item {{ border: none; }}

QPlainTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 9px;
    selection-background-color: {ACCENT_LO};
}}
QPlainTextEdit:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{ border-color: #3a3a48; }}
QPlainTextEdit:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QPlainTextEdit#prompt {{ font-size: 11.5pt; padding: 9px 11px; border-radius: 10px; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: url("{_CHEVRON}"); width: 10px; height: 6px; margin-right: 8px; }}
QComboBox QAbstractItemView {{ background: {CARD}; border: 1px solid {BORDER}; padding: 4px;
    selection-background-color: {ACCENT_LO}; outline: none; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: none; }}
QSpinBox:disabled, QLineEdit:disabled {{ color: {FAINT}; }}

QPushButton {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 7px 12px; }}
QPushButton:hover {{ background: {CARD_HI}; border-color: #3a3a48; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {FAINT}; border-color: {RULE}; background: {PANEL}; }}

QPushButton#pill {{ border-radius: 8px; padding: 7px 0; font-size: 9.5pt; color: {MUTED}; }}
QPushButton#pill:hover {{ color: {TEXT}; }}
QPushButton#pill:checked {{ background: {ACCENT_LO}; border-color: {ACCENT}; color: {TEXT}; font-weight: 600; }}
QPushButton#pill:disabled:checked {{ background: {CARD}; border-color: {BORDER}; color: {MUTED}; }}

QPushButton#preset {{ text-align: left; padding: 8px 12px; border-radius: 10px; font-size: 9.5pt; color: {MUTED}; }}
QPushButton#preset:hover {{ color: {TEXT}; border-color: #44445a; }}
QPushButton#preset:checked {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2f2850, stop:1 #3a2a48);
    border: 1px solid {ACCENT}; color: {TEXT}; }}

QPushButton#example {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 18px; padding: 9px 18px;
    color: {TEXT}; font-size: 10pt; min-width: 230px; }}
QPushButton#example:hover {{ border-color: {ACCENT}; background: {CARD_HI}; }}

QPushButton#dream {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT}, stop:1 {ACCENT_2});
    color: #17121f; border: none; font-size: 12.5pt; font-weight: 700; padding: 11px; border-radius: 10px; }}
QPushButton#dream:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_HI}, stop:1 #f2bfe3); }}
QPushButton#dream:disabled {{ background: {ACCENT_LO}; color: {MUTED}; }}
QPushButton#dream[running="true"] {{ background: transparent; color: {STOP}; border: 1px solid {STOP}; }}
QPushButton#secondary {{ padding: 11px 14px; border-radius: 10px; font-weight: 600; }}
QPushButton#tool {{ background: transparent; border: 1px solid {BORDER}; padding: 5px 11px; font-size: 9pt; border-radius: 7px; }}
QPushButton#tool:hover {{ border-color: {ACCENT}; }}
QPushButton#folder {{ text-align: left; padding: 7px 10px; font-size: 9pt; }}
QPushButton#link {{ background: transparent; border: none; color: {ACCENT}; padding: 3px 6px; font-size: 9pt; }}
QPushButton#link:hover {{ color: {ACCENT_HI}; text-decoration: underline; }}
QToolButton#ghost {{ background: transparent; border: none; color: {MUTED}; padding: 4px 6px; font-size: 11pt; }}
QToolButton#ghost:hover {{ color: {TEXT}; }}
QToolButton#disclosure {{ background: transparent; border: none; color: {MUTED}; font-size: 10pt; font-weight: 600; padding: 4px 0; }}
QToolButton#disclosure:hover {{ color: {TEXT}; }}

QCheckBox {{ spacing: 8px; color: {TEXT}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {BORDER}; border-radius: 5px; background: {PANEL}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT}, stop:1 {ACCENT_2}); border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; }}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}

QFrame#progress_card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
QLabel#phead {{ font-size: 10.5pt; font-weight: 600; }}
QLabel#pdetail, QLabel#pright {{ color: {MUTED}; font-size: 9pt; }}
QLabel#pchev {{ color: {FAINT}; font-size: 9pt; padding: 0 1px; }}
QLabel#pchip {{ color: {FAINT}; font-size: 8.5pt; padding: 2px 8px; border: 1px solid transparent; border-radius: 9px; }}
QLabel#pchip_on {{ color: {TEXT}; background: {ACCENT_LO}; border: 1px solid {ACCENT}; font-size: 8.5pt;
    font-weight: 600; padding: 2px 8px; border-radius: 9px; }}
QLabel#pchip_done {{ color: {MUTED}; font-size: 8.5pt; padding: 2px 8px; border: 1px solid transparent; }}

QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border: 2px solid transparent; border-radius: 8px; padding: 2px; }}
QListWidget::item:selected {{ border-color: {ACCENT}; background: transparent; }}
QListWidget::item:hover {{ border-color: {BORDER}; }}
QListWidget#queue {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 3px; }}
QListWidget#queue::item {{ border: none; border-radius: 5px; padding: 4px 6px; font-size: 9pt; }}
QListWidget#queue::item:selected {{ background: {ACCENT_LO}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: transparent; width: 8px; height: 8px; margin: 0; }}
QScrollBar::handle {{ background: {BORDER}; border-radius: 4px; min-height: 30px; min-width: 30px; }}
QScrollBar::handle:hover {{ background: #3a3a48; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ width: 0; height: 0; background: none; }}

QMenu {{ background: {CARD}; border: 1px solid {BORDER}; padding: 5px; border-radius: 8px; }}
QMenu::item {{ padding: 7px 22px 7px 14px; border-radius: 5px; }}
QMenu::item:selected {{ background: {ACCENT_LO}; }}
QMenu::item:disabled {{ color: {FAINT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
QToolTip {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; padding: 7px; border-radius: 6px; }}
QMessageBox, QInputDialog, QFileDialog {{ background: {PANEL}; }}
"""
