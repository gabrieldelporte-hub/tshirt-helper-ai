"""Main application window."""

from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QComboBox,
    QGroupBox, QSpinBox, QDoubleSpinBox, QColorDialog,
    QScrollArea, QFrame, QSizePolicy, QCheckBox, QLineEdit,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap, QDragEnterEvent, QDropEvent
from PIL import Image

from app.ui.preview_widget import PreviewWidget
from app.processing.background import (
    remove_solid_background, remove_background_ai, remove_background_bria,
    refine_edges, apply_color_hints, get_background_color,
)
from app.processing.colors import get_dominant_colors, rgb_to_hex, rgb_to_cmyk
from app.processing.exporter import export_png, export_resized, get_image_info, STANDARD_SIZES_CM


# ---------------------------------------------------------------------------
# Worker thread for heavy processing
# ---------------------------------------------------------------------------

class ProcessingWorker(QThread):
    done = pyqtSignal(object)  # emits PIL Image
    error = pyqtSignal(str)

    def __init__(self, image, mode="classic", tolerance=30, target_color=None,
                 exclude_colors=None, protect_colors=None, api_token=""):
        super().__init__()
        self.image = image
        self.mode = mode
        self.tolerance = tolerance
        self.target_color = target_color
        self.exclude_colors = exclude_colors or []
        self.protect_colors = protect_colors or []
        self.api_token = api_token

    def run(self):
        try:
            if self.mode == "ai":
                result = remove_background_ai(self.image)
            elif self.mode == "bria":
                result = remove_background_bria(self.image, self.api_token)
            else:
                result = remove_solid_background(
                    self.image,
                    tolerance=self.tolerance,
                    target_color=self.target_color,
                )
            if self.exclude_colors or self.protect_colors:
                result = apply_color_hints(result, self.image,
                                           self.exclude_colors, self.protect_colors,
                                           self.tolerance)
            self.done.emit(result)
        except (Exception, SystemExit) as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Color swatch
# ---------------------------------------------------------------------------

class ColorSwatch(QFrame):
    def __init__(self, color_info: dict, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        r, g, b = color_info["rgb"]
        swatch = QLabel()
        swatch.setFixedSize(24, 24)
        swatch.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #555; border-radius: 3px;"
        )

        c, m, y, k = color_info["cmyk"]
        info = QLabel(
            f"{color_info['hex']}  |  RGB({r},{g},{b})  |  "
            f"CMJN({c}%, {m}%, {y}%, {k}%)  |  {color_info['percentage']}%"
        )
        info.setStyleSheet("font-size: 11px; color: #ccc;")

        layout.addWidget(swatch)
        layout.addWidget(info)
        layout.addStretch()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("T-Shirt Helper")
        self.setMinimumSize(1100, 700)
        self._apply_dark_theme()

        self.original_image: Image.Image | None = None
        self.processed_image: Image.Image | None = None
        self._refined_image: Image.Image | None = None
        self._worker: ProcessingWorker | None = None
        self._manual_bg_color: tuple[int, int, int] | None = None
        self._refine_timer = QTimer(self)
        self._refine_timer.setSingleShot(True)
        self._refine_timer.setInterval(150)
        self._refine_timer.timeout.connect(self._run_refine)
        self._exclude_colors: list[tuple[int,int,int]] = []
        self._protect_colors: list[tuple[int,int,int]] = []
        self._eyedropper_target: str | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        left = self._build_left_panel()
        root.addWidget(left, stretch=0)

        center = self._build_center_panel()
        root.addWidget(center, stretch=1)

        right = self._build_right_panel()
        root.addWidget(right, stretch=0)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # --- Import ---
        import_group = QGroupBox("Image source")
        ig_layout = QVBoxLayout(import_group)

        self.import_btn = QPushButton("Ouvrir un fichier...")
        self.import_btn.clicked.connect(self._open_file)
        self.file_label = QLabel("Aucun fichier")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: #888; font-size: 11px;")

        ig_layout.addWidget(self.import_btn)
        ig_layout.addWidget(self.file_label)
        layout.addWidget(import_group)

        # --- Background removal ---
        bg_group = QGroupBox("Suppression du fond")
        bg_layout = QVBoxLayout(bg_group)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Methode :"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Classique (flood fill)", "classic")
        self.mode_combo.addItem("IA locale (rembg)", "ai")
        self.mode_combo.addItem("BRIA RMBG 2.0 (Replicate)", "bria")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        bg_layout.addLayout(mode_row)

        self.ai_hint = QLabel("Le modele (~170 Mo) sera telecharge\nautomatiquement au premier usage.")
        self.ai_hint.setStyleSheet("color: #888; font-size: 10px;")
        self.ai_hint.setVisible(False)

        self.bria_hint = QLabel("Renseigne ta cle Replicate dans\nle panneau droit avant de lancer.")
        self.bria_hint.setStyleSheet("color: #e9c46a; font-size: 10px;")
        self.bria_hint.setVisible(False)

        bg_layout.addWidget(self.ai_hint)
        bg_layout.addWidget(self.bria_hint)

        self.classic_widget = QWidget()
        classic_layout = QVBoxLayout(self.classic_widget)
        classic_layout.setContentsMargins(0, 0, 0, 0)

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Tolerance :"))
        self.tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self.tolerance_slider.setRange(0, 120)
        self.tolerance_slider.setValue(30)
        self.tolerance_label = QLabel("30")
        self.tolerance_label.setFixedWidth(28)
        self.tolerance_slider.valueChanged.connect(
            lambda v: self.tolerance_label.setText(str(v))
        )
        tol_row.addWidget(self.tolerance_slider)
        tol_row.addWidget(self.tolerance_label)

        color_row = QHBoxLayout()
        self.bg_color_btn = QPushButton("Couleur manuelle")
        self.bg_color_btn.clicked.connect(self._pick_bg_color)
        self.bg_color_preview = QLabel()
        self.bg_color_preview.setFixedSize(24, 24)
        self.bg_color_preview.setStyleSheet("background: transparent; border: 1px solid #555;")
        self.bg_color_clear = QPushButton("Auto")
        self.bg_color_clear.setFixedWidth(48)
        self.bg_color_clear.clicked.connect(self._clear_bg_color)
        color_row.addWidget(self.bg_color_btn)
        color_row.addWidget(self.bg_color_preview)
        color_row.addWidget(self.bg_color_clear)

        classic_layout.addLayout(tol_row)
        classic_layout.addLayout(color_row)
        bg_layout.addWidget(self.classic_widget)

        self.process_btn = QPushButton("Supprimer le fond")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._run_processing)
        self.process_btn.setStyleSheet(
            "QPushButton { background: #2d6a4f; color: white; font-weight: bold; padding: 6px; border-radius: 4px; }"
            "QPushButton:disabled { background: #333; color: #666; }"
            "QPushButton:hover { background: #40916c; }"
        )
        bg_layout.addWidget(self.process_btn)
        layout.addWidget(bg_group)

        # --- Finition des bords ---
        edge_group = QGroupBox("Finition des bords (temps reel)")
        edge_layout = QVBoxLayout(edge_group)

        def _slider_row(label, lo, hi, default, unit=""):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(80)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(default)
            val_lbl = QLabel(f"{default}{unit}")
            val_lbl.setFixedWidth(30)
            slider.valueChanged.connect(lambda v, l=val_lbl, u=unit: l.setText(f"{v}{u}"))
            row.addWidget(lbl)
            row.addWidget(slider)
            row.addWidget(val_lbl)
            return row, slider

        thr_row, self.threshold_slider = _slider_row("Nettete :", 0, 254, 128)
        thr_hint = QLabel("0 = degrades gardes  |  128 = bords nets  |  254 = max")
        thr_hint.setStyleSheet("color: #555; font-size: 10px;")

        ero_row, self.erode_slider = _slider_row("Erosion :", 0, 10, 0, "px")
        fea_row, self.feather_slider = _slider_row("Adoucir :", 0, 10, 0, "px")

        for sl in (self.threshold_slider, self.erode_slider, self.feather_slider):
            sl.valueChanged.connect(lambda _: self._refine_timer.start())

        self.refine_reset_btn = QPushButton("Retour a l'image sans finition")
        self.refine_reset_btn.setEnabled(False)
        self.refine_reset_btn.clicked.connect(self._reset_refine)
        self.refine_reset_btn.setStyleSheet(
            "QPushButton { background: #3a3a3a; color: #ddd; font-weight: bold; padding: 6px; border-radius: 4px; }"
            "QPushButton:disabled { background: #333; color: #666; }"
            "QPushButton:hover { background: #555; }"
        )

        edge_layout.addLayout(thr_row)
        edge_layout.addWidget(thr_hint)
        edge_layout.addLayout(ero_row)
        edge_layout.addLayout(fea_row)
        edge_layout.addWidget(self.refine_reset_btn)
        layout.addWidget(edge_group)

        # --- Pipette ---
        pip_group = QGroupBox("Pipette de couleurs")
        pip_layout = QVBoxLayout(pip_group)

        pip_hint = QLabel("Clique sur l'image originale (Avant)\npour echantillonner une couleur.")
        pip_hint.setStyleSheet("color: #888; font-size: 10px;")
        pip_layout.addWidget(pip_hint)

        pip_btn_row = QHBoxLayout()
        self.pip_exclude_btn = QPushButton("Exclure")
        self.pip_exclude_btn.setCheckable(True)
        self.pip_exclude_btn.setToolTip("Clique sur une couleur residuelle a supprimer")
        self.pip_protect_btn = QPushButton("Proteger")
        self.pip_protect_btn.setCheckable(True)
        self.pip_protect_btn.setToolTip("Clique sur une couleur du dessin a garder")
        for b in (self.pip_exclude_btn, self.pip_protect_btn):
            b.setStyleSheet(
                "QPushButton { background: #2a2a2a; border: 1px solid #444; border-radius: 4px; padding: 4px 8px; }"
                "QPushButton:checked { border: 2px solid #40916c; background: #1a3a2a; }"
                "QPushButton:hover { background: #3a3a3a; }"
            )
        self.pip_exclude_btn.clicked.connect(lambda: self._toggle_eyedropper("exclude"))
        self.pip_protect_btn.clicked.connect(lambda: self._toggle_eyedropper("protect"))
        pip_btn_row.addWidget(self.pip_exclude_btn)
        pip_btn_row.addWidget(self.pip_protect_btn)
        pip_layout.addLayout(pip_btn_row)

        pip_lists_row = QHBoxLayout()

        excl_col = QVBoxLayout()
        excl_col.addWidget(QLabel("A exclure :"))
        self.excl_container = QVBoxLayout()
        self.excl_container.setSpacing(2)
        excl_col.addLayout(self.excl_container)
        pip_lists_row.addLayout(excl_col)

        prot_col = QVBoxLayout()
        prot_col.addWidget(QLabel("A proteger :"))
        self.prot_container = QVBoxLayout()
        self.prot_container.setSpacing(2)
        prot_col.addLayout(self.prot_container)
        pip_lists_row.addLayout(prot_col)

        pip_layout.addLayout(pip_lists_row)

        pip_apply_row = QHBoxLayout()
        self.pip_apply_btn = QPushButton("Appliquer")
        self.pip_apply_btn.setEnabled(False)
        self.pip_apply_btn.clicked.connect(self._run_processing)
        self.pip_apply_btn.setStyleSheet(
            "QPushButton { background: #2d6a4f; color: white; font-weight: bold; padding: 5px; border-radius: 4px; }"
            "QPushButton:disabled { background: #333; color: #666; }"
            "QPushButton:hover { background: #40916c; }"
        )
        self.pip_clear_btn = QPushButton("Tout effacer")
        self.pip_clear_btn.clicked.connect(self._clear_color_hints)
        pip_apply_row.addWidget(self.pip_apply_btn)
        pip_apply_row.addWidget(self.pip_clear_btn)
        pip_layout.addLayout(pip_apply_row)

        layout.addWidget(pip_group)

        layout.addStretch()
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.preview = PreviewWidget()
        self.preview.before_panel.canvas.color_picked.connect(self._on_color_picked)
        layout.addWidget(self.preview, stretch=1)

        color_group = QGroupBox("Couleurs dominantes (fond exclu)")
        color_group.setMaximumHeight(220)
        color_inner = QVBoxLayout(color_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.color_container = QWidget()
        self.color_list_layout = QVBoxLayout(self.color_container)
        self.color_list_layout.setContentsMargins(0, 0, 0, 0)
        self.color_list_layout.setSpacing(2)
        self.color_list_layout.addStretch()
        scroll.setWidget(self.color_container)
        color_inner.addWidget(scroll)

        layout.addWidget(color_group, stretch=0)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(260)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # --- Replicate API Key ---
        api_group = QGroupBox("BRIA RMBG 2.0 — Replicate")
        api_layout = QVBoxLayout(api_group)

        api_info = QLabel("Cle API Replicate :")
        api_info.setStyleSheet("font-size: 11px; color: #aaa;")
        api_layout.addWidget(api_info)

        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("r8_xxxxxxxxxxxxxxxxxxxx")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setStyleSheet(
            "QLineEdit { background: #1e1e1e; border: 1px solid #444; border-radius: 4px; padding: 4px 6px; color: #ddd; }"
        )
        api_layout.addWidget(self.api_key_input)

        api_link = QLabel('<a href="https://replicate.com/account/api-tokens" style="color:#40916c;">Obtenir une cle gratuite</a>')
        api_link.setOpenExternalLinks(True)
        api_link.setStyleSheet("font-size: 10px;")
        api_layout.addWidget(api_link)

        layout.addWidget(api_group)

        # --- Export ---
        exp_group = QGroupBox("Export")
        exp_layout = QVBoxLayout(exp_group)

        exp_layout.addWidget(QLabel("Taille predéfinie :"))
        self.size_combo = QComboBox()
        self.size_combo.addItem("Taille originale", None)
        for name, dims in STANDARD_SIZES_CM.items():
            self.size_combo.addItem(name, dims)
        exp_layout.addWidget(self.size_combo)

        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("DPI :"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSingleStep(50)
        dpi_row.addWidget(self.dpi_spin)
        exp_layout.addLayout(dpi_row)

        self.export_btn = QPushButton("Exporter PNG...")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export)
        self.export_btn.setStyleSheet(
            "QPushButton { background: #1d4e89; color: white; font-weight: bold; padding: 6px; border-radius: 4px; }"
            "QPushButton:disabled { background: #333; color: #666; }"
            "QPushButton:hover { background: #2e6bba; }"
        )
        exp_layout.addWidget(self.export_btn)

        self.export_status = QLabel("")
        self.export_status.setWordWrap(True)
        self.export_status.setStyleSheet("color: #6bcb77; font-size: 11px;")
        exp_layout.addWidget(self.export_status)

        layout.addWidget(exp_group)

        # --- Image info ---
        info_group = QGroupBox("Informations image")
        self.info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("—")
        self.info_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self.info_label.setWordWrap(True)
        self.info_layout.addWidget(self.info_label)
        layout.addWidget(info_group)

        # --- Apercu fond colore ---
        preview_group = QGroupBox("Apercu fond colore")
        preview_layout = QVBoxLayout(preview_group)

        preview_hint = QLabel("Visualise le resultat sur une couleur\navant d'exporter.")
        preview_hint.setStyleSheet("color: #888; font-size: 10px;")
        preview_layout.addWidget(preview_hint)

        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Fond :"))
        self.preview_bg_combo = QComboBox()
        self.preview_bg_combo.addItem("Transparent", None)
        self.preview_bg_combo.addItem("Blanc", (255, 255, 255))
        self.preview_bg_combo.addItem("Noir", (0, 0, 0))
        self.preview_bg_combo.addItem("Rouge", (220, 50, 50))
        self.preview_bg_combo.addItem("Bleu marine", (20, 40, 100))
        self.preview_bg_combo.addItem("Gris", (160, 160, 160))
        self.preview_bg_combo.currentIndexChanged.connect(self._on_preview_bg_changed)
        bg_row.addWidget(self.preview_bg_combo)
        preview_layout.addLayout(bg_row)

        layout.addWidget(preview_group)

        layout.addStretch()
        return panel

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Ouvrir une image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.psd *.psb)",
        )
        if not path:
            return
        self._load_image(Path(path))

    def _load_image(self, path: Path):
        try:
            if path.suffix.lower() in (".psd", ".psb"):
                from psd_tools import PSDImage
                psd = PSDImage.open(str(path))
                image = psd.composite()
            else:
                image = Image.open(str(path))

            self.original_image = image.convert("RGBA").copy()
            self.processed_image = None
            self.file_label.setText(path.name)
            self.preview.set_before(self.original_image)
            self.preview.set_after(None)
            self.process_btn.setEnabled(True)
            self.export_btn.setEnabled(False)
            self.export_status.setText("")
            self._update_info(self.original_image)
            self._refresh_colors(self.original_image)
        except Exception as e:
            self.file_label.setText(f"Erreur : {e}")

    def _pick_bg_color(self):
        if self.original_image:
            r, g, b = get_background_color(self.original_image)
            initial = QColor(r, g, b)
        else:
            initial = QColor(255, 255, 255)

        color = QColorDialog.getColor(initial, self, "Choisir la couleur de fond")
        if color.isValid():
            self._manual_bg_color = (color.red(), color.green(), color.blue())
            self.bg_color_preview.setStyleSheet(
                f"background: rgb({color.red()},{color.green()},{color.blue()}); border: 1px solid #555;"
            )

    def _clear_bg_color(self):
        self._manual_bg_color = None
        self.bg_color_preview.setStyleSheet("background: transparent; border: 1px solid #555;")

    def _on_mode_changed(self):
        mode = self.mode_combo.currentData()
        self.classic_widget.setVisible(mode == "classic")
        self.ai_hint.setVisible(mode == "ai")
        self.bria_hint.setVisible(mode == "bria")

    def _run_processing(self):
        if self.original_image is None:
            return
        mode = self.mode_combo.currentData()

        if mode == "bria":
            api_token = self.api_key_input.text().strip()
            if not api_token:
                self.export_status.setText("Cle Replicate manquante dans le panneau droit.")
                self.export_status.setStyleSheet("color: #e63946; font-size: 11px;")
                return
        else:
            api_token = ""

        self.process_btn.setEnabled(False)
        if mode == "ai":
            self.process_btn.setText("Analyse IA locale...")
        elif mode == "bria":
            self.process_btn.setText("BRIA en cours...")
        else:
            self.process_btn.setText("Traitement...")

        self._worker = ProcessingWorker(
            self.original_image.copy(),
            mode=mode,
            tolerance=self.tolerance_slider.value(),
            target_color=self._manual_bg_color,
            exclude_colors=list(self._exclude_colors),
            protect_colors=list(self._protect_colors),
            api_token=api_token,
        )
        self._worker.done.connect(self._on_processing_done)
        self._worker.error.connect(self._on_processing_error)
        self._worker.start()

    def _on_processing_done(self, result: Image.Image):
        self.processed_image = result.copy()
        self._refined_image = None
        self.preview.set_after(result, keep_zoom=False)
        self.export_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.process_btn.setText("Supprimer le fond")
        self.refine_reset_btn.setEnabled(True)
        self._update_info(result)
        self._refresh_colors(result)

    def _run_refine(self):
        if self.processed_image is None:
            return
        threshold = self.threshold_slider.value()
        erode = self.erode_slider.value()
        feather = self.feather_slider.value()
        self._refined_image = refine_edges(
            self.processed_image.copy(),
            threshold=threshold,
            erode=erode,
            feather=feather,
        )
        self.preview.set_after(self._refined_image, keep_zoom=True)
        self._refresh_colors(self._refined_image)

    def _reset_refine(self):
        if self.processed_image is None:
            return
        self._refined_image = None
        self._refine_timer.stop()
        self.threshold_slider.setValue(128)
        self.erode_slider.setValue(0)
        self.feather_slider.setValue(0)
        self.preview.set_after(self.processed_image, keep_zoom=True)
        self._refresh_colors(self.processed_image)

    # ------------------------------------------------------------------
    # Eyedropper / color hints
    # ------------------------------------------------------------------

    def _toggle_eyedropper(self, target: str):
        if self._eyedropper_target == target:
            self._eyedropper_target = None
            self.pip_exclude_btn.setChecked(False)
            self.pip_protect_btn.setChecked(False)
            self.preview.before_panel.canvas.set_eyedropper(False)
        else:
            self._eyedropper_target = target
            self.pip_exclude_btn.setChecked(target == "exclude")
            self.pip_protect_btn.setChecked(target == "protect")
            self.preview.before_panel.canvas.set_eyedropper(True)

    def _on_color_picked(self, r: int, g: int, b: int):
        if self._eyedropper_target == "exclude":
            self._exclude_colors.append((r, g, b))
            self._add_color_swatch(self.excl_container, (r, g, b), self._exclude_colors)
        elif self._eyedropper_target == "protect":
            self._protect_colors.append((r, g, b))
            self._add_color_swatch(self.prot_container, (r, g, b), self._protect_colors)
        self.pip_apply_btn.setEnabled(self.original_image is not None)

    def _add_color_swatch(self, container: "QVBoxLayout", color: tuple, color_list: list):
        from PyQt6.QtWidgets import QHBoxLayout
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        swatch = QLabel()
        swatch.setFixedSize(18, 18)
        swatch.setStyleSheet(f"background: rgb{color}; border: 1px solid #555; border-radius: 2px;")

        lbl = QLabel(f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}")
        lbl.setStyleSheet("font-size: 10px; color: #bbb;")

        del_btn = QPushButton("x")
        del_btn.setFixedSize(18, 18)
        del_btn.setStyleSheet("QPushButton { background: #444; border: none; color: #aaa; font-weight: bold; border-radius: 3px; } QPushButton:hover { background: #c0392b; color: white; }")
        del_btn.clicked.connect(lambda: self._remove_color(row, color, color_list, container))

        row_layout.addWidget(swatch)
        row_layout.addWidget(lbl)
        row_layout.addWidget(del_btn)
        container.addWidget(row)

    def _remove_color(self, widget, color, color_list, container):
        if color in color_list:
            color_list.remove(color)
        widget.setParent(None)
        widget.deleteLater()

    def _clear_color_hints(self):
        self._exclude_colors.clear()
        self._protect_colors.clear()
        for container in (self.excl_container, self.prot_container):
            while container.count():
                item = container.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        self.pip_apply_btn.setEnabled(False)
        self._toggle_eyedropper(self._eyedropper_target or "")

    def _on_preview_bg_changed(self):
        color = self.preview_bg_combo.currentData()
        self.preview.set_background_color(color)

    def _on_processing_error(self, msg: str):
        self.process_btn.setEnabled(True)
        self.process_btn.setText("Supprimer le fond")
        self.export_status.setText(f"Erreur : {msg}")
        self.export_status.setStyleSheet("color: #e63946; font-size: 11px;")

    def _export(self):
        if self.processed_image is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter", "", "PNG (*.png)"
        )
        if not path:
            return

        dpi = self.dpi_spin.value()
        size_data = self.size_combo.currentData()

        to_export = self._refined_image if self._refined_image is not None else self.processed_image

        if size_data is None:
            out = export_png(to_export, path, dpi=dpi)
        else:
            w_cm, h_cm = size_data
            out = export_resized(to_export, path, w_cm, h_cm, dpi=dpi)

        self.export_status.setText(f"Exporte : {out.name}")
        self.export_status.setStyleSheet("color: #6bcb77; font-size: 11px;")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_info(self, img: Image.Image):
        info = get_image_info(img, dpi=self.dpi_spin.value())
        self.info_label.setText(
            f"{info['width_px']} x {info['height_px']} px\n"
            f"{info['width_cm']} x {info['height_cm']} cm @ {info['dpi']} DPI\n"
            f"Mode : {info['mode']}"
        )

    def _refresh_colors(self, img: Image.Image):
        while self.color_list_layout.count() > 1:
            item = self.color_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        colors = get_dominant_colors(img, n_colors=8)
        for c in colors:
            swatch = ColorSwatch(c)
            self.color_list_layout.insertWidget(self.color_list_layout.count() - 1, swatch)

    # ------------------------------------------------------------------
    # Drag & drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            self._load_image(Path(urls[0].toLocalFile()))

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #121212;
                color: #e0e0e0;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #333;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                color: #bbb;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background: #2a2a2a;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px 10px;
                color: #ddd;
            }
            QPushButton:hover { background: #3a3a3a; }
            QSlider::groove:horizontal {
                height: 4px;
                background: #333;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #40916c;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background: #1e1e1e;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 3px 6px;
                color: #ddd;
            }
            QScrollArea { background: transparent; border: none; }
        """)
