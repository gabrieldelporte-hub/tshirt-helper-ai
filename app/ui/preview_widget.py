"""Side-by-side before/after image preview widget with zoom, pan & background color."""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout,
    QPushButton, QHBoxLayout,
)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QMouseEvent, QColor, QCursor
from PIL import Image


# ---------------------------------------------------------------------------
# PIL RGBA -> QPixmap
# ---------------------------------------------------------------------------

def pil_rgba_to_qpixmap(img: Image.Image) -> QPixmap:
    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


def pil_to_qpixmap_opaque(img: Image.Image) -> QPixmap:
    img = img.convert("RGB")
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Zoomable canvas
# ---------------------------------------------------------------------------

class ZoomableCanvas(QWidget):
    ZOOM_STEP = 1.15
    ZOOM_MIN = 0.05
    ZOOM_MAX = 20.0

    color_picked = pyqtSignal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("border: 1px solid #444; border-radius: 6px;")
        self.setMouseTracking(True)

        self._pixmap: QPixmap | None = None
        self._pil_image: Image.Image | None = None
        self._zoom: float = 1.0
        self._offset: QPoint = QPoint(0, 0)
        self._drag_start: QPoint | None = None
        self._drag_offset: QPoint = QPoint(0, 0)
        self._bg_color: QColor | None = None
        self._checker_pixmap: QPixmap | None = None
        self._eyedropper_mode: bool = False

    def set_pixmap(self, pixmap: QPixmap | None, pil_image: Image.Image | None = None, keep_zoom: bool = False):
        self._pixmap = pixmap
        self._pil_image = pil_image
        if pixmap is not None:
            self._checker_pixmap = self._make_checker_pixmap(pixmap.size())
            if not keep_zoom:
                self._fit_to_view()
        self.update()

    def set_eyedropper(self, enabled: bool):
        self._eyedropper_mode = enabled
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor) if enabled else QCursor(Qt.CursorShape.ArrowCursor))

    def set_background(self, color: QColor | None):
        self._bg_color = color
        self.update()

    def _fit_to_view(self):
        if self._pixmap is None:
            return
        w, h = self.width(), self.height()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw == 0 or ph == 0:
            return
        self._zoom = min(w / pw, h / ph) * 0.95
        self._offset = QPoint(
            int((w - pw * self._zoom) / 2),
            int((h - ph * self._zoom) / 2),
        )

    def reset_zoom(self):
        self._fit_to_view()
        self.update()

    @staticmethod
    def _make_checker_pixmap(size, tile: int = 12) -> QPixmap:
        from PyQt6.QtCore import QSize
        w, h = size.width(), size.height()
        pm = QPixmap(w, h)
        painter = QPainter(pm)
        light = QColor(200, 200, 200)
        dark = QColor(150, 150, 150)
        for row in range((h // tile) + 1):
            for col in range((w // tile) + 1):
                color = light if (row + col) % 2 == 0 else dark
                painter.fillRect(col * tile, row * tile, tile, tile, color)
        painter.end()
        return pm

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._pixmap is None:
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Aucune image")
            return

        pw = int(self._pixmap.width() * self._zoom)
        ph = int(self._pixmap.height() * self._zoom)
        x, y = self._offset.x(), self._offset.y()

        if self._bg_color is not None:
            painter.fillRect(x, y, pw, ph, self._bg_color)
        elif self._checker_pixmap is not None:
            painter.drawPixmap(x, y, pw, ph, self._checker_pixmap)

        painter.drawPixmap(x, y, pw, ph, self._pixmap)

        painter.setPen(QColor(120, 120, 120))
        painter.drawText(8, self.height() - 8, f"{self._zoom * 100:.0f}%")

    def resizeEvent(self, event):
        if self._pixmap is not None:
            self._fit_to_view()
        super().resizeEvent(event)

    # ------------------------------------------------------------------
    # Mouse wheel -> zoom
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent):
        if self._pixmap is None:
            return
        cursor = event.position().toPoint()
        factor = self.ZOOM_STEP if event.angleDelta().y() > 0 else 1 / self.ZOOM_STEP
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom * factor))
        ratio = new_zoom / self._zoom
        self._offset = QPoint(
            int(cursor.x() - ratio * (cursor.x() - self._offset.x())),
            int(cursor.y() - ratio * (cursor.y() - self._offset.y())),
        )
        self._zoom = new_zoom
        self.update()

    # ------------------------------------------------------------------
    # Click-drag -> pan
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._eyedropper_mode:
            self._pick_color(event.pos())
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.pos()
            self._drag_offset = QPoint(self._offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _pick_color(self, pos: QPoint):
        if self._pil_image is None or self._pixmap is None:
            return
        img_x = (pos.x() - self._offset.x()) / self._zoom
        img_y = (pos.y() - self._offset.y()) / self._zoom
        pw, ph = self._pil_image.size
        if 0 <= img_x < pw and 0 <= img_y < ph:
            pixel = self._pil_image.convert("RGB").getpixel((int(img_x), int(img_y)))
            self.color_picked.emit(pixel[0], pixel[1], pixel[2])

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_start is not None:
            delta = event.pos() - self._drag_start
            self._offset = self._drag_offset + delta
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self.reset_zoom()


# ---------------------------------------------------------------------------
# Panel (title + canvas)
# ---------------------------------------------------------------------------

class ImagePanel(QWidget):
    def __init__(self, title: str, show_bg_selector: bool = False, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; color: #aaa; font-size: 12px;")

        hint = QLabel("Molette : zoom  •  Clic-glisser : deplacer  •  Double-clic : ajuster")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #555; font-size: 10px;")

        self.canvas = ZoomableCanvas()

        layout.addWidget(title_label)
        layout.addWidget(hint)
        layout.addWidget(self.canvas)

    def set_image(self, img: Image.Image | None, keep_zoom: bool = False):
        if img is None:
            self.canvas.set_pixmap(None, None)
            return
        if img.mode == "RGBA":
            pixmap = pil_rgba_to_qpixmap(img)
        else:
            pixmap = pil_to_qpixmap_opaque(img)
        self.canvas.set_pixmap(pixmap, img, keep_zoom=keep_zoom)


# ---------------------------------------------------------------------------
# PreviewWidget
# ---------------------------------------------------------------------------

class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.before_panel = ImagePanel("Avant", show_bg_selector=False)
        self.after_panel = ImagePanel("Apres", show_bg_selector=False)

        layout.addWidget(self.before_panel)
        layout.addWidget(self.after_panel)

    def set_before(self, img: Image.Image | None):
        self.before_panel.set_image(img)

    def set_after(self, img: Image.Image | None, keep_zoom: bool = False):
        self.after_panel.set_image(img, keep_zoom=keep_zoom)

    def set_background_color(self, color: tuple | None):
        if color is None:
            self.after_panel.canvas.set_background(None)
        else:
            self.after_panel.canvas.set_background(QColor(*color))
