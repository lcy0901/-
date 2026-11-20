import sys
import threading
import numpy as np
from PySide6.QtCore import Qt, QRect, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QPixmap, QImage
from PySide6.QtWidgets import QApplication, QWidget

# ---------- OCR 在后台线程加载，不阻塞界面 ----------
_ocr = None
_ocr_ready = threading.Event()
_ocr_error = None

def _load_ocr():
    global _ocr, _ocr_error
    try:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    except Exception as e:
        _ocr_error = e
    finally:
        _ocr_ready.set()

threading.Thread(target=_load_ocr, daemon=True).start()


class SnipOverlay(QWidget):
    def __init__(self, shot: QPixmap, on_done):
        super().__init__()
        self.shot = shot
        self.on_done = on_done
        self.start = None
        self.end = None
        self.finished = False

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setGeometry(QApplication.primaryScreen().geometry())
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        # 预先算好压暗后的背景，之后每次重绘直接用它
        self.dimmed = self.shot.copy()
        _p = QPainter(self.dimmed)
        _p.fillRect(self.shot.rect(), QColor(0, 0, 0, 90))
        _p.end()

    def paintEvent(self, event):
        p = QPainter(self)
        p.drawPixmap(0, 0, self.dimmed)  # 直接画预先压暗的背景，不再每次 fillRect
        if self.start and self.end:
            r = QRect(self.start, self.end).normalized()
            p.drawPixmap(r, self.shot, self._scaled_rect(r))  # 选区内画原图（亮的）
            p.setPen(QPen(QColor(0, 160, 255), 2))
            p.drawRect(r)
        p.end()

    def _scaled_rect(self, r: QRect) -> QRect:
        d = self.shot.devicePixelRatio()
        return QRect(int(r.x() * d), int(r.y() * d), int(r.width() * d), int(r.height() * d))

    def _finish(self, pixmap):
        if self.finished:
            return
        self.finished = True
        self.close()
        self.on_done(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:   # 右键取消
            self._finish(None)
            return
        if event.button() == Qt.LeftButton:
            self.start = event.position().toPoint()
            self.end = self.start

    def mouseMoveEvent(self, event):
        if self.start:
            old = QRect(self.start, self.end).normalized()
            self.end = event.position().toPoint()
            new = QRect(self.start, self.end).normalized()
            # 只刷新新旧选区的并集，而不是整个窗口
            self.update(old.united(new).adjusted(-4, -4, 4, 4))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start:
            rect = QRect(self.start, self.end).normalized()
            if rect.width() > 5 and rect.height() > 5:
                self._finish(self.shot.copy(self._scaled_rect(rect)))
            else:
                self._finish(None)


def pixmap_to_bgr(pixmap: QPixmap) -> np.ndarray:
    img = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    buf = np.frombuffer(bytes(img.constBits()), dtype=np.uint8, count=bpl * h)
    arr = buf.reshape(h, bpl)[:, : w * 3].reshape(h, w, 3)
    return np.ascontiguousarray(arr[:, :, ::-1])   # RGB -> BGR

def run_ocr(pixmap: QPixmap) -> str:
    if not _ocr_ready.wait(timeout=60):
        raise RuntimeError("OCR 模型加载超时")
    if _ocr_error is not None:
        raise _ocr_error

    arr = pixmap_to_bgr(pixmap)       # 这里调用，替换原来的 QBuffer/PNG/PIL 那一串
    result, _ = _ocr(arr)
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    shot = app.primaryScreen().grabWindow(0)

    def on_done(pixmap):
        if pixmap is None:
            print("已取消")
            app.quit()
            return
        try:
            text = run_ocr(pixmap)
        except Exception as e:
            print("OCR 出错:", repr(e))
            app.quit()
            return

        if text:
            app.clipboard().setText(text)
            print("已复制到剪贴板：\n" + text)
        else:
            print("未识别到文字")

        # 给剪贴板一点时间被系统接管后再退出（主要针对 Linux）
        QTimer.singleShot(300, app.quit)

    overlay = SnipOverlay(shot, on_done)
    overlay.showFullScreen()
    overlay.activateWindow()
    overlay.raise_()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()