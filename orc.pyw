import sys
import io
import threading
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QRect, QBuffer, QIODevice, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QPixmap
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

    def paintEvent(self, event):
        p = QPainter(self)
        p.drawPixmap(0, 0, self.shot)
        p.fillRect(self.rect(), QColor(0, 0, 0, 90))
        if self.start and self.end:
            r = QRect(self.start, self.end).normalized()
            p.drawPixmap(r, self.shot, self._scaled_rect(r))
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
            self.end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start:
            rect = QRect(self.start, self.end).normalized()
            if rect.width() > 5 and rect.height() > 5:
                self._finish(self.shot.copy(self._scaled_rect(rect)))
            else:
                self._finish(None)


def run_ocr(pixmap: QPixmap) -> str:
    if not _ocr_ready.wait(timeout=60):
        raise RuntimeError("OCR 模型加载超时")
    if _ocr_error is not None:
        raise _ocr_error

    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    pixmap.save(buf, "PNG")
    img = Image.open(io.BytesIO(bytes(buf.data()))).convert("RGB")
    arr = np.ascontiguousarray(np.array(img)[:, :, ::-1])  # RGB -> BGR，并保证内存连续
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