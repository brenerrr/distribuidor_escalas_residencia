import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5 import uic, QtGui
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt


class StatusWindow(QWidget):
    def __init__(self):
        super(
            StatusWindow, self
        ).__init__()  # Call the inherited classes __init__ method
        if getattr(sys, "frozen", False):
            filepath = os.path.join(sys._MEIPASS, "./src/")
        else:
            filepath = os.path.dirname(os.path.abspath(__file__))
        fullpath = os.path.join(filepath, "status_window.ui")
        uic.loadUi(fullpath, self)  # Load the .ui file
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.closable = False

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if self.closable:
            return super().closeEvent(a0)
        else:
            a0.ignore()

    def allow_closing(self):
        self.closable = True

    def prevent_closing(self):
        self.closable = False
