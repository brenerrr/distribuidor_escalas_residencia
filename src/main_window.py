import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()  # Call the inherited classes __init__ method
        path = os.path.dirname(os.path.abspath(__file__))
        fullpath = os.path.join(path, "main_window.ui")
        uic.loadUi(fullpath, self)  # Load the .ui file
