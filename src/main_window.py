import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()  # Call the inherited classes __init__ method
        if getattr(sys, "frozen", False):
            filepath = sys._MEIPASS
        else:
            filepath = os.path.dirname(os.path.abspath(__file__))
        print(filepath)
        filepath = os.path.dirname(os.path.abspath(__file__))
        fullpath = os.path.join(filepath, "main_window.ui")
        uic.loadUi(fullpath, self)  # Load the .ui file
