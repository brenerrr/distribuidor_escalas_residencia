import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtCore import pyqtSignal, QThread
from manager import Manager


class WorkerThread(QThread):
    finished = pyqtSignal()
    status = pyqtSignal(str)
    error = pyqtSignal(int)

    def run(self):
        try:
            manager = Manager("inputs.json")
            self.status.emit(
                "Checando se uma solução com os atuais inputs é possível..."
            )
            manager.create_solution()

            if self.check_interruptions():
                return

            self.status.emit("Criando diversas soluções...")
            solutions = manager.create_solutions(1)
            scores = manager.calculate_scores(solutions)
            i = scores.argmax()
            best = dict(score=scores[i], solution=solutions.loc[:, i])
            self.status.emit("Encontrando fins de semana de folga para todos...")
            manager.create_schedule(best)
            manager.export_results(os.path.dirname(sys.argv[0]))
            self.error.emit(1)
            self.finished.emit()
        except RuntimeError as e:
            self.error.emit(-1)
            self.finished.emit()

    def check_interruptions(self):
        if self.isInterruptionRequested():
            self.finished.emit()
            return True
