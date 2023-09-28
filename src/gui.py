from PyQt5 import QtWidgets, uic, QtCore
import sys
import json
from collections import defaultdict

import os

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"


class Ui(QtWidgets.QMainWindow):
    def __init__(self, startup_values: defaultdict):
        super(Ui, self).__init__()  # Call the inherited classes __init__ method
        uic.loadUi("src\\manager.ui", self)  # Load the .ui file
        self.startup_values = startup_values
        self.fill_widgets()
        self.setup_signals()

    def setup_signals(self):
        # Tab Areas
        # Add area button
        self.button_add_area.clicked.connect(self.add_area)
        self.button_add_employee.clicked.connect(self.add_employee)

        # Areas table
        table: QtWidgets.QTableWidget = self.table_areas
        table.doubleClicked.connect(self.remove_area)
        table.itemChanged.connect(self.export_startup_values)
        table.setToolTip("Dois clicks para deletar")
        self.input_area.installEventFilter(self)

        # Employees table
        table: QtWidgets.QTableWidget = self.table_employees
        table.doubleClicked.connect(self.remove_area)
        table.itemChanged.connect(self.export_startup_values)
        table.setToolTip("Dois clicks para deletar")
        self.input_employee.installEventFilter(self)

    def eventFilter(self, obj, event):
        # In case enter is pressed on input_area text edit
        if event.type() == QtCore.QEvent.KeyPress and obj is self.input_area:
            if (
                (event.key() == QtCore.Qt.Key.Key_Return)
                or (event.key() == QtCore.Qt.Key.Key_Enter)
            ) and self.input_area.hasFocus():
                self.add_area()

        # In case enter is pressed on input_employee text edit
        if event.type() == QtCore.QEvent.KeyPress and obj is self.input_employee:
            if (
                (event.key() == QtCore.Qt.Key.Key_Return)
                or (event.key() == QtCore.Qt.Key.Key_Enter)
            ) and self.input_employee.hasFocus():
                self.add_employee()

        return super().eventFilter(obj, event)

    def fill_widgets(self):
        # Tab areas
        tables = dict(
            areas=dict(values=self.startup_values["areas"], obj=self.table_areas),
            employees=dict(
                values=self.startup_values["employees"], obj=self.table_employees
            ),
        )
        for items in tables.values():
            table = items["obj"]
            values = items["values"]
            for i, value in enumerate(values):
                table.insertRow(i)
                table.setItem(i, 0, QtWidgets.QTableWidgetItem(value))

    def add_row(self, table: QtWidgets.QTableWidget, value: QtWidgets.QTableWidgetItem):
        row_position = table.rowCount()
        table.insertRow(row_position)
        table.setItem(row_position, 0, value)
        self.input_area.clear()

    def add_area(self):
        input_area: QtWidgets.QTextEdit = self.input_area
        area = input_area.toPlainText().replace("\n", "")

        if area:
            table: QtWidgets.QTableWidget = self.table_areas
            value = QtWidgets.QTableWidgetItem(area)
            self.add_row(table, value)
            self.input_area.clear()

    def add_employee(self):
        input_employee: QtWidgets.QTextEdit = self.input_employee
        employee = input_employee.toPlainText().replace("\n", "")

        if employee:
            table: QtWidgets.QTableWidget = self.table_employees
            value = QtWidgets.QTableWidgetItem(employee)
            self.add_row(table, value)
            self.input_employee.clear()

    def remove_area(self, row):
        table: QtWidgets.QTableWidget = self.sender()
        table.removeRow(row.row())
        self.export_startup_values()

    def export_startup_values(self):
        export_dict = {}
        with open("startup.json", "w") as f:
            # Areas table
            table: QtWidgets.QTableWidget = self.table_areas
            export_dict["areas"] = [
                table.item(i, 0).text() for i in range(table.rowCount())
            ]

            # Employees table
            table: QtWidgets.QTableWidget = self.table_employees
            export_dict["employees"] = [
                table.item(i, 0).text() for i in range(table.rowCount())
            ]
            json_object = json.dumps(export_dict, indent=4)
            f.write(json_object)


startup_values = defaultdict(lambda: "")
with open("startup.json", "r") as f:
    startup_values.update(json.load(f))

app = QtWidgets.QApplication(sys.argv)  # Create an instance of QtWidgets.QApplication
window = Ui(startup_values)  # Create an instance of our class
window.show()
app.exec()  # Start the application
