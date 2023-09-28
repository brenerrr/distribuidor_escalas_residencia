from PyQt5 import QtWidgets, uic, QtCore
import sys
import json
from collections import defaultdict
import pandas as pd
import os

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"


class Ui(QtWidgets.QMainWindow):
    def __init__(self, startup_values: defaultdict):
        super(Ui, self).__init__()  # Call the inherited classes __init__ method
        uic.loadUi("src\\manager.ui", self)  # Load the .ui file
        self.employees_areas = None
        self.startup_values = startup_values
        self.fill_widgets()
        self.setup_signals()

        # Stretch table columns
        self.format_tables_columns([self.table_areas, self.table_employees])

        # self.tabWidget.setCurrentIndex(1)
        # self.create_tab_areas_employees()

    def read_table(self, table):
        n_rows = table.rowCount()
        n_cols = table.columnCount()
        data = [table.item(r, c).text() for r in range(n_rows) for c in range(n_cols)]
        return pd.DataFrame(data)

    def create_tab_areas_employees(self, index):
        if index != 1:
            return
        layout: QtWidgets.QGridLayout = self.grid_employees_areas

        areas = self.read_table(self.table_areas)[0]
        areas = areas.str.split("_").str[0].unique()
        employees = self.read_table(self.table_employees)[0].values

        for i, area in enumerate(areas):
            layout.addWidget(QtWidgets.QLabel(area), i + 1, 0)

        for j, employee in enumerate(employees):
            layout.addWidget(QtWidgets.QLabel(employee), 0, j + 1)

        checks = defaultdict(lambda: dict())
        for i, area in enumerate(areas):
            layout.setRowStretch(i, 1)
            for j, employee in enumerate(employees):
                value = startup_values["employees_areas"].get(employee).get(area)
                check = QtWidgets.QCheckBox()

                check.setStyleSheet("QCheckBox::indicator{width: 20px; height: 20px;};")
                check.setChecked(value)

                layout.addWidget(check, i + 1, j + 1)
                layout.setColumnStretch(j, 1)
                layout.setAlignment(check, QtCore.Qt.AlignCenter)
                check.stateChanged.connect(self.export_startup_values)
                checks[employee][area] = check

        self.employees_areas = checks

        pass

    def format_tables_columns(self, tables):
        for table in tables:
            table.horizontalHeader().setSectionResizeMode(
                0, QtWidgets.QHeaderView.Stretch
            )

    def setup_signals(self):
        # Tab
        tab: QtWidgets.QTabWidget = self.tabWidget
        tab.currentChanged.connect(self.create_tab_areas_employees)

        # Tab Inputs
        # Add area button
        self.button_add_area.clicked.connect(self.add_area)
        self.button_add_employee.clicked.connect(self.add_employee)

        # Areas table
        table: QtWidgets.QTableWidget = self.table_areas
        table.doubleClicked.connect(self.remove_area)
        table.itemChanged.connect(self.export_startup_values)
        table.setToolTip("Dois clicks para deletar")
        self.input_area.installEventFilter(self)
        self.input_subarea.installEventFilter(self)

        # Employees table
        table: QtWidgets.QTableWidget = self.table_employees
        table.doubleClicked.connect(self.remove_area)
        table.itemChanged.connect(self.export_startup_values)
        table.setToolTip("Dois clicks para deletar")
        self.input_employee.installEventFilter(self)

    def eventFilter(self, obj, event):
        def pressed_Enter(widget):
            if event.type() == QtCore.QEvent.KeyPress and obj is widget:
                if (
                    (event.key() == QtCore.Qt.Key.Key_Return)
                    or (event.key() == QtCore.Qt.Key.Key_Enter)
                ) and widget.hasFocus():
                    return True

        # In case enter is pressed on input_area text edit
        if pressed_Enter(self.input_area) or pressed_Enter(self.input_subarea):
            self.add_area()
            return True

        # In case enter is pressed on input_employee text edit
        if pressed_Enter(self.input_employee):
            self.add_employee()
            return True

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

    def add_area(self):
        input_area: QtWidgets.QTextEdit = self.input_area
        input_subarea: QtWidgets.QTextEdit = self.input_subarea
        area = input_area.toPlainText().replace("\n", "")
        subarea = input_subarea.toPlainText().replace("\n", "")
        if subarea:
            area = area + "_" + subarea
        if area:
            table: QtWidgets.QTableWidget = self.table_areas
            value = QtWidgets.QTableWidgetItem(area)
            self.add_row(table, value)
        self.input_area.clear()
        self.input_subarea.clear()

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

        # Table that relates employees and areas
        if self.employees_areas is not None:
            d = defaultdict(lambda: dict())
            for employee, checks in self.employees_areas.items():
                for area, check in checks.items():
                    d[employee][area] = check.isChecked()
            export_dict["employees_areas"] = d
        elif startup_values.get("employee_areas") is not None:
            export_dict["employees_areas"] = startup_values["employees_areas"]

        with open("startup.json", "w") as f:
            json_object = json.dumps(export_dict, indent=4)
            f.write(json_object)


startup_values = defaultdict(lambda: "")
with open("startup.json", "r") as f:
    startup_values.update(json.load(f))

app = QtWidgets.QApplication(sys.argv)  # Create an instance of QtWidgets.QApplication
window = Ui(startup_values)  # Create an instance of our class
window.show()
app.exec()  # Start the application
