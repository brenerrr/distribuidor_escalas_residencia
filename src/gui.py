from PyQt5 import uic
from PyQt5.QtCore import Qt, QEvent, QDate
from PyQt5.QtWidgets import (
    QMainWindow,
    QGridLayout,
    QLabel,
    QCheckBox,
    QHeaderView,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QTextEdit,
    QDateEdit,
    QApplication,
)
from datetime import datetime
import sys
import json
from collections import defaultdict
import pandas as pd
import os

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"


class Ui(QMainWindow):
    def __init__(self, startup_values: defaultdict):
        super(Ui, self).__init__()  # Call the inherited classes __init__ method
        uic.loadUi("src\\manager.ui", self)  # Load the .ui file
        self.initialize_variables()
        self.startup_values = startup_values
        self.populate_widgets_startup()
        self.setup_signals()
        self.format_tables_columns(
            [self.table_areas, self.table_employees, self.table_shiftException]
        )

    @property
    def areas(self):
        return self.read_table(self.table_areas)[0]

    @property
    def employees(self):
        return self.read_table(self.table_employees)[0]

    def initialize_variables(self):
        self.employees_areas = None
        self.selected_dates = defaultdict(lambda: list())
        self.table_shifts_header = [
            "Nome",
            "Período",
            "Área",
            "Qtd. Pessoas",
            "Dias",
            "Obrigatório",
            "Balancear",
            "Duração",
            "Exceções",
        ]

    def read_table(self, table):
        n_rows = table.rowCount()
        n_cols = table.columnCount()
        data = [table.item(r, c).text() for r in range(n_rows) for c in range(n_cols)]
        return pd.DataFrame(data)

    def create_tab_areas_employees(
        self,
    ):
        layout: QGridLayout = self.grid_employees_areas

        areas = self.areas.str.split("_").str[0].unique()
        employees = self.employees.values

        for i, area in enumerate(areas):
            layout.addWidget(QLabel(area), i + 1, 0)

        for j, employee in enumerate(employees):
            layout.addWidget(QLabel(employee), 0, j + 1)

        checks = defaultdict(lambda: dict())
        for i, area in enumerate(areas):
            layout.setRowStretch(i, 1)
            for j, employee in enumerate(employees):
                check = QCheckBox()
                check.setStyleSheet("QCheckBox::indicator{width: 20px; height: 20px;};")
                value = (
                    self.startup_values.get("employees_areas", {})
                    .get(employee, {})
                    .get(area)
                )
                if value:
                    check.setChecked(value)

                layout.addWidget(check, i + 1, j + 1)
                layout.setColumnStretch(j, 1)
                layout.setAlignment(check, Qt.AlignCenter)
                check.stateChanged.connect(self.export_startup_values)
                checks[employee][area] = check

        self.employees_areas = checks

        pass

    def format_tables_columns(self, tables):
        for table in tables:
            n_columns = table.columnCount()
            for i in range(n_columns):
                if i == n_columns - 1:
                    policy = QHeaderView.Stretch
                else:
                    policy = QHeaderView.ResizeToContents
                table.horizontalHeader().setSectionResizeMode(i, policy)

    def setup_signals(self):
        # Tab
        tab: QTabWidget = self.tabWidget

        def tab_fn(index):
            d = {
                0: lambda: None,
                1: self.create_tab_areas_employees,
                2: self.populate_tab_shifts_widgets,
            }
            fn = d.get(index)
            fn()

        tab.currentChanged.connect(tab_fn)

        # Tab Inputs
        # Add area button
        self.button_addArea.clicked.connect(self.add_area)
        self.button_addEmployee.clicked.connect(self.add_employee)
        self.drop_month.currentTextChanged.connect(self.export_startup_values)

        # Areas table
        table: QTableWidget = self.table_areas
        table.doubleClicked.connect(self.remove_area)
        table.itemChanged.connect(self.export_startup_values)
        table.setToolTip("Dois clicks para deletar")
        self.input_area.installEventFilter(self)
        self.input_subarea.installEventFilter(self)

        # Employees table
        table: QTableWidget = self.table_employees
        table.doubleClicked.connect(self.remove_area)
        table.itemChanged.connect(self.export_startup_values)
        table.setToolTip("Dois clicks para deletar")
        self.input_employee.installEventFilter(self)

        # Tab Shifts
        self.button_addShift.clicked.connect(self.add_shift)
        self.button_addShiftDate.clicked.connect(self.add_shift_exception)
        table: QTableWidget = self.table_shifts
        table.doubleClicked.connect(self.remove_area)
        table.itemChanged.connect(self.export_startup_values)
        table.setToolTip("Dois clicks para deletar")

    def eventFilter(self, obj, event):
        def pressed_Enter(widget):
            if event.type() == QEvent.KeyPress and obj is widget:
                if (
                    (event.key() == Qt.Key.Key_Return)
                    or (event.key() == Qt.Key.Key_Enter)
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

    def populate_widgets_startup(self):
        # Tables
        tables = dict(
            areas=dict(values=self.startup_values["areas"], obj=self.table_areas),
            employees=dict(
                values=self.startup_values["employees"], obj=self.table_employees
            ),
            shifts=dict(
                values=[
                    [shift[header] for header in self.table_shifts_header]
                    for shift in self.startup_values["shifts"]
                ],
                obj=self.table_shifts,
            ),
        )
        for items in tables.values():
            table = items["obj"]
            values = items["values"]
            self.add_item(table, values)

        # Month dropdown
        comboBox: QComboBox = self.drop_month
        for month in [
            "Jan",
            "Fev",
            "Mar",
            "Abr",
            "Mai",
            "Jun",
            "Jul",
            "Ago",
            "Set",
            "Out",
            "Nov",
            "Dez",
        ]:
            comboBox.addItem(month)
        comboBox.setCurrentText(startup_values.get("month", "Jan"))

        # Tab employees areas
        self.create_tab_areas_employees()

        # Tab shifts
        comboBox: QComboBox = self.drop_shiftPeriod
        comboBox.addItems(["Manhã", "Tarde", "Noite"])

        table: QTableWidget = self.table_shifts
        table.setHorizontalHeaderLabels(self.table_shifts_header)

        dateEdit: QDateEdit = self.shiftExceptionDate
        month = self.drop_month.currentIndex() + 1
        dateEdit.setDate(QDate(datetime.now().year, month, 1))

    def add_item(self, table: QTableWidget, rows: list):
        for row in rows:
            n_rows = table.rowCount()
            table.insertRow(n_rows)

            if type(row) != list:
                row = [row]

            for j, value in enumerate(row):
                table.setItem(n_rows, j, QTableWidgetItem(value))

    def add_area(self):
        input_area: QTextEdit = self.input_area
        input_subarea: QTextEdit = self.input_subarea
        area = input_area.toPlainText().replace("\n", "")
        subarea = input_subarea.toPlainText().replace("\n", "")
        if subarea:
            area = area + "_" + subarea
        if area:
            table: QTableWidget = self.table_areas
            value = QTableWidgetItem(area)
            self.add_item(table, [value])
        self.input_area.clear()
        self.input_subarea.clear()

    def add_employee(self):
        input_employee: QTextEdit = self.input_employee
        employee = input_employee.toPlainText().replace("\n", "")

        if employee:
            table: QTableWidget = self.table_employees
            value = QTableWidgetItem(employee)
            self.add_item(table, [value])
        self.input_employee.clear()

    def remove_area(self, row):
        table: QTableWidget = self.sender()
        table.removeRow(row.row())
        self.export_startup_values()

    def export_startup_values(self):
        export_dict = {}

        comboBox: QComboBox = self.drop_month
        export_dict["month"] = comboBox.currentText()

        # Areas table
        table: QTableWidget = self.table_areas
        export_dict["areas"] = [
            table.item(i, 0).text() for i in range(table.rowCount())
        ]

        # Employees table
        table: QTableWidget = self.table_employees
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
        elif self.startup_values.get("employee_areas") is not None:
            export_dict["employees_areas"] = self.startup_values["employees_areas"]

        # Shifts table
        table: QTableWidget = self.table_shifts
        n_rows = table.rowCount()
        export_dict["shifts"] = []
        for i in range(n_rows):
            data = {}
            for j, header in enumerate(self.table_shifts_header):
                value = table.item(i, j)
                if value is None:
                    value = ""
                else:
                    value = value.text()
                data[header] = value
            export_dict["shifts"].append(data)

        with open("startup.json", "w") as f:
            json_object = json.dumps(export_dict, indent=4)
            f.write(json_object)

    def populate_tab_shifts_widgets(self):
        areas = self.areas.values
        for area in areas:
            drop: QComboBox = self.drop_shiftArea
            drop.addItem(area)
        pass

    def add_shift(self):
        data = {}
        data["Nome"] = self.input_shiftName.toPlainText()
        data["Período"] = self.drop_shiftPeriod.currentText()
        data["Área"] = self.drop_shiftArea.currentText()
        data["Qtd. Pessoas"] = str(self.sb_shiftNEmp.value())

        days_of_week = []
        for i in range(7):
            value = getattr(self, f"cb_shiftDay_{i}").isChecked()
            if value:
                label = getattr(self, f"label_dayOfWeek_{i}").text()
                days_of_week.append(label)
        data["Dias"] = ", ".join(days_of_week)

        data["Balancear"] = "Sim" if self.cb_shiftBalance.isChecked() else None
        data["Obrigatório"] = "Sim" if self.cb_shiftMandatory.isChecked() else None
        data["Duração"] = str(self.sb_shiftDuration.value())
        data["Exceções"] = self.read_table(self.table_shiftException).get(0)
        if data["Exceções"] is not None:
            l = data["Exceções"].astype(str).tolist()
            data["Exceções"] = ", ".join(l)

        table: QTableWidget = self.table_shifts
        i = table.rowCount()
        table.insertRow(i)
        for j, key in enumerate(self.table_shifts_header):
            table.setItem(i, j, QTableWidgetItem(data[key]))

    def add_shift_exception(self):
        dateEdit: QDateEdit = self.shiftExceptionDate
        year, month, day = dateEdit.date().getDate()
        string = f"{day:02d}/{month:02d}/{year}"
        string = QTableWidgetItem(string)
        self.add_item(self.table_shiftException, [string])


with open("startup.json", "r") as f:
    startup_values = json.load(f)

app = QApplication(sys.argv)  # Create an instance of Application
window = Ui(startup_values)  # Create an instance of our class
window.show()
app.exec()  # Start the application
