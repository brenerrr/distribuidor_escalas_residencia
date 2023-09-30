from PyQt5 import uic
from PyQt5.QtCore import Qt, QEvent, QDate, QCoreApplication
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
    QSpinBox,
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

        self.startup_values = defaultdict(lambda: defaultdict(lambda: {}))
        self.startup_values.update(startup_values)
        self.initialize_variables()
        self.populate_widgets_startup()
        self.setup_signals()
        self.format_tables_columns(
            [
                self.table_areas,
                self.table_employees,
                self.table_shiftException,
                self.table_timeoff,
                self.table_restrictions,
            ]
        )

    @property
    def all_areas(self):
        return self.read_table(self.table_areas)[0].values

    @property
    def areas(self):
        areas = self.read_table(self.table_areas)[0]
        return areas.str.split("_").str[0].unique()

    @property
    def employees(self):
        return self.read_table(self.table_employees)[0]

    def initialize_variables(self):
        self.employees_areas = {}
        self.employees_areas.update(self.startup_values["employees_areas"])
        self.selected_dates = defaultdict(lambda: list())
        self.header_shifts = [
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
        self.header_timeoff = ["Nome", "Data", "Período"]
        self.header_restrictions = ["Área", "Qtd. Máxima de Turnos"]
        self.checks_emp_areas = {}
        self.widgets_emp_areas = []

    def read_table(self, table):
        n_rows = table.rowCount()
        n_cols = table.columnCount()
        data = [table.item(r, c).text() for r in range(n_rows) for c in range(n_cols)]
        return pd.DataFrame(data)

    def create_tab_areas_employees(
        self,
    ):
        layout: QGridLayout = self.grid_employees_areas

        areas = self.areas
        employees = self.employees

        # Clean previously created widgets
        for i in range(len(self.widgets_emp_areas)):
            widget = self.widgets_emp_areas.pop()
            widget.deleteLater()

        for i, area in enumerate(areas):
            label = QLabel(area)
            layout.addWidget(label, i + 1, 0)
            self.widgets_emp_areas.append(label)

        for j, employee in enumerate(employees):
            label = QLabel(employee)
            layout.addWidget(label, 0, j + 1)
            self.widgets_emp_areas.append(label)

        checks = defaultdict(lambda: defaultdict())
        for i, area in enumerate(areas):
            layout.setRowStretch(i, 1)
            for j, employee in enumerate(employees):
                check = QCheckBox()
                check.setStyleSheet("QCheckBox::indicator{width: 20px; height: 20px;};")
                value = self.employees_areas.get(employee, {}).get(area, False)
                check.setChecked(value)
                self.widgets_emp_areas.append(check)
                checks[employee][area] = check

                layout.addWidget(check, i + 1, j + 1)
                layout.setColumnStretch(j, 1)
                layout.setAlignment(check, Qt.AlignCenter)
                check.stateChanged.connect(self.update_tab_areas_employees)
                check.stateChanged.connect(self.export_startup_values)

        self.checks_emp_areas = checks

    def update_tab_areas_employees(self):
        d = defaultdict(lambda: {})
        for e in self.employees:
            for a in self.areas:
                d[e][a] = self.checks_emp_areas[e][a].isChecked()
        self.employees_areas = d

    def format_tables_columns(self, tables):
        for table in tables:
            n_columns = table.columnCount()
            for i in range(n_columns):
                if i == n_columns - 1:
                    policy = QHeaderView.Stretch
                else:
                    policy = QHeaderView.Stretch
                    # policy = QHeaderView.ResizeToContents
                table.horizontalHeader().setSectionResizeMode(i, policy)

    def setup_signals(self):
        # Tab
        tab: QTabWidget = self.tabWidget

        def tab_fn(index):
            d = {
                0: lambda: None,
                1: self.update_tab_areas_employees,
                2: lambda: None,
                3: lambda: None,
            }
            fn = d.get(index)
            fn()

        tab.currentChanged.connect(tab_fn)

        # Press buttons
        self.button_addArea.clicked.connect(self.add_area)
        self.button_addEmployee.clicked.connect(self.add_employee)
        self.button_addRestriction.clicked.connect(self.add_restriction)
        self.button_addTimeoff.clicked.connect(self.add_timeoff)
        self.button_addShift.clicked.connect(self.add_shift)
        self.button_addShiftDate.clicked.connect(self.add_shift_exception)

        self.drop_month.currentTextChanged.connect(self.export_startup_values)

        # Event filters
        self.input_area.installEventFilter(self)
        self.input_subarea.installEventFilter(self)
        self.input_employee.installEventFilter(self)

        # Tables
        self.assign_table_signals(self.table_areas)
        self.assign_table_signals(self.table_employees)
        self.assign_table_signals(self.table_shifts)
        self.assign_table_signals(self.table_timeoff)
        self.assign_table_signals(self.table_restrictions)
        self.assign_table_signals(self.table_shiftException)

        self.table_employees.itemChanged.connect(self.update_dropEmp)
        self.table_areas.itemChanged.connect(self.update_dropAreas)
        self.table_areas.itemChanged.connect(self.create_tab_areas_employees)
        self.table_employees.itemChanged.connect(self.create_tab_areas_employees)

        self.table_areas.doubleClicked.connect(self.update_dropAreas)
        self.table_areas.doubleClicked.connect(self.create_tab_areas_employees)
        self.table_employees.doubleClicked.connect(self.update_dropEmp)
        self.table_employees.doubleClicked.connect(self.create_tab_areas_employees)

        self.sb_year.valueChanged.connect(self.export_startup_values)

    def assign_table_signals(self, table: QTableWidget):
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
                    [shift[header] for header in self.header_shifts]
                    for shift in self.startup_values["shifts"]
                ],
                obj=self.table_shifts,
            ),
            timeoff=dict(
                values=[
                    [row[header] for header in self.header_timeoff]
                    for row in self.startup_values["timeoff"]
                ],
                obj=self.table_timeoff,
            ),
            restrictions=dict(
                values=[
                    [row[header] for header in self.header_restrictions]
                    for row in self.startup_values["restrictions"]
                ],
                obj=self.table_restrictions,
            ),
        )
        for items in tables.values():
            table = items["obj"]
            values = items["values"]
            self.add_items(table, values)

        # Month dropdown
        comboBox: QComboBox = self.drop_month
        months = [
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
        ]

        comboBox.addItems(months)
        monthNow = self.startup_values.get("month", months[datetime.now().month - 1])
        comboBox.setCurrentText(monthNow)

        # Date edits
        comboBox.currentIndex()
        for dateEdit in [self.de_shiftException, self.de_timeoff]:
            dateEdit.setDate(QDate(datetime.now().year, months.index(monthNow) + 1, 1))

        # Tab employees areas
        self.create_tab_areas_employees()

        # Periods dropdowns
        self.drop_shiftPeriod.addItems(["Manhã", "Tarde", "Noite"])
        self.drop_timeoffPeriod.addItems(["Manhã", "Tarde", "Noite"])

        # Table headers
        self.table_shifts.setHorizontalHeaderLabels(self.header_shifts)
        self.table_timeoff.setHorizontalHeaderLabels(self.header_timeoff)
        self.table_restrictions.setHorizontalHeaderLabels(self.header_restrictions)

        # Dropdowns
        self.update_dropEmp()
        self.update_dropAreas()

        year = (
            self.startup_values["year"]
            if isinstance(self.startup_values["year"], int)
            else datetime.now().year
        )
        self.sb_year.setValue(year)

    def update_dropEmp(self):
        for comboBox in [self.drop_timeoffEmp]:
            comboBox.addItems(self.employees)

    def update_dropAreas(self):
        for comboBox in [self.drop_restrictionArea, self.drop_shiftArea]:
            comboBox.addItems(self.all_areas)

    def add_items(self, table: QTableWidget, rows: list):
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
            self.add_items(table, [value])
        self.input_area.clear()
        self.input_subarea.clear()

    def add_employee(self):
        input_employee: QTextEdit = self.input_employee
        employee = input_employee.toPlainText().replace("\n", "")

        if employee:
            table: QTableWidget = self.table_employees
            value = QTableWidgetItem(employee)
            self.add_items(table, [value])
        self.input_employee.clear()

    def remove_area(self, row):
        table: QTableWidget = self.sender()
        table.removeRow(row.row())
        self.export_startup_values()

    def export_startup_values(self):
        export_dict = {}

        comboBox: QComboBox = self.drop_month
        export_dict["month"] = comboBox.currentText()

        # Checks that relates employees and areas
        d = defaultdict(lambda: dict())
        for e in self.employees:
            for a in self.areas:
                d[e][a] = self.employees_areas.get(e, {}).get(a, False)
        export_dict["employees_areas"] = d

        # Tables
        export_dict["areas"] = self.get_table_data(self.table_areas)
        export_dict["employees"] = self.get_table_data(self.table_employees)
        export_dict["shifts"] = self.get_table_data(self.table_shifts)
        export_dict["timeoff"] = self.get_table_data(self.table_timeoff)
        export_dict["restrictions"] = self.get_table_data(self.table_restrictions)

        export_dict["year"] = self.sb_year.value()

        with open("inputs.json", "w") as f:
            json_object = json.dumps(export_dict, indent=4)
            f.write(json_object)

    def get_table_data(self, table: QTableWidget):
        n_rows = table.rowCount()
        data = []
        headers = [table.horizontalHeaderItem(i) for i in range(table.columnCount())]
        headers = [
            str(i) if header is None else header.text()
            for i, header in enumerate(headers)
        ]
        for i in range(n_rows):
            row = {}
            for j, header in enumerate(headers):
                value = table.item(i, j)
                if value is None:
                    value = ""
                else:
                    value = value.text()
                row[header] = value
            data.append(row)

        if len(headers) == 1:
            data = [value[headers[0]] for value in data]
        return data

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
        for j, key in enumerate(self.header_shifts):
            table.setItem(i, j, QTableWidgetItem(data[key]))

    def add_shift_exception(self):
        string = self.get_date(self.de_shiftException)
        string = QTableWidgetItem(string)
        self.add_items(self.table_shiftException, [string])

    def add_timeoff(self):
        employee = self.drop_timeoffEmp.currentText()
        date = self.get_date(self.de_timeoff)
        period = self.drop_timeoffPeriod.currentText()
        self.add_items(self.table_timeoff, [[employee, date, period]])

    def add_restriction(self):
        area = self.drop_restrictionArea.currentText()
        n = str(self.sb_restrictionN.value())
        self.add_items(self.table_restrictions, [[area, n]])

    def get_date(self, widget: QDateEdit):
        year, month, day = widget.date().getDate()
        string = f"{day:02d}/{month:02d}/{year}"
        return string


if __name__ == "__main__":
    with open("inputs.json", "r") as f:
        startup_values = json.load(f)

    app = QApplication(sys.argv)  # Create an instance of Application
    window = Ui(startup_values)  # Create an instance of our class
    window.show()
    app.exec()  # Start the application
