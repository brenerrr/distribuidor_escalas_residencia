import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from PyQt5.QtCore import Qt, QEvent, QDate

from PyQt5.QtWidgets import (
    QWidget,
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
    QMessageBox,
    QSizePolicy,
)
from datetime import datetime
import sys
import json
from collections import defaultdict
import pandas as pd
import os
from waiting_spinner import QtWaitingSpinner
from worker_thread import WorkerThread
from main_window import MainWindow
from status_window import StatusWindow

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"


class App(QApplication):
    def __init__(self, args, inputs_path: str):
        super(App, self).__init__(args)  # Call the inherited classes __init__ method

        self.read_inputs(inputs_path)
        self.main = MainWindow()
        self.progress = StatusWindow()
        self.progress.waitingSpinner = QtWaitingSpinner(self.progress, False)
        self.progress.verticalLayout.insertWidget(0, self.progress.waitingSpinner)
        self.progress.waitingSpinner.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.progress.waitingSpinner.start()
        self.setup_error_window()
        self.setup_success_window()
        self.main.show()

        self.initialize_variables()
        self.populate_widgets_startup()
        self.setup_signals()
        self.format_tables_columns(
            [
                self.main.table_areas,
                self.main.table_employees,
                self.main.table_shiftException,
                self.main.table_timeoff,
                self.main.table_restrictions,
            ]
        )

    @property
    def all_areas(self):
        areas = self.read_table(self.main.table_areas)
        if areas.empty:
            return []
        else:
            return areas[0].values

    @property
    def areas(self):
        areas = self.read_table(self.main.table_areas)
        if areas.empty:
            return []
        else:
            return areas[0].str.split("_").str[0].unique()

    @property
    def employees(self):
        employees = self.read_table(self.main.table_employees)
        if employees.empty:
            return []
        else:
            return employees[0]

    def read_inputs(self, inputs_path: str):
        try:
            with open(inputs_path, "r") as f:
                startup_values = json.load(f)
        except FileNotFoundError:
            startup_values = {}

        self.inputs = defaultdict(lambda: defaultdict(lambda: {}))
        self.inputs.update(startup_values)
        self.inputs_path = inputs_path

    def initialize_variables(self):
        self.employees_areas = {}
        self.employees_areas.update(self.inputs["employees_areas"])
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
        self.started_worker_thread = False

    def setup_error_window(self):
        error = QMessageBox(self.progress)
        error.setWindowTitle("Erro")
        error.setIcon(QMessageBox.Icon.Critical)
        self.error = error

    def setup_success_window(self):
        success = QMessageBox(self.progress)
        success.setWindowTitle("Concluído")
        success.setIcon(QMessageBox.Icon.Information)
        self.success = success

    def read_table(self, table):
        n_rows = table.rowCount()
        n_cols = table.columnCount()
        data = [table.item(r, c).text() for r in range(n_rows) for c in range(n_cols)]
        return pd.DataFrame(data)

    def create_tab_areas_employees(
        self,
    ):
        layout: QGridLayout = self.main.grid_employees_areas

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
                check.stateChanged.connect(self.export_inputs)

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
        tab: QTabWidget = self.main.tabWidget

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
        self.main.button_addArea.clicked.connect(self.add_area)
        self.main.button_addEmployee.clicked.connect(self.add_employee)
        self.main.button_addRestriction.clicked.connect(self.add_restriction)
        self.main.button_addTimeoff.clicked.connect(self.add_timeoff)
        self.main.button_addShift.clicked.connect(self.add_shift)
        self.main.button_addShiftDate.clicked.connect(self.add_shift_exception)
        self.main.button_executeManager.clicked.connect(self.call_manager)

        self.main.drop_month.currentTextChanged.connect(self.export_inputs)

        # Event filters
        self.main.input_area.installEventFilter(self)
        self.main.input_subarea.installEventFilter(self)
        self.main.input_employee.installEventFilter(self)

        # Tables
        self.assign_table_signals(self.main.table_areas)
        self.assign_table_signals(self.main.table_employees)
        self.assign_table_signals(self.main.table_shifts)
        self.assign_table_signals(self.main.table_timeoff)
        self.assign_table_signals(self.main.table_restrictions)
        self.assign_table_signals(self.main.table_shiftException)

        self.main.table_employees.itemChanged.connect(self.update_dropEmp)
        self.main.table_areas.itemChanged.connect(self.update_dropAreas)
        self.main.table_areas.itemChanged.connect(self.create_tab_areas_employees)
        self.main.table_employees.itemChanged.connect(self.create_tab_areas_employees)

        self.main.table_areas.doubleClicked.connect(self.update_dropAreas)
        self.main.table_areas.doubleClicked.connect(self.create_tab_areas_employees)
        self.main.table_employees.doubleClicked.connect(self.update_dropEmp)
        self.main.table_employees.doubleClicked.connect(self.create_tab_areas_employees)

        self.main.sb_year.valueChanged.connect(self.export_inputs)
        self.main.sb_nSolutions.valueChanged.connect(self.export_inputs)

    def assign_table_signals(self, table: QTableWidget):
        table.doubleClicked.connect(self.remove_area)
        table.itemChanged.connect(self.export_inputs)
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
        if pressed_Enter(self.main.input_area) or pressed_Enter(
            self.main.input_subarea
        ):
            self.add_area()
            return True

        # In case enter is pressed on input_employee text edit
        if pressed_Enter(self.main.input_employee):
            self.add_employee()
            return True

        return super().eventFilter(obj, event)

    def populate_widgets_startup(self):
        # Tables
        tables = dict(
            areas=dict(values=self.inputs["areas"], obj=self.main.table_areas),
            employees=dict(
                values=self.inputs["employees"], obj=self.main.table_employees
            ),
            shifts=dict(
                values=[
                    [shift[header] for header in self.header_shifts]
                    for shift in self.inputs["shifts"]
                ],
                obj=self.main.table_shifts,
            ),
            timeoff=dict(
                values=[
                    [row[header] for header in self.header_timeoff]
                    for row in self.inputs["timeoff"]
                ],
                obj=self.main.table_timeoff,
            ),
            restrictions=dict(
                values=[
                    [row[header] for header in self.header_restrictions]
                    for row in self.inputs["restrictions"]
                ],
                obj=self.main.table_restrictions,
            ),
        )
        for items in tables.values():
            table = items["obj"]
            values = items["values"]
            self.add_items(table, values)

        # Month dropdown
        comboBox: QComboBox = self.main.drop_month
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
        monthNow = self.inputs.get("month", months[datetime.now().month - 1])
        comboBox.setCurrentText(monthNow)

        # Date edits
        comboBox.currentIndex()
        for dateEdit in [self.main.de_shiftException, self.main.de_timeoff]:
            dateEdit.setDate(QDate(datetime.now().year, months.index(monthNow) + 1, 1))

        # Tab employees areas
        self.create_tab_areas_employees()

        # Periods dropdowns
        self.main.drop_shiftPeriod.addItems(["Manhã", "Tarde", "Noite"])
        self.main.drop_timeoffPeriod.addItems(["Manhã", "Tarde", "Noite"])

        # Table headers
        self.main.table_shifts.setHorizontalHeaderLabels(self.header_shifts)
        self.main.table_timeoff.setHorizontalHeaderLabels(self.header_timeoff)
        self.main.table_restrictions.setHorizontalHeaderLabels(self.header_restrictions)

        # Dropdowns
        self.update_dropEmp()
        self.update_dropAreas()

        year = (
            self.inputs["year"]
            if isinstance(self.inputs["year"], int)
            else datetime.now().year
        )
        self.main.sb_year.setValue(year)

    def update_dropEmp(self):
        for comboBox in [self.main.drop_timeoffEmp]:
            comboBox.addItems(self.employees)

    def update_dropAreas(self):
        for comboBox in [self.main.drop_restrictionArea, self.main.drop_shiftArea]:
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
        input_area: QTextEdit = self.main.input_area
        input_subarea: QTextEdit = self.main.input_subarea
        area = input_area.toPlainText().replace("\n", "")
        subarea = input_subarea.toPlainText().replace("\n", "")
        if subarea:
            area = area + "_" + subarea
        if area:
            table: QTableWidget = self.main.table_areas
            value = QTableWidgetItem(area)
            self.add_items(table, [value])
        self.main.input_area.clear()
        self.main.input_subarea.clear()

    def add_employee(self):
        input_employee: QTextEdit = self.main.input_employee
        employee = input_employee.toPlainText().replace("\n", "")

        if employee:
            table: QTableWidget = self.main.table_employees
            value = QTableWidgetItem(employee)
            self.add_items(table, [value])
        self.main.input_employee.clear()

    def remove_area(self, row):
        table: QTableWidget = self.sender()
        table.removeRow(row.row())
        self.export_inputs()

    def export_inputs(self):
        export_dict = {}

        comboBox: QComboBox = self.main.drop_month
        export_dict["month"] = comboBox.currentText()

        # Checks that relates employees and areas
        d = defaultdict(lambda: dict())
        for e in self.employees:
            for a in self.areas:
                d[e][a] = self.employees_areas.get(e, {}).get(a, False)
        export_dict["employees_areas"] = d

        # Tables
        export_dict["areas"] = self.get_table_data(self.main.table_areas)
        export_dict["employees"] = self.get_table_data(self.main.table_employees)
        export_dict["shifts"] = self.get_table_data(self.main.table_shifts)
        export_dict["timeoff"] = self.get_table_data(self.main.table_timeoff)
        export_dict["restrictions"] = self.get_table_data(self.main.table_restrictions)

        export_dict["year"] = self.main.sb_year.value()
        export_dict["n_solutions"] = self.main.sb_nSolutions.value()

        path = os.path.join(self.inputs_path)
        with open(path, "w") as f:
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
        data["Nome"] = self.main.input_shiftName.toPlainText()
        data["Período"] = self.main.drop_shiftPeriod.currentText()
        data["Área"] = self.main.drop_shiftArea.currentText()
        data["Qtd. Pessoas"] = str(self.main.sb_shiftNEmp.value())

        days_of_week = []
        for i in range(7):
            value = getattr(self.main, f"cb_shiftDay_{i}").isChecked()
            if value:
                label = getattr(self.main, f"label_dayOfWeek_{i}").text()
                days_of_week.append(label)
        data["Dias"] = ", ".join(days_of_week)

        data["Balancear"] = "Sim" if self.main.cb_shiftBalance.isChecked() else None
        data["Obrigatório"] = "Sim" if self.main.cb_shiftMandatory.isChecked() else None
        data["Duração"] = str(self.main.sb_shiftDuration.value())
        data["Exceções"] = self.read_table(self.main.table_shiftException).get(0)
        if data["Exceções"] is not None:
            l = data["Exceções"].astype(str).tolist()
            data["Exceções"] = ", ".join(l)

        table: QTableWidget = self.main.table_shifts
        i = table.rowCount()
        table.insertRow(i)
        for j, key in enumerate(self.header_shifts):
            table.setItem(i, j, QTableWidgetItem(data[key]))

    def add_shift_exception(self):
        string = self.get_date(self.main.de_shiftException)
        string = QTableWidgetItem(string)
        self.add_items(self.main.table_shiftException, [string])

    def add_timeoff(self):
        employee = self.main.drop_timeoffEmp.currentText()
        date = self.get_date(self.main.de_timeoff)
        period = self.main.drop_timeoffPeriod.currentText()
        self.add_items(self.main.table_timeoff, [[employee, date, period]])

    def add_restriction(self):
        area = self.main.drop_restrictionArea.currentText()
        n = str(self.main.sb_restrictionN.value())
        self.add_items(self.main.table_restrictions, [[area, n]])

    def get_date(self, widget: QDateEdit):
        year, month, day = widget.date().getDate()
        string = f"{day:02d}/{month:02d}/{year}"
        return string

    def call_manager(self):
        # Worker thread
        if not self.started_worker_thread:
            self.w_thread = WorkerThread()

            self.w_thread.finished.connect(self.finish_worker_thread)
            self.w_thread.error.connect(self.create_popup)
            self.w_thread.status.connect(self.update_label)

            self.w_thread.start()
            self.started_worker_thread = True
            self.progress.show()

    def finish_worker_thread(self):
        print("finishing worker_thread")
        self.started_worker_thread = False
        self.w_thread.quit()
        self.w_thread.wait()
        self.progress.allow_closing()
        self.progress.close()
        self.progress.prevent_closing()
        print("finished_worker_thread")

    def create_popup(self, n: int):
        if n == -1:
            self.error.setText(
                "Não foi possível achar uma solução com os inputs fornecidos"
            )
            self.error.show()
            self.progress.close()
        if n == 1:
            self.success.setText("Escala gerada com sucesso.")
            self.success.show()
            self.progress.close()

    def update_label(self, s: str):
        self.progress.label.setText(s)
