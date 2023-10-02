import unittest
import time
import sys
import json
import os
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt, QPoint

src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src/")
sys.path.append(src_path)
from app import App

inputs = {
    "month": "Out",
    "employees_areas": {
        "Henrique": {
            "Plantao": True,
            "Puerperio": False,
            "Gineco": True,
            "AltoRisco": False,
            "Ambulatorio": False,
            "SalaParto": False,
            "Permanencia": True,
            "Evolucao": True,
            "BaixoRisco": False,
        },
        "Jessika": {
            "Plantao": False,
            "Puerperio": False,
            "Gineco": False,
            "AltoRisco": False,
            "Ambulatorio": False,
            "SalaParto": True,
            "Permanencia": False,
            "Evolucao": False,
            "BaixoRisco": False,
        },
        "Isabelle": {
            "Plantao": True,
            "Puerperio": False,
            "Gineco": False,
            "AltoRisco": False,
            "Ambulatorio": True,
            "SalaParto": False,
            "Permanencia": False,
            "Evolucao": False,
            "BaixoRisco": False,
        },
        "Rafaela": {
            "Plantao": True,
            "Puerperio": True,
            "Gineco": False,
            "AltoRisco": False,
            "Ambulatorio": False,
            "SalaParto": False,
            "Permanencia": True,
            "Evolucao": True,
            "BaixoRisco": False,
        },
        "Giovanna": {
            "Plantao": True,
            "Puerperio": False,
            "Gineco": True,
            "AltoRisco": False,
            "Ambulatorio": False,
            "SalaParto": False,
            "Permanencia": True,
            "Evolucao": True,
            "BaixoRisco": False,
        },
        "Ana Luiza": {
            "Plantao": True,
            "Puerperio": False,
            "Gineco": False,
            "AltoRisco": True,
            "Ambulatorio": False,
            "SalaParto": False,
            "Permanencia": True,
            "Evolucao": True,
            "BaixoRisco": False,
        },
        "Joyce": {
            "Plantao": True,
            "Puerperio": True,
            "Gineco": False,
            "AltoRisco": False,
            "Ambulatorio": False,
            "SalaParto": False,
            "Permanencia": True,
            "Evolucao": True,
            "BaixoRisco": False,
        },
        "Lilian": {
            "Plantao": True,
            "Puerperio": True,
            "Gineco": False,
            "AltoRisco": False,
            "Ambulatorio": False,
            "SalaParto": False,
            "Permanencia": True,
            "Evolucao": True,
            "BaixoRisco": False,
        },
        "Vitoria": {
            "Plantao": False,
            "Puerperio": False,
            "Gineco": False,
            "AltoRisco": False,
            "Ambulatorio": False,
            "SalaParto": False,
            "Permanencia": False,
            "Evolucao": False,
            "BaixoRisco": True,
        },
        "Duda": {
            "Plantao": False,
            "Puerperio": False,
            "Gineco": False,
            "AltoRisco": False,
            "Ambulatorio": False,
            "SalaParto": True,
            "Permanencia": False,
            "Evolucao": False,
            "BaixoRisco": False,
        },
        "Iara": {
            "Plantao": True,
            "Puerperio": False,
            "Gineco": True,
            "AltoRisco": False,
            "Ambulatorio": False,
            "SalaParto": False,
            "Permanencia": True,
            "Evolucao": True,
            "BaixoRisco": False,
        },
        "Marcela": {
            "Plantao": True,
            "Puerperio": False,
            "Gineco": True,
            "AltoRisco": False,
            "Ambulatorio": False,
            "SalaParto": False,
            "Permanencia": True,
            "Evolucao": True,
            "BaixoRisco": False,
        },
        "Camila": {
            "Plantao": False,
            "Puerperio": False,
            "Gineco": False,
            "AltoRisco": False,
            "Ambulatorio": False,
            "SalaParto": False,
            "Permanencia": False,
            "Evolucao": False,
            "BaixoRisco": True,
        },
    },
    "areas": [
        "Plantao",
        "Puerperio",
        "Puerperio_Ambulatorio",
        "Gineco",
        "Gineco_Cirurgia",
        "AltoRisco",
        "Ambulatorio",
        "SalaParto",
        "Permanencia",
        "Evolucao",
        "BaixoRisco",
    ],
    "employees": [
        "Henrique",
        "Jessika",
        "Isabelle",
        "Rafaela",
        "Giovanna",
        "Ana Luiza",
        "Joyce",
        "Lilian",
        "Vitoria",
        "Duda",
        "Iara",
        "Marcela",
        "Camila",
    ],
    "shifts": [
        {
            "Nome": "Plantao_Noturno",
            "Per\u00edodo": "Noite",
            "\u00c1rea": "Plantao",
            "Qtd. Pessoas": "2",
            "Dias": "Ter, Qua, Qui, Sex, S\u00e1b, Dom",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "Sim",
            "Dura\u00e7\u00e3o": "12.0",
            "Exce\u00e7\u00f5es": "",
        },
        {
            "Nome": "Plantao_Noturno_Seg",
            "Per\u00edodo": "Noite",
            "\u00c1rea": "Plantao",
            "Qtd. Pessoas": "1",
            "Dias": "Seg",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "Sim",
            "Dura\u00e7\u00e3o": "12.0",
            "Exce\u00e7\u00f5es": "",
        },
        {
            "Nome": "Plantao_Diurno",
            "Per\u00edodo": "Manh\u00e3",
            "\u00c1rea": "Plantao",
            "Qtd. Pessoas": "2",
            "Dias": "S\u00e1b, Dom",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "Sim",
            "Dura\u00e7\u00e3o": "12.0",
            "Exce\u00e7\u00f5es": "",
        },
        {
            "Nome": "Puerperio_Manha_Ambulatorio",
            "Per\u00edodo": "Manh\u00e3",
            "\u00c1rea": "Puerperio_Ambulatorio",
            "Qtd. Pessoas": "1",
            "Dias": "Ter, Qui, Sex",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "Sim",
            "Dura\u00e7\u00e3o": "8.0",
            "Exce\u00e7\u00f5es": "12/10/2023",
        },
        {
            "Nome": "Puerperio_Manha2",
            "Per\u00edodo": "Manh\u00e3",
            "\u00c1rea": "Puerperio",
            "Qtd. Pessoas": "2",
            "Dias": "Ter, Qui, Sex",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "",
            "Dura\u00e7\u00e3o": "5.0",
            "Exce\u00e7\u00f5es": "12/10/2023",
        },
        {
            "Nome": "Puerperio_Manha1",
            "Per\u00edodo": "Manh\u00e3",
            "\u00c1rea": "Puerperio",
            "Qtd. Pessoas": "3",
            "Dias": "Seg, Qua",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "",
            "Dura\u00e7\u00e3o": "5.0",
            "Exce\u00e7\u00f5es": "12/10/2023",
        },
        {
            "Nome": "Gineco_Manha1",
            "Per\u00edodo": "Manh\u00e3",
            "\u00c1rea": "Gineco",
            "Qtd. Pessoas": "2",
            "Dias": "Seg, Qui",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "",
            "Dura\u00e7\u00e3o": "5.0",
            "Exce\u00e7\u00f5es": "12/10/2023",
        },
        {
            "Nome": "Gineco_Manha2",
            "Per\u00edodo": "Manh\u00e3",
            "\u00c1rea": "Gineco",
            "Qtd. Pessoas": "3",
            "Dias": "Qua",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "",
            "Dura\u00e7\u00e3o": "5.0",
            "Exce\u00e7\u00f5es": "12/10/2023",
        },
        {
            "Nome": "Gineco_Manha3",
            "Per\u00edodo": "Manh\u00e3",
            "\u00c1rea": "Gineco",
            "Qtd. Pessoas": "4",
            "Dias": "Sex",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "",
            "Dura\u00e7\u00e3o": "5.0",
            "Exce\u00e7\u00f5es": "12/10/2023",
        },
        {
            "Nome": "Gineco_Manha_Cirurgia1",
            "Per\u00edodo": "Manh\u00e3",
            "\u00c1rea": "Gineco_Cirurgia",
            "Qtd. Pessoas": "1",
            "Dias": "Qua",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "Sim",
            "Dura\u00e7\u00e3o": "9.0",
            "Exce\u00e7\u00f5es": "12/10/2023",
        },
        {
            "Nome": "Gineco_Manha_Cirurgia2",
            "Per\u00edodo": "Manh\u00e3",
            "\u00c1rea": "Gineco_Cirurgia",
            "Qtd. Pessoas": "2",
            "Dias": "Seg, Qui",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "Sim",
            "Dura\u00e7\u00e3o": "9.0",
            "Exce\u00e7\u00f5es": "12/10/2023",
        },
        {
            "Nome": "Gineco_Manha_Cirurgia3",
            "Per\u00edodo": "Manh\u00e3",
            "\u00c1rea": "Gineco_Cirurgia",
            "Qtd. Pessoas": "4",
            "Dias": "Ter",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "Sim",
            "Dura\u00e7\u00e3o": "9.0",
            "Exce\u00e7\u00f5es": "12/10/2023",
        },
        {
            "Nome": "Alto_Risco",
            "Per\u00edodo": "Manh\u00e3",
            "\u00c1rea": "AltoRisco",
            "Qtd. Pessoas": "1",
            "Dias": "Seg, Ter, Qua, Qui, Sex",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "",
            "Dura\u00e7\u00e3o": "5.0",
            "Exce\u00e7\u00f5es": "12/10/2023",
        },
        {
            "Nome": "Ambulatorio_Manha",
            "Per\u00edodo": "Manh\u00e3",
            "\u00c1rea": "Ambulatorio",
            "Qtd. Pessoas": "1",
            "Dias": "Seg, Ter, Qua, Qui, Sex",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "",
            "Dura\u00e7\u00e3o": "4.0",
            "Exce\u00e7\u00f5es": "12/10/2023",
        },
        {
            "Nome": "Ambulatorio_Tarde",
            "Per\u00edodo": "Tarde",
            "\u00c1rea": "Ambulatorio",
            "Qtd. Pessoas": "1",
            "Dias": "Seg, Ter, Qua, Qui, Sex",
            "Obrigat\u00f3rio": "",
            "Balancear": "",
            "Dura\u00e7\u00e3o": "4.0",
            "Exce\u00e7\u00f5es": "12/10/2023",
        },
        {
            "Nome": "Sala_Parto",
            "Per\u00edodo": "Manh\u00e3",
            "\u00c1rea": "SalaParto",
            "Qtd. Pessoas": "2",
            "Dias": "Seg, Ter, Qua, Qui, Sex",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "",
            "Dura\u00e7\u00e3o": "12.0",
            "Exce\u00e7\u00f5es": "",
        },
        {
            "Nome": "Permanencia2",
            "Per\u00edodo": "Tarde",
            "\u00c1rea": "Permanencia",
            "Qtd. Pessoas": "2",
            "Dias": "Seg, Qua, Qui, Sex",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "Sim",
            "Dura\u00e7\u00e3o": "4.5",
            "Exce\u00e7\u00f5es": "12/10/2023",
        },
        {
            "Nome": "Permanencia3",
            "Per\u00edodo": "Tarde",
            "\u00c1rea": "Permanencia",
            "Qtd. Pessoas": "3",
            "Dias": "Ter",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "Sim",
            "Dura\u00e7\u00e3o": "4.5",
            "Exce\u00e7\u00f5es": "12/10/2023",
        },
        {
            "Nome": "Evolucao",
            "Per\u00edodo": "Manh\u00e3",
            "\u00c1rea": "Evolucao",
            "Qtd. Pessoas": "2",
            "Dias": "S\u00e1b, Dom",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "Sim",
            "Dura\u00e7\u00e3o": "5.0",
            "Exce\u00e7\u00f5es": "",
        },
        {
            "Nome": "BaixoRisco",
            "Per\u00edodo": "Manh\u00e3",
            "\u00c1rea": "BaixoRisco",
            "Qtd. Pessoas": "2",
            "Dias": "Seg, Ter, Qua, Qui, Sex",
            "Obrigat\u00f3rio": "Sim",
            "Balancear": "Sim",
            "Dura\u00e7\u00e3o": "12.0",
            "Exce\u00e7\u00f5es": "",
        },
    ],
    "timeoff": [
        {"Nome": "Giovanna", "Data": "07/10/2023", "Per\u00edodo": "Manh\u00e3"},
        {"Nome": "Giovanna", "Data": "07/10/2023", "Per\u00edodo": "Tarde"},
        {"Nome": "Giovanna", "Data": "07/10/2023", "Per\u00edodo": "Noite"},
        {"Nome": "Giovanna", "Data": "08/10/2023", "Per\u00edodo": "Manh\u00e3"},
        {"Nome": "Joyce", "Data": "01/10/2023", "Per\u00edodo": "Manh\u00e3"},
        {"Nome": "Joyce", "Data": "01/10/2023", "Per\u00edodo": "Tarde"},
        {"Nome": "Joyce", "Data": "01/10/2023", "Per\u00edodo": "Noite"},
        {"Nome": "Joyce", "Data": "13/10/2023", "Per\u00edodo": "Noite"},
        {"Nome": "Joyce", "Data": "14/10/2023", "Per\u00edodo": "Manh\u00e3"},
        {"Nome": "Joyce", "Data": "14/10/2023", "Per\u00edodo": "Tarde"},
        {"Nome": "Joyce", "Data": "14/10/2023", "Per\u00edodo": "Noite"},
        {"Nome": "Joyce", "Data": "15/10/2023", "Per\u00edodo": "Manh\u00e3"},
        {"Nome": "Joyce", "Data": "15/10/2023", "Per\u00edodo": "Tarde"},
        {"Nome": "Joyce", "Data": "15/10/2023", "Per\u00edodo": "Noite"},
        {"Nome": "Lilian", "Data": "01/10/2023", "Per\u00edodo": "Manh\u00e3"},
        {"Nome": "Lilian", "Data": "01/10/2023", "Per\u00edodo": "Tarde"},
        {"Nome": "Lilian", "Data": "01/10/2023", "Per\u00edodo": "Noite"},
        {"Nome": "Lilian", "Data": "27/10/2023", "Per\u00edodo": "Noite"},
        {"Nome": "Lilian", "Data": "28/10/2023", "Per\u00edodo": "Manh\u00e3"},
        {"Nome": "Lilian", "Data": "28/10/2023", "Per\u00edodo": "Tarde"},
        {"Nome": "Lilian", "Data": "28/10/2023", "Per\u00edodo": "Noite"},
        {"Nome": "Lilian", "Data": "29/10/2023", "Per\u00edodo": "Manh\u00e3"},
        {"Nome": "Lilian", "Data": "29/10/2023", "Per\u00edodo": "Tarde"},
        {"Nome": "Lilian", "Data": "29/10/2023", "Per\u00edodo": "Noite"},
        {"Nome": "Iara", "Data": "13/10/2023", "Per\u00edodo": "Noite"},
        {"Nome": "Iara", "Data": "14/10/2023", "Per\u00edodo": "Manh\u00e3"},
        {"Nome": "Iara", "Data": "14/10/2023", "Per\u00edodo": "Tarde"},
        {"Nome": "Iara", "Data": "14/10/2023", "Per\u00edodo": "Noite"},
        {"Nome": "Iara", "Data": "15/10/2023", "Per\u00edodo": "Manh\u00e3"},
        {"Nome": "Iara", "Data": "15/10/2023", "Per\u00edodo": "Tarde"},
        {"Nome": "Iara", "Data": "15/10/2023", "Per\u00edodo": "Noite"},
        {"Nome": "Iara", "Data": "11/10/2023", "Per\u00edodo": "Tarde"},
        {"Nome": "Marcela", "Data": "11/10/2023", "Per\u00edodo": "Tarde"},
    ],
    "restrictions": [{"\u00c1rea": "Plantao", "Qtd. M\u00e1xima de Turnos": "8"}],
    "year": 2023,
    "n_solutions": 1,
}


folderpath = os.path.dirname(sys.argv[0])
inputs_path = os.path.join(folderpath, "test_inputs.json")


def create_json():
    with open(inputs_path, "w") as f:
        json.dump(inputs, f, indent=4)


def read_json():
    with open(inputs_path, "r") as f:
        return json.load(f)


def delete_json():
    try:
        os.remove(inputs_path)
    except:
        pass


create_json()
app = App(sys.argv, inputs_path)


class Test(unittest.TestCase):
    def test_add_person(self):
        """
        Test adding a person
        """
        QTest.mouseClick(app.main.input_employee, Qt.LeftButton)
        QTest.keyClicks(app.main.input_employee, "ABC", Qt.NoModifier)
        QTest.mouseClick(app.main.button_addEmployee, Qt.LeftButton)
        data = read_json()
        self.assertEqual(
            data["employees"],
            [
                "Henrique",
                "Jessika",
                "Isabelle",
                "Rafaela",
                "Giovanna",
                "Ana Luiza",
                "Joyce",
                "Lilian",
                "Vitoria",
                "Duda",
                "Iara",
                "Marcela",
                "Camila",
                "ABC",
            ],
        )

    def test_add_area(self):
        """
        Test adding an area
        """
        QTest.mouseClick(app.main.input_area, Qt.LeftButton)
        QTest.keyClicks(app.main.input_area, "ABC", Qt.NoModifier)
        QTest.mouseClick(app.main.button_addArea, Qt.LeftButton)
        data = read_json()
        self.assertEqual(
            data["areas"],
            [
                "Plantao",
                "Puerperio",
                "Puerperio_Ambulatorio",
                "Gineco",
                "Gineco_Cirurgia",
                "AltoRisco",
                "Ambulatorio",
                "SalaParto",
                "Permanencia",
                "Evolucao",
                "BaixoRisco",
                "ABC",
            ],
        )

    def test_add_timeoff(self):
        """
        Test adding time off
        """
        QTest.mouseClick(app.main.button_addTimeoff, Qt.LeftButton)
        data = read_json()
        self.assertEqual(
            data["timeoff"][-1],
            {"Nome": "Henrique", "Data": "01/10/2023", "Período": "Manhã"},
        )

    def test_add_restrictions(self):
        """
        Test adding restrictions
        """
        QTest.mouseClick(app.main.button_addRestriction, Qt.LeftButton)
        data = read_json()
        self.assertEqual(
            data["restrictions"][-1],
            {"Área": "Plantao", "Qtd. Máxima de Turnos": "0"},
        )

    def test_add_shift(self):
        """
        Test adding shift
        """
        QTest.mouseClick(app.main.button_addShift, Qt.LeftButton)
        data = read_json()
        self.assertEqual(
            data["shifts"][-1],
            {
                "Nome": "",
                "Per\u00edodo": "Manh\u00e3",
                "\u00c1rea": "Plantao",
                "Qtd. Pessoas": "0",
                "Dias": "",
                "Obrigat\u00f3rio": "",
                "Balancear": "",
                "Dura\u00e7\u00e3o": "4.0",
                "Exce\u00e7\u00f5es": "",
            },
        )

    def test_assign_area(self):
        QTest.mouseClick(app.main.tabAreasEmployees, Qt.LeftButton)
        checkbox = app.checks_emp_areas["Henrique"]["BaixoRisco"]
        QTest.mouseClick(
            checkbox, Qt.LeftButton, pos=QPoint(2, int(checkbox.height() / 2))
        )
        app.checks_emp_areas["Henrique"]["BaixoRisco"].isChecked()
        data = read_json()
        self.assertEqual(
            data["employees_areas"]["Henrique"],
            {
                "Plantao": True,
                "Puerperio": False,
                "Gineco": True,
                "AltoRisco": False,
                "Ambulatorio": False,
                "SalaParto": False,
                "Permanencia": True,
                "Evolucao": True,
                "BaixoRisco": True,
                "ABC": False,
            },
        )

    def tearDown(self) -> None:
        delete_json()
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
