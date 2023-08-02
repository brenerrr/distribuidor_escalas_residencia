# %%
from src.manager import Manager
import pandas as pd

GA_params = dict(n_solutions=5)
manager = Manager(2023, 9, GA_params)

inputs_areas = pd.read_excel("inputs.xlsx", keep_default_na=False, sheet_name="areas")
for _, row in inputs_areas.iterrows():
    row[5] = row[5].replace(" ", "").split(",")
    row[8] = row[8].replace(" ", "").split(",")
    manager.add_shift_params(*row)
manager.shifts_params

employees = pd.read_excel("inputs.xlsx", sheet_name="employees")
employees = employees.fillna(False)
for col in employees.columns[1:]:
    employees[col] = employees[col].astype(bool)
for _, row in employees.iterrows():
    name = row[0]
    manager.add_employee(name, employees.columns[1:][row[1:]].values)
inputs_areas

# %%
manager.create_schedule()

manager.export_results(inputs_areas)

# %%
manager.employees.sort_values("hours_worked").to_csv("hours_worked.csv")
manager.employees.sort_values("hours_worked")
