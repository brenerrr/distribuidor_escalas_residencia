# %%
from src.manager import Manager
import pandas as pd

GA_params = dict(n_solutions=20)

month = pd.read_excel("inputs.xlsx", keep_default_na=False, sheet_name="mês").columns[0]
manager = Manager(2023, month, GA_params)

timeoff_inputs = pd.read_excel(
    "inputs.xlsx", keep_default_na=False, sheet_name="folgas"
)
manager.add_timeoff(timeoff_inputs)

areas = pd.read_excel("inputs.xlsx", keep_default_na=False, sheet_name="áreas")
for _, row in areas.iterrows():
    row[5] = row[5].replace(" ", "").split(",")
    row[8] = row[8].replace(" ", "").split(",")
    manager.add_shift_params(*row)

employees = pd.read_excel("inputs.xlsx", sheet_name="funcionários")
employees = employees.fillna(False)
for col in employees.columns[1:]:
    employees[col] = employees[col].astype(bool)
for _, row in employees.iterrows():
    name = row[0]
    manager.add_employee(name, employees.columns[1:][row[1:]].values)

# %%
constraints = pd.read_excel("inputs.xlsx", sheet_name="condições")
manager.add_constraints(constraints)
# %%
manager.create_schedule()
# %%
manager.export_results(areas, month)

# %%
manager.employees.sort_values("hours_worked").to_csv(f"horas_trabalhadas_{month}.csv")
manager.employees.sort_values(f"hours_worked")
