# %%
from src.manager import Manager
import pandas as pd
import json


manager = Manager("inputs.json")
manager.create_schedule()
manager.export_results()

# %%
# manager.employees.sort_values("hours_worked").to_csv(f"horas_trabalhadas_{month}.csv")
# manager.employees.sort_values(f"hours_worked")
