# %%
from src.manager import Manager
import pandas as pd
import json


manager = Manager("inputs.json")
manager.create_schedule()
manager.export_results()
