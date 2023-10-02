from src.manager import Manager
import json
import os
import sys

from src.app import App

if sys.version_info[0:2] != (3, 10):
    raise Exception("Requires python 3.10")


if __name__ == "__main__":
    folderpath = os.path.dirname(sys.argv[0])
    inputs_path = os.path.join(folderpath, "inputs.json")
    app = App(sys.argv, inputs_path)
    app.exec()

    # manager = Manager("inputs.json")
    # manager.create_schedule()
    # manager.export_results()
