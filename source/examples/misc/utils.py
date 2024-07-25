import os
import json
import math
import pandas

def save_results(history, parameters, experiment_path, average_epochs: int=10):
    os.makedirs(experiment_path, exist_ok=True)

    history = pandas.DataFrame(history)
    history.to_csv(experiment_path / "history.csv", index=False)

    parameters["estimated_mutual_information"] = history["mutual_information"][-average_epochs:].mean()
    parameters["estimated_mutual_information_std"] = history["mutual_information"][-average_epochs:].std() / math.sqrt(average_epochs)
    with open(experiment_path / "parameters.json", 'w') as file:
        json.dump(parameters, file, indent=4)