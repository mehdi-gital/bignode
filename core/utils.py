from typing import Dict, Union
import os
import datetime
import pickle

import torch


def experiment_init(
        experiment_name: str,
        graph_obj: Dict[str, Union[int, torch.Tensor]],
        system_data: Dict[str, Union[str, torch.Tensor]] = None,
) -> Dict[str, str]:
    # creating experiment name
    run_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_artifact_dir = os.path.join("artifacts", f"{experiment_name}_{run_timestamp}")

    # creating experiment directories
    model_dir = os.path.join(experiment_artifact_dir, 'model')
    visualization_dir = os.path.join(experiment_artifact_dir, 'visualization')
    if system_data is not None:
        os.makedirs(model_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    # if not prefix:
    data_dir = os.path.join(experiment_artifact_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    if system_data is not None:
        dirs = {
            "data_dir": data_dir,
            "model_dir": model_dir,
            "visualization_dir": visualization_dir,
        }
    else:
        dirs = {
            "data_dir": data_dir,
            "visualization_dir": visualization_dir,
        }
    # storing generated graph and system data
    with open(os.path.join(data_dir, 'graph_obj.pickle'), 'wb') as handle:
        pickle.dump(graph_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if system_data is not None:
        with open(os.path.join(data_dir, 'system_data.pickle'), 'wb') as handle:
            pickle.dump(system_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dirs
