import logging
import numbers
import os
from typing import Dict, Optional, Tuple

import mlflow
from aim import Run
from glom import glom


def update_records(records, new_record):
    if not isinstance(records, type(new_record)):
        print("")
        return records
    if isinstance(records, list):
        records += new_record
    elif isinstance(records, dict):
        for key in new_record:
            records[key] = update_records(records[key], new_record[key]) if key in records else new_record[key]
    return records

def init_mlflow_run(
    debug: bool = False,
    tracking_uri: str = "./mlruns/"
) -> str:
    mlflow.utils.logging_utils.suppress_logs('mlflow', '.*')
    mlflow.set_experiment("beh-seq-modelling"+ ("-debug" if debug else ""))
    
    # mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow_run = mlflow.start_run(log_system_metrics=True)
    mlflow_run_id = mlflow_run.info.run_id
    mlflow_run_name = None if not mlflow_run else mlflow_run.data.tags.get('mlflow.runName')
    logging.info(f"Mlflow run name: {mlflow_run_name}, id: {mlflow_run_id}")
    return mlflow_run_id

def init_aim_run(
    debug: bool = False,
    exp_name: str = 'beh-seq-modelling',
    repo: str = '.aim'
) -> Run:
    aim_run = Run(
        repo=repo,
        experiment=exp_name + ("-debug" if debug else ""),
        log_system_params=True,
        capture_terminal_logs=True
    )
    return aim_run


def log_params(
    experiment_tracker: Optional[dict],
    params: Optional[dict],
    prefix: str = "",
) -> None:
    primitives = (bool, str, int, float, type(None))
    formatted_params = {k:str(type(v)) if type(v) not in primitives else v for k,v in params.items()}
    if experiment_tracker is None:
        return
    if 'mlflow' in experiment_tracker.keys():
        mlflow.log_params(formatted_params, run_id=glom(experiment_tracker, 'mlflow.run_id'))
    if 'aim' in experiment_tracker.keys():
        if prefix and glom(experiment_tracker, 'aim.run').get(prefix):
            glom(experiment_tracker, 'aim.run')[prefix].update(formatted_params)
        elif prefix:
            glom(experiment_tracker, 'aim.run').set(prefix, formatted_params)
        else:
            list(map(lambda kv: glom(experiment_tracker, 'aim.run').set(kv[0], kv[1]), formatted_params.items()))
            # glom(experiment_tracker, 'aim.run')[prefix] = formatted_params


def log_metrics(
    experiment_tracker: Optional[dict] = None,
    prefix="train",
    batch_id: int = 0,
    metrics: Optional[dict] = None
) -> None:
    formatted_metrics = {f"{prefix}/{key}": float(value) if not isinstance(value, numbers.Number) else value for key, value in metrics.items()}
    # __import__('ipdb').set_trace()
    if experiment_tracker is None: 
        return
    if 'mlflow' in experiment_tracker.keys():
        mlflow.log_metrics(formatted_metrics, step=batch_id, run_id=glom(experiment_tracker, 'mlflow.run_id'))
    if 'aim' in experiment_tracker.keys():
        glom(experiment_tracker, 'aim.run').track(metrics, step=batch_id)
    if 'tensorboard' in experiment_tracker.keys():
        for key, value in formatted_metrics.items():
            experiment_tracker['tensorboard'].add_scalar(key, value, batch_id)

def log_state_dict(
    experiment_tracker: Optional[dict],
    batch_id: int,
    state_dict: dict,
    file_path: str,  # Aim cannot directly log state_dict, it copys the file to the artifact
) -> None:
    if experiment_tracker is None:
        return
    if 'mlflow' in experiment_tracker.keys():
        mlflow.pytorch.log_state_dict(state_dict, artifact_path=f"checkpoint/{batch_id}")
    if 'aim' in experiment_tracker.keys() and file_path is not None:
        glom(experiment_tracker, 'aim.run').log_artifact(file_path, name=f'checkpoint/{batch_id}/state_dict.pth')

def log_csv(
    artifact,
    experiment_tracker: Optional[dict],
    batch_id: int,
    file_path: str = None,
    tmp_path = "/data/tmp"
) -> None:
    if experiment_tracker is None:
        return
    os.makedirs(os.path.join(tmp_path, str(batch_id)), exist_ok=True)
    tmp_path = os.path.join(tmp_path, str(batch_id), file_path)
    try:
        if isinstance(artifact, polars.DataFrame):
            artifact.write_csv(tmp_path)
        # artifact.to_csv(tmp_path, index=False, encoding='utf-8')
    except:
        print("Failed to save csv file")
        return
    
    # if 'mlflow' in experiment_tracker.keys():
    #     mlflow.log_artifact(file_path, artifact_path=artifact_path)
    if 'aim' in experiment_tracker.keys() and file_path is not None:
        glom(experiment_tracker, 'aim.run').log_artifact(tmp_path, name=f"checkpoint/{batch_id}/{file_path}")

def log_parquet(
    artifact,
    experiment_tracker: Optional[dict],
    batch_id: int,
    file_path: str = None,
    tmp_path = "/data/tmp"
) -> None:
    if experiment_tracker is None:
        return
    os.makedirs(os.path.join(tmp_path, str(batch_id)), exist_ok=True)
    tmp_path = os.path.join(tmp_path, str(batch_id), file_path)
    try:
        artifact.write_parquet(tmp_path, statistics=False)
    except Exception as e:
        print(e)
        print("Failed to save parquet file")
        return
    if 'aim' in experiment_tracker.keys() and file_path is not None:
        glom(experiment_tracker, 'aim.run').log_artifact(tmp_path, name=f"checkpoint/{batch_id}/{file_path}")
