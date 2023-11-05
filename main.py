import optuna
from optuna import Trial
from optuna.study import StudyDirection
from src.train import train
import os
from src.generate_val import generate_val
from src.parsedata import get_data, parse
from src.cluster_policies import cluster_policies

if __name__ == "__main__":
    
    generate_val(5)
    number_final_policies = 6

    try:
        file_path = 'db.sqlite3'
        os.remove(file_path)
    except:
        pass

    def objective(trial: Trial) -> float:
        return train(
            trial.suggest_int("max_episodes", 3, 5),
            trial.suggest_int("train_freq", 10, 50),
            trial.suggest_int("new_freq", 1, 3),
            5, 
            trial.suggest_categorical("mask_option", [0,1]),
            trial.suggest_int("sel_k", 1, 3),
            trial.suggest_categorical("batch_size", [64, 128]),
            trial.suggest_float("lr", 0.0002, 0.0004),
            trial.suggest_categorical("hidden_channels", [32, 64, 128]),
            1,
            1,
            trial.suggest_int("j_max",10,15),
            trial.suggest_int("j_min",5,9),
            trial.suggest_int("m_max",9,13),
            trial.suggest_int("m_min",4,7),
            trial.suggest_int("op_max",5,9),
            trial.suggest_int("max_processing",8,25),
            )

    study = optuna.create_study(
            storage="sqlite:///db.sqlite3",
            study_name="fjsp_study",
            load_if_exists=True,
            direction=StudyDirection.MAXIMIZE,
        )

    study.optimize(objective, n_trials=5)

    cluster_policies(number_final_policies)

