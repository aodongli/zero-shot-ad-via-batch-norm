from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from models.feature_model import FeatureSVDD
from losses.dsvdd_loss import CenterDistLoss
from trainers.zeroshot_trainer import ZeroShotMetaTrainer

model_map = {
    'feat_svdd': FeatureSVDD,
}

trainer_map = {
    'zeroshot_meta_trainer': ZeroShotMetaTrainer,
}

loss_map = {
    'center_dist': CenterDistLoss,
}

optimizer_map = {
    'Adam': Adam,
}

scheduler_map = {
    'StepLR': StepLR
}

# early_stopper_map = {
#     'Patience': Patience
# }

import os
from pathlib import Path
import json
import yaml
import pickle
import numpy as np

def model_config_reader(config_file_name):
    model_config = None
    if isinstance(config_file_name, dict):
        model_config =  config_file_name

    path = Path(os.path.join('config_files', config_file_name))
    if path.suffix == ".json":
        model_config =  json.load(open(path, "r"))
    elif path.suffix in [".yaml", ".yml"]:
        model_config =  yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    elif path.suffix in [".pkl", ".pickle"]:
        model_config =  pickle.load(open(path, "rb"))
    else:
        raise ValueError("Only JSON, YaML and pickle files supported.")

    model_config['model_class'] = model_map[model_config['model']]
    model_config['trainer_class'] = trainer_map[model_config['trainer']]
    model_config['loss_class'] = loss_map[model_config['loss']]
    model_config['optimizer_class'] = optimizer_map[model_config['optimizer']]

    sched_dict = model_config['scheduler']
    sched_s = sched_dict['class']
    args = sched_dict['args']
    model_config['scheduler_class'] = lambda opt: scheduler_map[sched_s](opt, **args)

    return model_config