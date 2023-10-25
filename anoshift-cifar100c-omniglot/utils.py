
from pathlib import Path
import json
import yaml
import pickle
import numpy as np


def read_config_file(dict_or_filelike):
    if isinstance(dict_or_filelike, dict):
        return dict_or_filelike

    path = Path(dict_or_filelike)
    if path.suffix == ".json":
        return json.load(open(path, "r"))
    elif path.suffix in [".yaml", ".yml"]:
        return yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    elif path.suffix in [".pkl", ".pickle"]:
        return pickle.load(open(path, "rb"))

    raise ValueError("Only JSON, YaML and pickle files supported.")


class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str):
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')
        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()


class EarlyStopper:

    def stop(self, epoch, train_loss, val_auc, test_auc):
        raise NotImplementedError("Implement this method!")

    def get_best_vl_metrics(self):
        return  self.local_val_optimum, self.metrics, self.best_epoch


class Metrics:
    def __init__(self):
        self.res = []

    def add(self, res):
        self.res.append(res)

    def report(self):
        return np.asarray(self.res)


class Patience(EarlyStopper):

    '''
    Implement common "patience" technique
    '''

    def __init__(self, patience=10, use_train_loss=True):
        self.local_val_optimum = 0
        self.use_train_loss = use_train_loss
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1
        self.metrics = Metrics()

    def stop(self, epoch, train_loss, val_auc, res: list):
        if self.use_train_loss:
            acc = train_loss
        else:
            acc = val_auc

        if acc >= self.local_val_optimum:
            self.counter = 0
            self.local_val_optimum = acc
            self.best_epoch = epoch
            self.metrics.add(res)
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

