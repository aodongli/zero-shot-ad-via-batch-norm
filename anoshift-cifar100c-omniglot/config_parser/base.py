from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils import Patience
from .data_info import *
from copy import deepcopy
from utils import read_config_file
from models.image_model import OmniglotCNN, OmniglotDSVDD, Cifar100DSVDD, CNN4DSVDD,CNN4ClS
from models.tabular_model import TabDSVDD,TabNeutralAD
from losses.dsvdd_loss import CenterDistLoss
from losses.ntl_loss import DCL
from trainers.zeroshot_trainer import ZeroShotMetaTrainer
from losses.clip_score import ClipScore


class Config:
    """
    Specifies the configuration for a single model.
    """
    datasets = {
        'omniglot': Omniglot,
        'cifar100': Cifar100,
        'anoshift': Anoshift,
    }

    models = {
        'omniglotCNN': OmniglotCNN,
        'omniglotDSVDD': OmniglotDSVDD,
        'cifar100DSVDD': Cifar100DSVDD,
        'cnn4DSVDD': CNN4DSVDD,
        'tabDSVDD':TabDSVDD,
        'tabNTL':TabNeutralAD,
        'cnn4cls':CNN4ClS
    }
    
    trainers = {
        'zeroshot_meta_trainer': ZeroShotMetaTrainer,
    }

    losses = {
        'center_dist': CenterDistLoss,
        'clip_score': ClipScore,
        'ntl_loss':DCL,

    }

    optimizers = {
        'Adam': Adam,
    }

    schedulers = {
        'StepLR': StepLR
    }

    early_stoppers = {
        'Patience': Patience
    }

    def __init__(self, **attrs):

        # print(attrs)
        self.config = dict(attrs)

        for attrname, value in attrs.items():
            if attrname in ['dataset', 'model', 'loss', 'optimizer', 'scheduler','early_stopper','trainer']:
                if attrname == 'dataset':
                    setattr(self, 'dataset_name', value)
                if attrname == 'model':
                    setattr(self, 'model_name', value)
                if attrname == 'loss':
                    setattr(self, 'loss_name', value)

                fn = getattr(self, f'parse_{attrname}')
                setattr(self, attrname, fn(value))
            else:
                setattr(self, attrname, value)

    def __getitem__(self, name):
        # print("attr", name)
        return getattr(self, name)

    def __contains__(self, attrname):
        return attrname in self.__dict__

    def __repr__(self):
        name = self.__class__.__name__
        return f'<{name}: {str(self.__dict__)}>'

    @property
    def exp_name(self):
        return f'{self.dataset_name}'

    @property
    def data_name(self):
        return f'{self.dataset_name}'
        
    @property
    def config_dict(self):
        return self.config

    @staticmethod
    def parse_dataset(dataset_s):
        assert dataset_s in Config.datasets, f'Could not find {dataset_s} in dictionary!'
        return Config.datasets[dataset_s]

    @staticmethod
    def parse_model(model_s):
        assert model_s in Config.models, f'Could not find {model_s} in dictionary!'
        return Config.models[model_s]

    @staticmethod
    def parse_trainer(trainer_s):
        assert trainer_s in Config.trainers, f'Could not find {trainer_s} in dictionary!'
        return Config.trainers[trainer_s]

    @staticmethod
    def parse_loss(loss_s):
        assert loss_s in Config.losses, f'Could not find {loss_s} in dictionary!'
        return Config.losses[loss_s]

    @staticmethod
    def parse_optimizer(optim_s):
        assert optim_s in Config.optimizers, f'Could not find {optim_s} in dictionary!'
        return Config.optimizers[optim_s]

    @staticmethod
    def parse_scheduler(sched_dict):
        if sched_dict is None:
            return None

        sched_s = sched_dict['class']
        args = sched_dict['args']

        assert sched_s in Config.schedulers, f'Could not find {sched_s} in schedulers dictionary'

        return lambda opt: Config.schedulers[sched_s](opt, **args)

    @staticmethod
    def parse_early_stopper(stopper_dict):
        if stopper_dict is None:
            return None

        stopper_s = stopper_dict['class']
        args = stopper_dict['args']

        assert stopper_s in Config.early_stoppers, f'Could not find {stopper_s} in early stoppers dictionary'

        return lambda: Config.early_stoppers[stopper_s](**args)

    @classmethod
    def from_dict(cls, dict_obj):
        return Config(**dict_obj)


class Grid:
    """
    Specifies the configuration for multiple models.
    """

    def __init__(self, path_or_dict, dataset_name):
        self.configs_dict = read_config_file(path_or_dict)
        self.configs_dict['dataset'] = [dataset_name]
        self.num_configs = 0  # must be computed by _create_grid
        self._configs = self._create_grid()

    def __getitem__(self, index):
        return self._configs[index]

    def __len__(self):
        return self.num_configs

    def __iter__(self):
        assert self.num_configs > 0, 'No configurations available'
        return iter(self._configs)

    def _grid_generator(self, cfgs_dict):
        keys = cfgs_dict.keys()
        result = {}

        if cfgs_dict == {}:
            yield {}
        else:
            configs_copy = deepcopy(cfgs_dict)  # create a copy to remove keys

            # get the "first" key
            param = list(keys)[0]
            del configs_copy[param]

            first_key_values = cfgs_dict[param]
            for value in first_key_values:
                result[param] = value

                for nested_config in self._grid_generator(configs_copy):
                    result.update(nested_config)
                    yield deepcopy(result)

    def _create_grid(self):
        '''
        Takes a dictionary of key:list pairs and computes all possible permutations.
        :param configs_dict:
        :return: A dictionary generator
        '''
        config_list = [cfg for cfg in self._grid_generator(self.configs_dict)]
        self.num_configs = len(config_list)
        return config_list
