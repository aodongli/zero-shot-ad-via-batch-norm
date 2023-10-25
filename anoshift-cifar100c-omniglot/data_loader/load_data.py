import os
from scipy import io
import torch
import numpy as np

from .cifar100_c import CIFAR100C
from .omniglot import Omniglot
from .anoshift import Anoshift


def load_data(data_name, args=None, env_args=None,data_transform=None):
    if data_name == 'cifar100':
        database = CIFAR100C(os.path.join('./data', 'cifar100'), 
                            batchsz=args['batch_size'], 
                            k_query=args['k_query'],
                            args=args,
                            env_args=env_args,
                            data_transform=data_transform)
 
    elif data_name == 'omniglot':
        database = Omniglot(os.path.join('./data', 'omniglot'), 
                            batchsz=args['batch_size'], 
                            k_query=args['k_query'],
                            args=args,
                            env_args=env_args)
    elif data_name == 'anoshift':
        database = Anoshift(os.path.join('./data', 'anoshift'), 
                            batchsz=args['batch_size'], 
                            k_query=args['k_query'],
                            args=args,
                            env_args=env_args)

    else:
        raise NotImplementedError()

    return database






