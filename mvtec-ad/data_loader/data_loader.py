import os
import torch
from data_loader.mvtec import MVTecFeature

def dataloader(dataset_name, env_config, model_config):
    if dataset_name in ['bottle', 'cable', 'capsule', 'carpet', 'grid',
           'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
           'tile', 'toothbrush', 'transistor', 'wood', 'zipper']:
        if env_config.dataset_name == 'mvtec':
            database = MVTecFeature(os.path.join('data', 'mvtec_feature_layer3', 'wide_resnet50_2'), 
                                    batchsz=model_config['batch_size'], 
                                    k_query=model_config['k_query'],
                                    args=model_config,
                                    env_args=env_config)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return database
