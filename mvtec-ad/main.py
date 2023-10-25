import argparse
import os
import numpy as np
from data_loader.data_loader import dataloader
from config.parser import model_config_reader
from util import Logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file', default='config_mvtec.yml')
    parser.add_argument('--dataset-name', dest='dataset_name', default='mvtec')
    parser.add_argument('--class-name', dest='class_name', default='all')
    parser.add_argument('--qry-anomaly-ratio', dest='contamination_ratio', default=0.5, type=float)
    parser.add_argument('--ckpt-path', dest='ckpt_path', default='unspecified')
    return parser.parse_args()


def run_dataset(dataset, env_config, model_config):
    exp_path = os.path.join(model_config['result_folder'], env_config.dataset_name)
    exp_path = os.path.join(exp_path, env_config.class_name) if env_config.dataset_name == 'mvtec' else exp_path
    os.makedirs(exp_path, exist_ok=True)
    logger = Logger(str(os.path.join(exp_path, 'experiment.log')), mode='a')

    model_class = model_config['model_class']
    model = model_class(config=model_config)
    loss_class = model_config['loss_class']
    loss = loss_class(config=model_config)
    optimizer_class = model_config['optimizer_class']
    optimizer = optimizer_class(model.parameters(),
                                lr=model_config['learning_rate'], 
                                weight_decay=model_config['l2'])
    scheduler_class = model_config['scheduler_class']
    scheduler = scheduler_class(optimizer)
    trainer_class = model_config['trainer_class']
    trainer = trainer_class(model, loss, exp_path, model_config)

    if env_config.ckpt_path != 'unspecified':
        trainer.test(test_loader=dataset,
                     config=model_config,
                     env_config=env_config)
        return

    eval_metrics = trainer.train(train_loader=dataset,
                                  optimizer=optimizer, 
                                  scheduler=scheduler,
                                  validation_loader=dataset, 
                                  test_loader=dataset, 
                                  early_stopping=None,
                                  logger=logger,
                                  config=model_config,
                                  env_config=env_config)
    return eval_metrics


if __name__ == "__main__":
    env_config = get_args()
    model_config = model_config_reader(env_config.config_file)

    if env_config.dataset_name == 'mvtec' or env_config.dataset_name == 'mvtec_avgpool':
        if env_config.class_name == 'all':
            res_list = []
            dataset_name_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
            # dataset_name_list = ['cardio']
            for dataset_name in dataset_name_list:
                env_config.class_name = dataset_name
                dataset = dataloader(dataset_name, env_config, model_config)
                res = run_dataset(dataset, env_config, model_config)
                res_list.append(res)
                # print(np.array(res_list))
            res_list = np.array(res_list)
            print(res_list)
            print(res_list.mean(0))
        else:
            dataset_name = env_config.class_name
            dataset = dataloader(dataset_name, env_config, model_config)
            res = run_dataset(dataset, env_config, model_config)
    else:
        dataset = dataloader(env_config.dataset_name, env_config, model_config)
        res = run_dataset(dataset, env_config, model_config)
