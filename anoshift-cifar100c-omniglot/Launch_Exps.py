
import argparse
import os
from config_parser.base import Grid, Config
from evaluation.Experiments import runExperiment
from evaluation.Kvariants_Eval import KVariantEval

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file', default='config_anoshift.yml')
    parser.add_argument('--dataset-name', dest='dataset_name', default='anoshift')
    parser.add_argument('--qry-anomaly-ratio', dest='contamination_ratio', default=0.01, type=float)
    parser.add_argument('--ckpt-path', dest='ckpt_path', default='unspecified') #'ViT-B/32','unspecified'
    parser.add_argument('--corruption_file',dest='corruption_file',default='gaussian_noise.npy')
    return parser.parse_args()

def EndtoEnd_Experiments(model_config_file, env_config):
    '''
    Args:
      model_config_file -- model configs
      env_config -- environment configs
    '''
    model_configurations = Grid(model_config_file, env_config.dataset_name)
    model_configuration = Config(**model_configurations[0])
    print(model_configuration)
    dataset = model_configuration.dataset
    result_folder = model_configuration.result_folder + model_configuration.exp_name

    exp_path = os.path.join(result_folder)

    risk_assesser = KVariantEval(dataset, exp_path, model_configurations, env_config)

    risk_assesser.risk_assessment(runExperiment)

if __name__ == "__main__":

    args = get_args()
    config_file = 'config_files/'+ args.config_file

    EndtoEnd_Experiments(config_file, args)
