import os
import json
import torch
import random
import numpy as np
from utils import  Logger

class KVariantEval:

    def __init__(self, dataset, exp_path, model_configs, env_configs):
        self.data_name = dataset.data_name
        self.env_configs = env_configs
        self.model_configs = model_configs

        self._NESTED_FOLDER = exp_path
        if self.env_configs.dataset_name == 'anoshift':
            self._FOLD_BASE = str(env_configs.contamination_ratio)
        else:
            self._FOLD_BASE = ''
        self._RESULTS_FILENAME = 'results.json'
        self._ASSESSMENT_FILENAME = 'assessment_results.json'

    def process_results(self):

        assessment_results = {}

        avg_eval_res = []

        try:
            config_filename = os.path.join(self._NESTED_FOLDER, self._FOLD_BASE,
                                           self._RESULTS_FILENAME)

            with open(config_filename, 'r') as fp:
                variant_scores = json.load(fp)
                eval_res = np.array(variant_scores['eval_res'])

            assessment_results['eval_res'] = eval_res.tolist() # (num_repeat, num_metric)
            # avg_eval_res.append(eval_res)

        except Exception as e:
            print(e)

        avg_eval_res = eval_res.mean(0) # average over classes
        std_eval_res = eval_res.std(0)
        assessment_results['avg_eval_res'] = avg_eval_res.tolist()
        assessment_results['std_eval_res'] = std_eval_res.tolist()
        try:
            assessment_results['avg_eval_res_all'] = eval_res.mean(1).mean(0).tolist()
            assessment_results['std_eval_res_all'] = eval_res.mean(1).std(0).tolist()
            assessment_results['avg_eval_near'] = eval_res[:,5:8].mean(1).mean(0).tolist()
            assessment_results['std_eval_near'] = eval_res[:,5:8].mean(1).std(0).tolist()
            assessment_results['avg_eval_far'] = eval_res[:,8:].mean(1).mean(0).tolist()
            assessment_results['std_eval_far'] = eval_res[:,8:].mean(1).std(0).tolist()
            assessment_results['avg_eval_ood'] = eval_res[:,5:].mean(1).mean(0).tolist()
            assessment_results['std_eval_ood'] = eval_res[:,5:].mean(1).std(0).tolist()
        except:
            None

        print(assessment_results)

        with open(os.path.join(self._NESTED_FOLDER,  self._FOLD_BASE, self._ASSESSMENT_FILENAME), 'w') as fp:
            json.dump(assessment_results, fp, indent=0)

    def risk_assessment(self, experiment_class):
            
        if not os.path.exists(self._NESTED_FOLDER):
            os.makedirs(self._NESTED_FOLDER)

        folder = os.path.join(self._NESTED_FOLDER, self._FOLD_BASE)
        if not os.path.exists(folder):
            os.makedirs(folder)

        if self.env_configs.ckpt_path != 'unspecified':
            self._ckpt_risk_assessment(experiment_class, folder)
            self.process_results()
            return

        json_results = os.path.join(folder, self._RESULTS_FILENAME)
        if not os.path.exists(json_results):
            self._risk_assessment_helper(experiment_class, folder)
        else:
            print(
                f"File {json_results} already present! Shutting down to prevent loss of previous experiments")

        self.process_results()

    def _risk_assessment_helper(self, experiment_class, exp_path):

        best_config = self.model_configs[0]
        experiment = experiment_class(best_config, self.env_configs, exp_path)

        logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')

        eval_metrics_list = []
        num_repeat = best_config['num_repeat']

        saved_results = {}
        for i in range(num_repeat):
            torch.cuda.empty_cache()
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(i + 42)
            random.seed(i + 42)
            torch.manual_seed(i + 42)
            torch.cuda.manual_seed(i + 42)
            torch.cuda.manual_seed_all(i + 42)
            
            eval_metrics = experiment.run(self.data_name, logger, repeat=i)
            print(f'Final training run {i}: {eval_metrics}')

            eval_metrics_list.append(eval_metrics)
            # if best_config['save_scores']:
            #     saved_results['scores_'+str(i)] = test_score.tolist()

        if best_config['save_scores']:
            save_path = os.path.join(self._NESTED_FOLDER, self._FOLD_BASE, 'scores_labels.json')
            json.dump(saved_results, open(save_path, 'w'))
            # saved_results = json.load(open(save_path))

        message = f'End of Assessment:\n\t mean: {np.array(eval_metrics_list).mean(0)}, std: {np.array(eval_metrics_list).std(0)}'
        if logger is not None:
            logger.log(message)
        print(message)

        with open(os.path.join(exp_path, self._RESULTS_FILENAME), 'w') as fp:
            json.dump({
                        'best_config': best_config, 
                        'eval_res': eval_metrics_list,
                      }, fp)


    def _ckpt_risk_assessment(self, experiment_class, exp_path):
        best_config = self.model_configs[0]
        experiment = experiment_class(best_config, self.env_configs, exp_path)
        # database = load_data(self.data_name, best_config, self.env_configs)

        eval_metrics_list = []
        num_repeat = best_config['num_repeat']

        for i in range(num_repeat):    
            eval_metrics = experiment.run(self.data_name, logger=None)
            eval_metrics_list.append(eval_metrics)
        with open(os.path.join(exp_path, self._RESULTS_FILENAME), 'w') as fp:
            json.dump({
                        'best_config': best_config, 
                        'eval_res': eval_metrics_list,
                      }, fp)