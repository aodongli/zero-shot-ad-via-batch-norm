import os
from config_parser.base import Config
import numpy as np
from data_loader.load_data import load_data
class runExperiment():

    def __init__(self, model_configuration, env_configs, exp_path):
        self.model_config = Config.from_dict(model_configuration)
        self.env_configs = env_configs
        self.exp_path = exp_path

    def run(self, data_name, logger, repeat=0):
        if self.model_config.model_name == 'clip':
            model_class = self.model_config.model
            model, data_transform = model_class(self.env_configs.ckpt_path)
 
        else:
            model_class = self.model_config.model
            optim_class = self.model_config.optimizer
            sched_class = self.model_config.scheduler
            stopper_class = self.model_config.early_stopper
            # shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True
            model = model_class(config=self.model_config)
            optimizer = optim_class(model.parameters(),
                                    lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])
            if sched_class is not None:
                scheduler = sched_class(optimizer)
            else:
                scheduler = None
            data_transform= None
            
        loss_class = self.model_config.loss
        trainer_class = self.model_config.trainer
        

        loss = loss_class(config=self.model_config)


        exp_path = self.exp_path
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        trainer = trainer_class(model, loss_function=loss,
                                exp_path=exp_path,
                                config=self.model_config)        
        eval_metrics = []
        if self.env_configs.ckpt_path != 'unspecified':

            database = load_data(data_name, self.model_config.config, self.env_configs,data_transform=data_transform)

            auroc = trainer.test(test_loader=database,
                        config=self.model_config,
                        env_config=self.env_configs)
            
            # print('mean auc:',np.mean(eval_metrics))
            # return np.mean(eval_metrics)
        else:

            database = load_data(data_name, self.model_config.config, self.env_configs,data_transform=data_transform)

            auroc = trainer.train(train_loader=database,
                                        optimizer=optimizer, 
                                        scheduler=scheduler,
                                        validation_loader=database, 
                                        test_loader=database, 
                                        early_stopping=stopper_class,
                                        logger=logger,
                                        config=self.model_config,
                                        env_config=self.env_configs)
        eval_metrics.append(auroc)

        return np.mean(eval_metrics)
