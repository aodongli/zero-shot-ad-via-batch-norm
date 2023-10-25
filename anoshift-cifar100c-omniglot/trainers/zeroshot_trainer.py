
""" This code is shared for review purposes only. Do not copy, reproduce, share, publish,
or use for any purpose except to review our submission. Please delete after the review process.
The authors plan to publish the code deanonymized and with a proper license upon publication of the paper. """

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import time


class ZeroShotMetaTrainer:
    def __init__(self, model, loss_function, exp_path, config):

        self.loss_fun = loss_function
        self.config = config
        self.device = torch.device(self.config['device'])
        self.model = model.to(self.device)

        self.max_iterations = config['max_iterations']

        self.exp_path = exp_path

    def _train(self, epoch, train_loader, optimizer):

        self.model.train()

        # x_qry, y_qry,normal_class = train_loader.next()
        x_qry, y_qry = train_loader.next()
        # x_qry, y_qry = x_qry.to(self.device), y_qry.to(self.device)

        task_num= len(x_qry)
        loss = 0
        for i in range(task_num): # cannot parallel, otherwise the nice majority dominance property diminishes
            x = x_qry[i].to(self.device)
            y = y_qry[i].to(self.device)
            z, center = self.model(x)
            ln,la = self.loss_fun(z, center)
            # loss_i = torch.cat([dist[y == 0], 1/dist[y == 1]], 0).mean()
            loss_i = torch.cat([ln[y == 0], la[y == 1]], 0).mean()


            loss += loss_i

        loss /= task_num
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss


    def detect_outliers(self, loader, allowed_task_num=-1, mode='test'):

        auc = []
        loss = 0

        # x_qry, y_qry,normal_class = loader.next(mode=mode)
        x_qry, y_qry = loader.next(mode=mode)
        # x_qry, y_qry = x_qry.to(self.device), y_qry.to(self.device)
        task_num = len(x_qry)
        if allowed_task_num > 0:
            task_num = min(task_num, allowed_task_num)

        with torch.no_grad():
            for i in range(task_num):
                x = x_qry[i].to(self.device)
                y = y_qry[i].to(self.device)
                if len(x)<=1000:
                    z, center = self.model(x)
                    ln,la = self.loss_fun(z, center)
                else:
                    start_idx= 0
                    ln,la = [],[]
                    batch_size = 1000
                    for k in range(int(np.ceil(len(x)/batch_size))):
                        start_time = time.time()
                        x_sub = x[start_idx:start_idx+batch_size]
                        z_sub, center = self.model(x_sub)
                        start_idx= start_idx+batch_size
                        ln_sub,la_sub = self.loss_fun(z_sub, center)
                        ln.append(ln_sub)
                        la.append(la_sub)
                        end_time = time.time()
                        # if k==0:
                        #     print("--- %s seconds ---" % (end_time - start_time))
                    ln = torch.cat(ln,0)
                    la = torch.cat(la,0)
                
                scores = ln.cpu().numpy()
                # scores 
                loss += torch.cat([ln[y == 0], la[y == 1]], 0).mean()
                # loss+= RankLoss(dist[y == 1],dist[y == 0])
                # torch.cat([dist[y == 0], -torch.log(1-torch.exp(-dist[y == 1]))], 0).mean()
                # auc
                _auc = roc_auc_score(y.cpu().numpy(), scores)
                auc.append(_auc)
                # print(f"year {normal_class[i]} auc: {_auc}")

        return loss/task_num, auc


    def train(self, 
              train_loader, 
              optimizer=None, 
              scheduler=None,
              validation_loader=None, 
              test_loader=None, 
              early_stopping=None, 
              logger=None,
              config=None,
              env_config=None):

        early_stopper = early_stopping() if early_stopping is not None else None

        val_auc, val_f1, = -1, -1
        test_auc, test_f1, test_score = None, None,None,

        if 'test' not in test_loader.datasets_cache:
            msg = 'test not exists in dataloader.'
            if logger is not None:
                logger.log(msg)
            print(msg)
        if 'val' not in validation_loader.datasets_cache:
            msg = 'val not exists in dataloader. Use test evaluations instead.'
            if logger is not None:
                logger.log(msg)
            print(msg)

        for iteration in range(1, self.max_iterations+1):

            train_loss = self._train(iteration, train_loader, optimizer)

            if scheduler is not None:
                scheduler.step()

            # TODO: clean the returned values
            if iteration % config.log_every == 0 and iteration>10:
                if 'test' in test_loader.datasets_cache:
                    test_score, test_auc = self.detect_outliers(test_loader, 
                                                                # allowed_task_num=2,  # uncomment to speedup training
                                                                mode='test')
                    test_auc = np.mean(test_auc)


                if 'val' in validation_loader.datasets_cache:
                    val_score, val_auc = self.detect_outliers(validation_loader, 
                                                                # allowed_task_num=1,  # uncomment to speedup training
                                                                mode='val')
                    val_auc = np.mean(val_auc)
                else:
                    val_score = test_score
                    val_auc = test_auc

                best_val_auc, _,_ = early_stopper.get_best_vl_metrics()

                if val_auc>=best_val_auc:
                    torch.save(self.model.state_dict(), self.exp_path + f'/model_bestval.pt')
                if early_stopper is not None and early_stopper.stop(iteration, train_loss, val_auc, test_auc):
                    break

            # if iteration % config.log_every == 0 or iteration == 1:
                msg = f'Iteration: {iteration}, TR loss: {train_loss}, VAL loss: {val_score}, VL auc: {val_auc}, TS loss: {test_score}, TS auc: {test_auc}'

                if logger is not None:
                    logger.log(msg)
                    print(msg)
                else:
                    print(msg)

                # save model
                if iteration >= 500 and iteration % 100 == 0:
                    torch.save(self.model.state_dict(), self.exp_path + f'/ckpt_{iteration}.pt')

        if early_stopper is not None:
            best_val_auc, eval_metrics, best_iteration = early_stopper.get_best_vl_metrics()
            msg = f'Stopping at iteration {best_iteration},  VAL acc: {best_val_auc}, TS: {eval_metrics.res[-1]}'
            if logger is not None:
                logger.log(msg)
                print(msg)
            else:
                print(msg)

        return eval_metrics.res[-1] if early_stopper is not None else [test_auc]


    def test(self, 
             test_loader,
             config=None,
             env_config=None):
        
        self.model.load_state_dict(torch.load(env_config.ckpt_path, 
                                              map_location=self.device))
        self.model.train()

        test_score, test_auc = 0, 0

        print('test checkpoint')

        if 'test' in test_loader.datasets_cache:
            # start_time = time.time()
            test_score, test_auc = self.detect_outliers(test_loader, 
                                                        allowed_task_num=-1,
                                                        mode='test')

        else:
            msg = 'test not exists.'

            raise NotImplementedError(msg)

        return test_auc