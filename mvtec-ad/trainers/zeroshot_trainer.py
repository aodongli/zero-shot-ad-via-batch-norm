import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter
import numpy as np

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

        x_qry, y_qry = train_loader.next()
        x_qry, y_qry = x_qry.to(self.device), y_qry.to(self.device)

        oc_loss = 'oc_query' in self.config and self.config['oc_query']

        # task_num, querysz, c, h, w = x_qry.size()
        task_num = x_qry.shape[0]
        loss = 0
        for i in range(task_num): # cannot parallel, otherwise the nice majority dominance property diminishes
            x = x_qry[i]
            # print("mean:", torch.mean(x[y_qry[i] == 0], 0))
            # print("std:", torch.std(x[y_qry[i] == 0], 0))
            # print("snr:", torch.square(torch.mean(x[y_qry[i] == 0], 0)/torch.std(x[y_qry[i] == 0], 0)))
            z, center = self.model(x)
            dist = self.loss_fun(z, center)
            if not oc_loss:
                loss_i = torch.cat([dist[y_qry[i] == 0], 1/dist[y_qry[i] == 1]], 0)
            else:
                loss_i = dist[y_qry[i] == 0]
            loss_i = loss_i.mean()

            loss += loss_i

        loss /= task_num
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss


    def detect_outliers(self, loader, allowed_task_num=32, mode='test', save_feat=False):

        auc = 0
        loss = 0

        x_qry, y_qry = loader.next(mode=mode)
        # x_qry, y_qry = x_qry.to(self.device), y_qry.to(self.device)
        x_qry = [x.to(self.device) for x in x_qry]
        y_qry = [y.to(self.device) for y in y_qry]
        task_num_tot = len(x_qry) # 196

        if allowed_task_num > 0:
            task_num = min(task_num_tot, allowed_task_num)
            task_num_list = np.random.choice(task_num_tot, task_num, replace=False)
        else:
            task_num_list = range(task_num_tot)
            task_num = task_num_tot

        zs = []
        labels = []
        center = None

        oc_loss = 'oc_query' in self.config and self.config['oc_query']

        score_list = []
        height = int(np.sqrt(task_num_tot))
        batch_size = loader.test_image_num
        with torch.no_grad():
            for i in task_num_list:
                z, center = self.model(x_qry[i])
                dist = self.loss_fun(z, center)
                scores = dist.cpu().numpy()

                score_list.append(dist[y_qry[i] == 0])

                # scores 
                if not oc_loss:
                    loss_i = torch.cat([dist[y_qry[i] == 0], 1/dist[y_qry[i] == 1]], 0).mean()
                else:
                    loss_i = torch.mean(dist[y_qry[i] == 0])
                loss += loss_i
                # auc
                try:
                    # doesn't use this value in practice
                    _auc = roc_auc_score(y_qry[i].cpu().numpy(), scores)
                    auc += _auc
                except:
                    pass

                zs.append(z.cpu().numpy())
                labels.append(y_qry[i].cpu().numpy())

        score_list = torch.stack(score_list, 1)
        score_list = score_list.reshape(batch_size, height, height)
        score_map = F.interpolate(score_list.unsqueeze(1), size=224, mode='bilinear',
                                  align_corners=False).squeeze().cpu().numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)

        gt_list = loader.test_gt_list
        gt_mask = loader.test_gt_mask_list
        
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        # fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        msg = 'image ROCAUC: %.3f' % (img_roc_auc)
        if logger is not None:
            logger.log(msg)
            print(msg)
        else:
            print(msg)
        # fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))

        # calculate per-pixel level ROCAUC
        # fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        msg = 'pixel ROCAUC: %.3f' % (per_pixel_rocauc)
        if logger is not None:
            logger.log(msg)
            print(msg)
        else:
            print(msg)

        # investigate per-image anomaly segmentation
        # print(roc_auc_score(gt_mask[gt_list==1][0].flatten(), scores[gt_list==1][0].flatten()))

        if save_feat:
            center = center.cpu().detach().numpy()
            np.savez(self.exp_path + '/features.npz', 
                     zs=zs, 
                     labels=labels, 
                     center=center)

            print(f'features saved to {self.exp_path}/features.npz')

        # print(center)

        # return loss/task_num, auc/task_num
        return img_roc_auc, per_pixel_rocauc


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
            if 'test' in test_loader.datasets_cache:
                test_score, test_auc = self.detect_outliers(test_loader, 
                                                            allowed_task_num=-1,
                                                            # allowed_task_num=2,  # uncomment to speedup training
                                                            mode='test')
            else:
                msg = 'test not exists.'
                if logger is not None:
                    logger.log(msg)
                print(msg)
                raise NotImplementedError(msg)

            if 'val' in validation_loader.datasets_cache:
                val_score, val_auc = self.detect_outliers(validation_loader, 
                                                          allowed_task_num=-1,
                                                          # allowed_task_num=1,  # uncomment to speedup training
                                                          mode='val')
            else:
                val_score = test_score
                val_auc = test_auc

            if early_stopper is not None and early_stopper.stop(iteration, train_loss, val_score, [test_auc]):
                break

            if iteration % config['log_every'] == 0 or iteration == 1:
                msg = f'Iteration: {iteration}, TR loss: {train_loss}, VAL loss: {val_score}, VL auc: {val_auc}, TS loss: {test_score}, TS auc: {test_auc}'

                if logger is not None:
                    logger.log(msg)
                    print(msg)
                else:
                    print(msg)

                # save model
                if iteration >= 200:
                    torch.save(self.model.state_dict(), self.exp_path + f'/ckpt_{iteration}.pt')

        if early_stopper is not None:
            best_loss, eval_metrics, best_iteration = early_stopper.get_best_vl_metrics()
            msg = f'Stopping at iteration {best_iteration}, TR loss: {train_loss}, VAL loss: {val_score}, TS: {eval_metrics.res[-1]}'
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
            test_score, test_auc = self.detect_outliers(test_loader, 
                                                        allowed_task_num=-1,
                                                        mode='test',
                                                        save_feat=False)
        else:
            msg = 'test not exists.'
            if logger is not None:
                logger.log(msg)
            print(msg)
            raise NotImplementedError(msg)

        print(test_score, test_auc)