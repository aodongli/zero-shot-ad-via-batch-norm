from torchvision.datasets import CIFAR100 as CIFAR100Download
import torchvision.transforms as transforms
from PIL import Image
import os.path
import numpy as np
import torch

stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))

class CIFAR100C:
    '''
    meta-tr (64):
        normal + abnormal 
    meta-val (16):
        normal + abnormal 
    meta-ts (20):
        normal + abnormal 
    train/val/test split follows:
      Bertinetto L., Henriques J. F., Torr P. H.S., Vedaldi A. (2019).
      Meta-learning with differentiable closed-form solvers. In International
      Conference on Learning Representations (https://arxiv.org/abs/1805.08136)
    '''

    def __init__(self, root, batchsz, k_query, args=None, env_args=None):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param k_qry:
        """

        if env_args is None:
            self.qry_anomaly_ratio = 0.2
            self.corruption_file = 'gaussian_noise.npy'
        else:
            self.qry_anomaly_ratio = env_args.contamination_ratio
            self.corruption_file = env_args.corruption_file
        print(self.corruption_file)

        data_transform = transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize(*stats)
                                            ])

        if not os.path.isfile(os.path.join(root, 'cifar100_train.pt')): # cifar100_no_norm.pt
            # if root/data.pt does not exist, just download it
            
            train_dataset = CIFAR100Download(download=True, root=root, transform=data_transform)
            temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
            for (img, label) in train_dataset:
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label] = [img]

            self.x = []
            labels = []
            for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
                self.x.append(torch.stack(imgs, 0))
                labels.append(label)

            # as different class may have different number of imgs
            self.x = torch.stack(self.x, 0)  # [[20 imgs],..., 1623 classes in total]
            print('data shape:', self.x.shape)
            temp = []  # Free memory
            # save all dataset into pt file.
            torch.save(self.x, os.path.join(root, 'cifar100_train.pt'))
            np.save(os.path.join(root, 'cifar100_labels.npy'), labels)
            print('write into cifar100_train.pt.')
        else:
            # if data.pt exists, just load it.
            self.x = torch.load(os.path.join(root, 'cifar100_train.pt'))
            print('load from cifar100_train.pt.')

        self.x_train  = self.x

        test_dataset = np.load(os.path.join(root, 'CIFAR-100-C', self.corruption_file))
        test_dataset = test_dataset[40000:] # take the most corrupted version
        test_label = np.load(os.path.join(root, 'CIFAR-100-C/labels.npy'))
        test_label = test_label[40000:]

        temp = dict()
        for (img, label) in zip(test_dataset, test_label):
            img = data_transform(img)
            if label in temp.keys():
                temp[label].append(img)
            else:
                temp[label] = [img]

        self.x_test = [torch.stack(temp[label], 0) for label in range(100)]
        self.x_test = torch.stack(self.x, 0)
        self.x_val = self.x_test # temporary split
    
        self.batchsz = batchsz
        self.n_cls = self.x.shape[0]
        self.n_way = 2  # n way # always one-vs-rest
        self.k_query = k_query  # k query
        assert self.k_query <= 600

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0, 'val': 0}
        self.datasets = {"train": self.x_train, "test": self.x_test, 'val': self.x_val}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape, 'val', self.x_val.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"], meta_split='train'),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"], meta_split='test'),
                               "val": self.load_data_cache(self.datasets["val"], meta_split='val')}



    def _load_data_cache_train(self, data_pack):
        querysz = self.k_query * self.n_way
        
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_qrys, y_qrys = [], []
            for i in range(self.batchsz):  # one batch means one set

                x_qry, y_qry = [], []

                # normal data
                selected_normal_cls = np.random.choice(data_pack.shape[0], 1, replace=False)[0]
                selected_img = np.random.choice(100, self.k_query, False)
                x_qry.append(data_pack[selected_normal_cls][selected_img])
                y_qry.append([0 for _ in range(self.k_query)])

                # outlier 
                abnormal_cls = np.arange(data_pack.shape[0])
                mask = np.ones(data_pack.shape[0], dtype=bool)
                mask[selected_normal_cls] = False
                abnormal_cls = abnormal_cls[mask]
                selected_img_cls = np.random.choice(abnormal_cls, self.k_query, replace=True)
                selected_img_id = np.random.choice(100, self.k_query, replace=True)
                x_qry.append(data_pack[(selected_img_cls, selected_img_id)])
                y_qry.append([1 for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(querysz)
                x_qry = torch.cat(x_qry, 0).reshape(querysz, 3, 32, 32)[perm]
                y_qry = torch.tensor(y_qry).reshape(querysz)[perm]

                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            x_qrys = torch.cat(x_qrys, 0).float().reshape(self.batchsz, querysz, 3, 32, 32)
            y_qrys = torch.cat(y_qrys, 0).int().reshape(self.batchsz, querysz)

            data_cache.append([x_qrys, y_qrys])

        return data_cache


    def _load_data_cache_test(self, data_pack):
        class_size = data_pack.shape[1]
        batch_size = data_pack.shape[0]
        qry_anomaly_ratio = self.qry_anomaly_ratio
        num_qry_normal = class_size
        num_qry_abnormal = int(num_qry_normal*qry_anomaly_ratio / (1-qry_anomaly_ratio))
        if num_qry_abnormal == 0:
            num_qry_abnormal = 1 
        querysz = num_qry_normal + num_qry_abnormal

        x_qrys, y_qrys = [], []

        for i in range(batch_size):  # one batch means one set

            x_qry, y_qry = [], []

            # normal data
            selected_normal_cls = i
            x_qry.append(data_pack[selected_normal_cls])
            y_qry.append(torch.zeros(num_qry_normal))

            # outlier 
            abnormal_cls = np.arange(data_pack.shape[0])
            mask = np.ones(data_pack.shape[0], dtype=bool)
            mask[selected_normal_cls] = False
            abnormal_cls = abnormal_cls[mask]
            selected_img_cls = np.random.choice(abnormal_cls, num_qry_abnormal, replace=True)
            selected_img_id = np.random.choice(100, num_qry_abnormal, replace=True)
            x_qry.append(data_pack[(selected_img_cls, selected_img_id)])
            y_qry.append(torch.ones(num_qry_abnormal))

            # shuffle inside a batch
            perm = np.random.permutation(querysz)
            x_qry = torch.cat(x_qry, 0).reshape(querysz, 3, 32, 32)[perm]
            y_qry = torch.cat(y_qry, 0).reshape(querysz)[perm]

            # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)

        x_qrys = torch.cat(x_qrys, 0).float().reshape(batch_size, querysz, 3, 32, 32)
        y_qrys = torch.cat(y_qrys, 0).int().reshape(batch_size, querysz)

        return [x_qrys, y_qrys]


    def load_data_cache(self, data_pack, meta_split='train'):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        if meta_split == 'train':
            return self._load_data_cache_train(data_pack)

        elif meta_split == 'test':
            return self._load_data_cache_test(data_pack)

        elif meta_split == 'val':
            return self._load_data_cache_test(data_pack)

        else:
            raise NotImplementedError()
        

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        if mode == 'train':
            # update cache if indexes is larger cached num
            if self.indexes[mode] >= len(self.datasets_cache[mode]):
                self.indexes[mode] = 0
                self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode], meta_split=mode)

            next_batch = self.datasets_cache[mode][self.indexes[mode]]
            self.indexes[mode] += 1

            return next_batch
        elif mode == 'test' or mode == 'val':
            return self.datasets_cache[mode]
        else:
            raise NotImplementedError()