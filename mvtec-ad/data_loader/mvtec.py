import os
from PIL import Image
from tqdm import tqdm
# import tarfile
# import urllib.request

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


ROOT = '/extra/ucibdl0/aodong/anomaly-segmentation'


class MVTecDataset(Dataset):
    def __init__(self, dataset_path=os.path.join(ROOT,'data/'), class_name='bottle', is_train=True,
                 resize=256, cropsize=224):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        # self.download()

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)


class MVTecFeature:
    def __init__(self, root, batchsz, k_query, args=None, env_args=None):
        # root=os.path.join(ROOT,'data/mvtec_feature/wide_resnet50_2/')
        self.args = args
        self.env_args = env_args

        self.qry_anomaly_ratio = self.env_args.contamination_ratio

        self.batchsz = batchsz
        self.n_way = 2  # n way # always one-vs-rest
        self.k_query = k_query  # k query

        self.test_class_name = self.env_args.class_name
        if self.test_class_name not in CLASS_NAMES:
            raise NotImplementedError('test class not in dataset')
        self.test_class_id = CLASS_NAMES.index(self.test_class_name)
        print(self.test_class_name)

        self.x_train = []
        for i, class_name in enumerate(CLASS_NAMES):
            if class_name != self.test_class_name:
                self.x_train.append(torch.load(os.path.join(root, 'train_%s.pt' % class_name)))
        # self.train_all_abnormal_data = torch.cat(self.x_train, 0)
        # print(self.train_all_abnormal_data.shape)

        self.x_test = torch.load(os.path.join(root, 'test_%s.pt' % self.test_class_name))
        self.x_test_all_abnormal_data = [] # used as manual outliers
        for i, class_name in enumerate(CLASS_NAMES):
            if class_name != self.test_class_name:
                self.x_test_all_abnormal_data.append(torch.load(os.path.join(root, 'test_%s.pt' % class_name)))
        self.x_test_all_abnormal_data = torch.cat(self.x_test_all_abnormal_data, 0)
        # print(self.x_test_all_abnormal_data.shape)

        self.test_gt_list = torch.load(os.path.join(root, 'test_%s_gt.pt' % self.test_class_name))
        self.test_gt_mask_list = torch.load(os.path.join(root, 'test_%s_gt_mask.pt' % self.test_class_name))
        self.test_image_num = len(self.x_test)

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0, 'val': 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", len(self.x_train), "test", len(self.x_test))

        # # cache abnormal data to speedup
        self.train_all_abnormal_data = self.cache_abnormal_data(self.datasets["train"])
        # self.test_all_abnormal_data = self.cache_abnormal_data(self.datasets["test"])
        # self.val_all_abnormal_data = self.cache_abnormal_data(self.datasets["val"])

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"], meta_split='train'),
                               "test": self.load_data_cache(self.datasets["test"], meta_split='test')}
                               # "val": self.load_data_cache(self.datasets["test"], meta_split='val')}

        # # sanity check
        # self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"], meta_split='train'),
        #                        "test": self.load_data_cache(self.datasets["train"], meta_split='train')}
        #                        # "val": self.load_data_cache(self.datasets["test"], meta_split='val')}

    def load_data_cache(self, data_pack, meta_split='train'):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
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


    def cache_abnormal_data(self, data_pack):
        tot_cls = len(data_pack)

        all_abnormal_data = []
        for selected_normal_cls in np.arange(tot_cls):
            abnormal_cls = np.arange(tot_cls)
            mask = np.ones(tot_cls, dtype=bool)
            mask[selected_normal_cls] = False
            abnormal_cls = abnormal_cls[mask]
            _all_abnormal_data = [data_pack[_cls] for _cls in abnormal_cls]
            _all_abnormal_data = torch.cat(_all_abnormal_data, 0)
            all_abnormal_data.append(_all_abnormal_data)

        return all_abnormal_data


    def _load_data_cache_train(self, data_pack):
        querysz = self.k_query * self.n_way
        qry_anomaly_ratio = self.qry_anomaly_ratio
        tot_cls = len(data_pack)
        
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_qrys, y_qrys = [], []
            for i in range(self.batchsz):  # one batch means one set
                selected_normal_cls = np.random.choice(tot_cls, 1, False)[0]
                class_size, num_channel, num_patch = data_pack[selected_normal_cls].shape

                x_qry, y_qry = [], []

                num_qry_abnormal = int(querysz*qry_anomaly_ratio)
                if num_qry_abnormal == 0:
                    num_qry_abnormal = 1 
                num_qry_normal = querysz - num_qry_abnormal

                # normal data
                selected_img = np.random.choice(class_size, num_qry_normal, False)
                selected_patch = np.random.choice(num_patch, 1, False)[0]
                x_qry.append(data_pack[selected_normal_cls][selected_img, :, selected_patch])
                y_qry.append(torch.zeros(num_qry_normal))

                # outlier: random Gaussian noise corrupted feature sets
                normal_data = x_qry[-1].detach()[:num_qry_abnormal]
                # mu = torch.mean(normal_data, 0)
                # target_snr = torch.tensor(1.)
                # noise_scale = mu / torch.sqrt(target_snr)
                # noise = noise_scale*torch.randn(*normal_data.shape)
                noise = 0.1*torch.randn(*normal_data.shape)
                abnormal_data = normal_data + noise
                x_qry.append(abnormal_data)
                y_qry.append(torch.ones(num_qry_abnormal))

                # shuffle inside a batch
                perm = np.random.permutation(querysz)
                x_qry = torch.cat(x_qry, 0).reshape(querysz, num_channel)[perm]
                y_qry = torch.cat(y_qry, 0).reshape(querysz)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            x_qrys = torch.cat(x_qrys, 0).float().reshape(self.batchsz, querysz, -1)
            y_qrys = torch.cat(y_qrys, 0).int().reshape(self.batchsz, querysz)

            data_cache.append([x_qrys, y_qrys])

        return data_cache


    def _load_data_cache_test(self, data_pack):
        qry_anomaly_ratio = self.qry_anomaly_ratio
    
        x_qrys, y_qrys = [], []

        class_size, num_channel, num_patch = data_pack.shape

        fixed_abnormal_set = None

        for patch in range(num_patch):
            num_qry_normal = class_size
            num_qry_abnormal = int(num_qry_normal*qry_anomaly_ratio / (1-qry_anomaly_ratio))
            if num_qry_abnormal == 0:
                num_qry_abnormal = 1 
            querysz = num_qry_normal + num_qry_abnormal

            x_qry, y_qry = [], []

            # normal data
            selected_normal_patch = patch
            x_qry.append(data_pack[..., selected_normal_patch])
            y_qry.append(torch.zeros(num_qry_normal))

            # outlier
            normal_data = x_qry[-1].detach()[:num_qry_abnormal]
            # mu = torch.mean(normal_data, 0)
            # target_snr = torch.tensor(1.)
            # noise_scale = mu / torch.sqrt(target_snr)
            # noise = noise_scale*torch.randn(*normal_data.shape)
            noise = 0.1*torch.randn(*normal_data.shape)
            abnormal_data = normal_data + noise
            x_qry.append(abnormal_data)
            y_qry.append(torch.ones(num_qry_abnormal))

            x_qry = torch.cat(x_qry, 0)
            y_qry = torch.cat(y_qry, 0)

            # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)

        return [x_qrys, y_qrys]


if __name__ == '__main__':

    # db = CIFAR100NShotAD('db/cifar100', batchsz=20, k_shot=5, k_query=15, imgsz=32)

    import  time
    import  torch
    import  visdom

    class env_config:
        contamination_ratio = 0.5
        class_name = 'bottle'

    viz = visdom.Visdom(env='mvtec_feat_view')

    db = MVTecFeature(os.path.join(ROOT,'data/mvtec_feature_layer3/wide_resnet50_2/'), batchsz=20, k_query=15, env_args=env_config)

    for i in range(1000):
        x_qry, y_qry = db.next('train')

        batchsz, setsz, c = x_qry.size()
        print(batchsz, setsz, c)
        # x_qry = x_qry.reshape()

        # print(x_qry[0])

        viz.heatmap(x_qry[0], win='x_qry', opts=dict(title='x_qry'))
        viz.text(str(y_qry[0]), win='y_qry', opts=dict(title='y_qry'))


        time.sleep(10)