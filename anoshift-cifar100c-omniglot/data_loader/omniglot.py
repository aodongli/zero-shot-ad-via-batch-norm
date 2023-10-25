
import torch
import torchvision.transforms as transforms
from    PIL import Image
import  os.path
import  numpy as np

import  torch.utils.data as data
import  os
import  os.path
import  errno


def find_classes(root_dir):
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if (f.endswith("png")):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    return idx


class OmniglotDownload(data.Dataset):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.all_items = find_classes(os.path.join(self.root, self.processed_folder))
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        img = str.join('/', [self.all_items[index][2], filename])

        target = self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.all_items)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, "images_background"))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.processed_folder)
            print("== Unzip from " + file_path + " to " + file_processed)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")


class Omniglot:
    '''
    meta-tr (random 1200):
        query: normal + abnormal (constrained abnormal)
    meta-ts (random 423):
        query: normal + abnormal (constrained abnormal)

    train/test split follows 
      Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic
      meta-learning for fast adaptation of deep networks." International
      conference on machine learning. PMLR, 2017.
    '''

    def __init__(self, root, batchsz, k_query, args=None, env_args=None):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param k_qry:
        :param imgsz:
        """

        self.qry_anomaly_ratio = env_args.contamination_ratio
        self.resize = 28
        if not os.path.isfile(os.path.join(root, 'omniglot.pt')):
            # if root/data.pt does not exist, just download it
            self.x = OmniglotDownload(root, download=True,
                              transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                            lambda x: x.resize((self.resize, self.resize)),
                                                            lambda x: np.reshape(x, (self.resize, self.resize, 1)),
                                                            lambda x: np.transpose(x, [2, 0, 1]),
                                                            lambda x: x/255.])
                              )

            temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
            for (img, label) in self.x:
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label] = [img]

            self.x = []
            for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
                self.x.append(np.array(imgs))

            # as different class may have different number of imgs
            self.x = np.array(self.x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
            self.x = torch.tensor(self.x)
            # each character contains 20 imgs
            print('data shape:', self.x.shape)  # [1623, 20, 84, 84, 1]
            temp = []  # Free memory
            # save all dataset into pt file.
            torch.save(self.x, os.path.join(root, 'omniglot.pt'))
            print('write into omniglot.pt.')
        else:
            # if data.pt exists, just load it.
            self.x = torch.load(os.path.join(root, 'omniglot.pt'))
            print('load from omniglot.pt.')

        # [1623, 20, 84, 84, 1]
        self.x_train, self.x_test = self.x[:1200], self.x[1200:]

        # self.normalization()

        self.batchsz = batchsz
        self.n_cls = self.x.shape[0]  # 1623
        self.n_way = 2  # n way # always one-vs-rest
        self.k_query = k_query  # k query
        assert self.k_query <=20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"], meta_split='train'),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"], meta_split='test')}




    def _load_data_cache_train(self, data_pack):
        querysz = self.k_query * self.n_way

        data_cache = []

        for sample in range(10):  # num of episodes

            x_qrys, y_qrys = [], []
            for i in range(self.batchsz):  # one batch means one set
                x_qry, y_qry = [], []

                # normal data
                selected_normal_cls = np.random.choice(data_pack.shape[0], 1, replace=False)[0]
                selected_img = np.random.choice(20, self.k_query, False)
                x_qry.append(data_pack[selected_normal_cls][selected_img])
                y_qry.append([0 for _ in range(self.k_query)])

                # outlier 
                abnormal_cls = np.arange(data_pack.shape[0])
                mask = np.ones(data_pack.shape[0], dtype=bool)
                mask[selected_normal_cls] = False
                abnormal_cls = abnormal_cls[mask]
                selected_img_cls = np.random.choice(abnormal_cls, self.k_query, replace=True)
                selected_img_id = np.random.choice(20, self.k_query, replace=True)
                x_qry.append(data_pack[(selected_img_cls, selected_img_id)])
                y_qry.append([1 for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(querysz)
                x_qry = torch.cat(x_qry, 0).reshape(querysz, 1, self.resize, self.resize)[perm]
                y_qry = torch.tensor(y_qry).reshape(querysz)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            x_qrys = torch.cat(x_qrys, 0).float().reshape(self.batchsz, querysz, 1, self.resize, self.resize)
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
        for i in range(batch_size):
            x_qry, y_qry = [], []

            # normal
            selected_normal_cls = i
            x_qry.append(data_pack[selected_normal_cls])
            y_qry.append(torch.zeros(num_qry_normal))

            # outlier 
            abnormal_cls = np.arange(data_pack.shape[0])
            mask = np.ones(data_pack.shape[0], dtype=bool)
            mask[selected_normal_cls] = False
            abnormal_cls = abnormal_cls[mask]
            selected_img_cls = np.random.choice(abnormal_cls, num_qry_abnormal, replace=True)
            selected_img_id = np.random.choice(class_size, num_qry_abnormal, replace=True)
            x_qry.append(data_pack[(selected_img_cls, selected_img_id)])
            y_qry.append(torch.ones(num_qry_abnormal))

            # shuffle inside a batch
            perm = np.random.permutation(querysz)
            x_qry = torch.cat(x_qry, 0).reshape(querysz, 1, self.resize, self.resize)[perm]
            y_qry = torch.cat(y_qry, 0).reshape(querysz)[perm]

            # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)

        x_qrys = torch.cat(x_qrys, 0).float().reshape(batch_size, querysz, 1, self.resize, self.resize)
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
        elif mode == 'test':
            return self.datasets_cache[mode]
        else:
            raise NotImplementedError()



if __name__ == '__main__':

    import  time
    import  torch
    import  visdom

    # plt.ion()
    viz = visdom.Visdom(env='omniglot_view')

    db = Omniglot('db/omniglot', batchsz=20, k_query=15)

    for i in range(1000):
        x_qry, y_qry = db.next('test')


        # [b, setsz, h, w, c] => [b, setsz, c, w, h] => [b, setsz, 3c, w, h]
        batchsz, setsz, c, h, w = x_qry.size()


        viz.images(x_qry[0], nrow=15, win='x_qry', opts=dict(title='x_qry'))
        viz.text(str(y_qry[0]), win='y_qry', opts=dict(title='y_qry'))


        time.sleep(10)

