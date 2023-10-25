import os.path
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class Anoshift:
    '''
    meta-tr (2006-2010):
        normal
    meta-val (2006-2010)
        normal + abnormal
    meta-ts (2006-2015):
        normal + abnormal 
    '''

    def __init__(self, root, batchsz, k_query, args=None, env_args=None):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param k_qry:
        """

        self.qry_anomaly_ratio = env_args.contamination_ratio
        train_years= [2006, 2007, 2008, 2009, 2010]
        test_years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]
        if not os.path.isfile(os.path.join(root, 'x_train.npz')):
            dfs = []
            nums= []
            for year in train_years:
                df_year = self.load_train_year(year)
                count_norm = df_year[df_year["18"] == "1"].shape[0]
                count_anomal = df_year[df_year["18"] != "1"].shape[0]
                print(year, "normal:", count_norm, "anomalies:", count_anomal)
                nums.append(len(df_year))
                dfs.append(df_year)

            print("Preprocess train data...")
            df_all_years = pd.concat(dfs, ignore_index=True)
            df_all_years = self.rename_columns(df_all_years)
            df_new, ohe_enc = self.preprocess(df_all_years)
            numerical_cols = df_new.columns.to_numpy()[['num_' in i for i in df_new.columns]]
            self.x_train = []
            self.y_train = []
            self.x_val = []
            self.y_val = []
            start_idx = 0
            for i in range(len(train_years)):
                df_new_year = df_new[start_idx:start_idx+nums[i]]
                x_train, x_val = train_test_split(df_new_year, test_size=0.001, random_state=0, shuffle=True)
                x_train_clear = x_train[x_train["label"] == 0]
                self.x_train.append(x_train_clear[numerical_cols].to_numpy())
                
                self.x_val.append(x_val[numerical_cols].to_numpy())
                self.y_val.append(x_val["label"].to_numpy())
                start_idx = start_idx+nums[i]        # split train + val
            
                
            self.x_test = []
            self.y_test = []

            for year in test_years:
                df_year = self.load_test_year(year)
                df_year = self.rename_columns(df_year)
                df_test, _ = self.preprocess(df_year, ohe_enc)
                X_test = df_test[numerical_cols].to_numpy()
                y_test = df_test["label"].to_numpy()

                X_test = np.nan_to_num(X_test)
                y_test = np.nan_to_num(y_test, 0)
                X_test_normal = X_test[y_test==0]
                X_test_abnormal = X_test[y_test==1]
                X_test_sub = np.concatenate([X_test_normal,X_test_abnormal[:len(X_test_normal)]],0)
                y_test_sub = np.ones(X_test_sub.shape[0])
                y_test_sub[:len(X_test_normal)]=0
                self.x_test.append(X_test_sub)
                self.y_test.append(y_test_sub)
            np.savez(os.path.join(root,"x_train.npz"),*self.x_train)
            np.savez(os.path.join(root,"x_val.npz"),*self.x_val)
            np.savez(os.path.join(root,"y_val.npz"),*self.y_val)
            np.savez(os.path.join(root,"x_test.npz"),*self.x_test)
            np.savez(os.path.join(root,"y_test.npz"),*self.y_test)
            np.save(os.path.join(root,"ohe_cats.npy"),[len(ohe_enc.categories_[i]) for i in range(len(ohe_enc.categories_))])
        else:
            x_train = np.load(os.path.join(root,"x_train.npz"))
            # np.random.seed(seed=42)
            num_samples = []
            for k in x_train:
                num_samples.append(len(x_train[k]))
            print(f"num samples {num_samples}")
                # x_train_idx.append(np.random.choice(num_samples,int(num_samples*0.05),replace=False))
            
            # sub_num = int(np.sum(num_samples)*0.05/5)
            sub_num = 5000
            x_train_idx = [np.random.choice(num_samples[k],sub_num,replace=False) for k in range(len(num_samples))]
            self.x_train = [x_train[k][x_train_idx[i]] for i,k in enumerate(x_train)]
            self.y_train = []
            x_val = np.load(os.path.join(root,"x_val.npz"))
            self.x_val = [x_val[k] for k in x_val]
            y_val = np.load(os.path.join(root,"y_val.npz"))
            self.y_val = [y_val[k] for k in y_val]
            x_test = np.load(os.path.join(root,"x_test.npz"))
            self.x_test = [x_test[k] for k in x_test]
            y_test = np.load(os.path.join(root,"y_test.npz"))
            self.y_test = [y_test[k] for k in y_test]
            self.num_cats = np.load(os.path.join(root,"ohe_cats.npy"))
            
            self.num_cats = np.concatenate([np.ones(9),self.num_cats]).astype(int)

        self.batchsz = batchsz
        self.n_cls = len(self.x_test)
        self.n_way = 2  # n way # always one-vs-rest
        self.k_query = k_query  # k query

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0, 'val': 0}
        self.datasets = { "train": [self.x_train,self.y_train,train_years],
                        "val": [self.x_val,self.y_val,train_years],
                        "test": [self.x_test,self.y_test,test_years], }  # original data cached
        
        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"], meta_split='train'),
                               "test": self.load_data_cache(self.datasets["test"], meta_split='test'),
                               "val": self.load_data_cache(self.datasets["val"], meta_split='test')}

    def perm_attributes(self,cat_num,data,perm_idx):
        new_data = []
        init_idx = 0
        for lengh in cat_num:
            new_data.append(data[:,init_idx:init_idx+lengh])
            init_idx= init_idx+lengh
        new_data = np.concatenate([new_data[i] for i in perm_idx],1)
        return new_data

    def _load_data_cache_train(self, data_pack):
        data,_,years = data_pack
        # querysz = self.k_query * self.n_way
        num_feat = data[0].shape[1]
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for _ in range(10):  # num of episodes

            x_qrys, y_qrys, normal_classes = [], [], []

            for k in range(self.batchsz):  # one batch means one set

                x_qry, y_qry = [], []
                perm_attr = np.random.permutation(data[0].shape[1])

                # normal data
                
                selected_normal_cls = np.random.choice(len(data),1, replace=False)
                normal_data = np.concatenate([data[i] for i in selected_normal_cls])
                
                selected_normal_id = np.random.choice(len(normal_data), int(self.k_query*0.8), False)
                selected_normal = normal_data[selected_normal_id]
                selected_normal = selected_normal[:,perm_attr]


                x_qry.append(torch.tensor(selected_normal))
                y_qry.append(torch.zeros(len(selected_normal)))

                # outlier 
                abnormal_cls = np.arange(len(data))
                mask = np.ones(len(data), dtype=bool)
                mask[list(selected_normal_cls)] = False
                abnormal_cls = abnormal_cls[mask]
                abnormal_data = np.concatenate([data[i] for i in abnormal_cls],0)
                selected_abnormal_id = np.random.choice(len(abnormal_data), self.k_query-int(self.k_query*0.8), replace=False)
                selected_abnormal = abnormal_data[selected_abnormal_id]
                selected_abnormal = selected_abnormal[:,perm_attr]

                x_qry.append(torch.tensor(selected_abnormal))
                y_qry.append(torch.ones(len(selected_abnormal)))

                # shuffle inside a batch
                perm = np.random.permutation(len(selected_normal)+len(selected_abnormal))
                x_qry = torch.cat(x_qry, 0)[perm]
                y_qry = torch.cat(y_qry)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_qrys.append(x_qry.to(torch.float32))
                y_qrys.append(y_qry)
                normal_classes.append([years[i] for i in selected_normal_cls])

            # data_cache.append([x_qrys, y_qrys,normal_classes])
            data_cache.append([x_qrys, y_qrys])

        return data_cache

    def _load_data_cache_test(self, data_pack):
        data,labels,years = data_pack
        tot_cls = len(data)
        qry_anomaly_ratio = self.qry_anomaly_ratio

        x_qrys, y_qrys = [], []

        for i in range(tot_cls):  # one batch means one set
            num_qry_normal = np.sum(labels[i]==0)
            num_qry_abnormal = int(num_qry_normal*qry_anomaly_ratio / (1-qry_anomaly_ratio))
            if num_qry_abnormal == 0:
                num_qry_abnormal = 1 
            # querysz = num_qry_normal + num_qry_abnormal

            x_qry, y_qry = [], []

            # normal data
            x_qry.append(torch.tensor(data[i][labels[i]==0]))
            y_qry.append(torch.zeros(num_qry_normal))

            # outlier 
            x_abnormal = data[i][labels[i]==1]
            try:
                selected_abnormal_id = np.random.choice(len(x_abnormal),num_qry_abnormal,replace=False)
            except:
                selected_abnormal_id = np.arange(len(x_abnormal))
            x_qry.append(torch.tensor(x_abnormal[selected_abnormal_id]))
            y_qry.append(torch.ones(num_qry_abnormal))

            # shuffle inside a batch
            perm = np.random.permutation(num_qry_normal+len(selected_abnormal_id))
            x_qry = torch.cat(x_qry, 0)[perm]
            y_qry = torch.cat(y_qry)[perm]

            x_qrys.append(x_qry.to(torch.float32))
            y_qrys.append(y_qry)


        return [x_qrys, y_qrys,years]


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

    def load_raw_traindata(self):
        train_data = torch.tensor(np.concatenate(self.x_train,0))
        train_labels = torch.zeros(len(train_data))
        return train_data,train_labels

    def load_train_year(self,year):
        if year <= 2010:
            df = pd.read_parquet(f'/shared_scratch/chen/anoshift/Kyoto-2016_AnoShift/subset/{year}_subset.parquet',  engine='fastparquet')
        else:
            import sys
            sys.exit(-1)
        df = df.reset_index(drop=True)
        return df

    def load_test_year(self,year):
        if year <= 2010:
            df = pd.read_parquet(f'/shared_scratch/chen/anoshift/Kyoto-2016_AnoShift/subset/{year}_subset_valid.parquet',  engine='fastparquet')
        else:
            df = pd.read_parquet(f'/shared_scratch/chen/anoshift/Kyoto-2016_AnoShift/subset/{year}_subset.parquet',  engine='fastparquet')

        df = df.reset_index(drop=True)
        return df

    def rename_columns(self,df):    
        categorical_cols = ["0", "1", "2", "3", "13"]
        numerical_cols = ["4", "5", "6", "7", "8", "9", "10", "11", "12"]
        additional_cols = ["14", "15", "16", "17", "19"]
        label_col = ["18"]

        new_names = []
        for col_name in df.columns.astype(str).values:
            if col_name in numerical_cols:
                df[col_name] = pd.to_numeric(df[col_name])
                new_names.append((col_name, "num_" + col_name))
            elif col_name in categorical_cols:
                new_names.append((col_name, "cat_" + col_name))
            elif col_name in additional_cols:
                new_names.append((col_name, "bonus_" + col_name))
            elif col_name in label_col:
                df[col_name] = pd.to_numeric(df[col_name])
                new_names.append((col_name, "label"))
            else:
                new_names.append((col_name, col_name))
        df.rename(columns=dict(new_names), inplace=True)
        
        return df

    def preprocess(self,df, enc=None):
        if not enc:
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(df.loc[:,['cat_' in i for i in df.columns]])
        
        num_cat_features = enc.transform(df.loc[:,['cat_' in i for i in df.columns]]).toarray()

        df_catnum = pd.DataFrame(num_cat_features)
        df_catnum = df_catnum.add_prefix('catnum_')

        df.reset_index(drop=True)
        df_new = pd.concat([df,  df_catnum], axis=1)
        
        
        filter_clear = df_new["label"] == 1
        filter_infected = df_new["label"] < 0
        df_new["label"][filter_clear] = 0
        df_new["label"][filter_infected] = 1

        return df_new, enc


