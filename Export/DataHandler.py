"""
DataHandler aims to
- load data from our end
- normalize data
- shuffle your DataSet along with LabelSet
- split data to traning set and testing set

"""

import numpy as np
import os
import sys
import csv

from Private import Variables, DataPreprocessing, Wavelet


class DataHandler:
    def __init__(self, path='/Users/AppleUser/MyProjects/MyThesis/Private/mit-bih-arrhythmia-database-1.0.0/', ann=Variables.ann5, ):
        """
        :param path: the folder you place the mitdb data and save the data
        """
        self._dir = path
        pass

    def load_data(self, ):
        """

        :return: DataSet and LabelSet
        """
        return np.loadtxt(fname=self._dir + 'data_set.csv', delimiter=',', dtype='float32')\
            , np.loadtxt(fname=self._dir + 'label_set.csv', delimiter=',', dtype='float32')
        pass

    def normalize_data(self, _data):
        """
        This method will normalize the data from 0 to 1

        :param _data: raw-DataSet
        :return: Normalized data
        """
        from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
        norm = Normalizer(norm='l2')
        norm = norm.fit(X=_data)
        _data = norm.transform(X=_data)
        return _data

    def shuffle_data(self, _data_set, _label_set):
        """
        This method will shuffle the DataSet and LabelSet at the same time

        :param data_set: DataSet
        :param label_set: LabelSet
        :return: shffled Dataset and LabelSet
        """
        random_idx = np.random.choice(a=len(_label_set), size=len(_label_set), replace=False)
        return _data_set[random_idx], _label_set[random_idx]

    def kf_shuffle(self, data_set, label_set, fold=10):
        """
        This method will shuffle each k sub-set
        :param data_set:
        :param label_set:
        :param fold: K-fold shffled Dataset and LabelSet
        :return:
        """
        set_size = data_set.shape[1]
        for i in range(fold):
            random_idx = np.random.choice(a=set_size, size=set_size, replace=False)
            data_set[i] = data_set[i][random_idx]
            label_set[i] = label_set[i][random_idx]
        return data_set, label_set

    def split_data(self, _data_set, _label_set, _train_ratio, ):
        """

        :param label_set:
        :param train_ratio:
        :return: training-DataSet, training-LabelSet, testing-DataSet, testing-LabelSet
        """
        N_idx, V_idx, P_idx, L_idx, R_idx = np.where(_label_set == 0), np.where(_label_set == 1), np.where(_label_set == 2), np.where(_label_set == 3), np.where(_label_set == 4)
        N_ds,  V_ds,  P_ds,  L_ds,  R_ds  = _data_set[N_idx], _data_set[V_idx], _data_set[P_idx], _data_set[L_idx], _data_set[R_idx]
        N_ls,  V_ls,  P_ls,  L_ls,  R_ls  = _label_set[N_idx], _label_set[V_idx], _label_set[P_idx], _label_set[L_idx], _label_set[R_idx]

        N_train_ds, V_train_ds, P_train_ds, L_train_ds, R_train_ds = N_ds[0: int(len(N_ds) * _train_ratio)],\
                                                                     V_ds[0: int(len(V_ds) * _train_ratio)], \
                                                                     P_ds[0: int(len(P_ds) * _train_ratio)], \
                                                                     L_ds[0: int(len(L_ds) * _train_ratio)], \
                                                                     R_ds[0: int(len(R_ds) * _train_ratio)]

        N_train_ls, V_train_ls, P_train_ls, L_train_ls, R_train_ls = N_ls[0: int(len(N_ls) * _train_ratio)],\
                                                                     V_ls[0: int(len(V_ls) * _train_ratio)], \
                                                                     P_ls[0: int(len(P_ls) * _train_ratio)], \
                                                                     L_ls[0: int(len(L_ls) * _train_ratio)], \
                                                                     R_ls[0: int(len(R_ls) * _train_ratio)]

        N_test_ds, V_test_ds, P_test_ds, L_test_ds, R_test_ds = N_ds[int(len(N_ds) * _train_ratio):], \
                                                                V_ds[int(len(V_ds) * _train_ratio):], \
                                                                P_ds[int(len(P_ds) * _train_ratio):], \
                                                                L_ds[int(len(L_ds) * _train_ratio):], \
                                                                R_ds[int(len(R_ds) * _train_ratio):]

        N_test_ls, V_test_ls, P_test_ls, L_test_ls, R_test_ls = N_ls[int(len(N_ls) * _train_ratio):], \
                                                                V_ls[int(len(V_ls) * _train_ratio):], \
                                                                P_ls[int(len(P_ls) * _train_ratio):], \
                                                                L_ls[int(len(L_ls) * _train_ratio):], \
                                                                R_ls[int(len(R_ls) * _train_ratio):]

        # training dataset
        # training labelset
        # testing dataset
        # testing labelset
        return np.concatenate((N_train_ds, V_train_ds, P_train_ds, L_train_ds, R_train_ds)), \
               np.concatenate((N_train_ls, V_train_ls, P_train_ls, L_train_ls, R_train_ls)), \
               np.concatenate((N_test_ds, V_test_ds, P_test_ds, L_test_ds, R_test_ds)), \
               np.concatenate((N_test_ls, V_test_ls, P_test_ls, L_test_ls, R_test_ls))


if __name__ == '__main__':
    r = DataHandler()
    a,b = r.load_data()

    # normalize data
    a = r.normalize_data(_data=a)

    # shuffle
    a,b = r.shuffle_data(_data_set=a, _label_set=b)

    # make training set and testing set
    a,b,c,d = r.split_data(_data_set=a, _label_set=b, _train_ratio=0.8)
    pass