"""
DataReader aims to
- read data by using WFDB pkg
- customize which annotation you are going to use
- provide wavelet transform as pre-processing feature extraction
- save data as .csv file

"""
import numpy as np
import wfdb
import os
import sys
import time
import csv

from Private import Variables, DataPreprocessing, Wavelet


class DataReader:
    def __init__(self, path='/Users/AppleUser/MyProjects/MyThesis/Private/mit-bih-arrhythmia-database-1.0.0/', ann=Variables.ann5, ):
        """
        :param path: the folder you place the mitdb data and save the data
        :param ann: annotation types, refert to Variables.py
        """
        self._dir = path
        self._ann = ann
        pass

    def read_data(self, record):
        """
        This method reads files with WFDB pkg to deal with .hea, .art, .dat files...
        :return: data_set, label_set
        """

        # Initialization
        _tmp_data_set = []
        _tmp_label_set = []


        print('reading case #{}'.format(record))
        record = self._dir + record
        rec = wfdb.rdrecord(record_name=record, channels=[0], physical=False)
        ann = wfdb.rdann(record_name=record, extension='atr', return_label_elements=['symbol'])

        # Make a screener to fire 5 types of annotation
        ann_ids = np.in1d(ann.symbol, self._ann)
        beats = np.array(ann.sample)[ann_ids]
        label = np.array(ann.symbol)[ann_ids]
        label = DataPreprocessing.str_to_int(label)
        sig = rec.d_signal.ravel()


        # Abandon head and tail [1: -1]
        for j, beat in enumerate(beats[1:-1]):

            # Beat width = 256 sample points
            _from, _to = beat - 128, beat + 128

            if _from < 0:
                # skip uncompleted beat
                pass
            else:
                buffer = self.process(sig, _from, _to)

            # Append sample to _tmp_data_set and _tmp_label_set
            _tmp_data_set = np.concatenate((_tmp_data_set, [buffer])) if _tmp_data_set != list([]) else [buffer]
            _tmp_label_set = np.concatenate((_tmp_label_set, [label[j]]), axis=-1) if _tmp_label_set != list([]) else [label[j]]

        return _tmp_data_set, _tmp_label_set

    def save_data(self, _data, _name, _ext='.csv', ):
        """
        write n-dimension data for data_set (n_sample, n_elements)
        :param _input: input ndarray type
        :param filename: input string


        ----------------------
        example:

        """
        _name = self._dir + _name + _ext
        csvfile2 = open(_name , 'w', newline='')
        writer = csv.writer(csvfile2)

        try:
            _data = np.reshape(_data, (len(_data), len(_data[0])))
        except Exception as e:
            print(e)
            _data = np.reshape(_data, (len(_data), 1))
            print('change object type')
        finally:
            for val in _data:
                writer.writerow(val)
            csvfile2.close()

    @classmethod
    def process(self, sig, _from, _to, ):
        pass


class Raw(DataReader):
    def __init__(self):
        super().__init__()

    def process(self, sig, _from, _to):
        return sig[_from: _to]


class WaveletTransform(DataReader):
    def __init__(self, _level=4, ):
        super().__init__()
        self._wav = Wavelet.Wavelet()
        self._level = _level;

    def process(self, sig, _from, _to):
        _, cd, _, _ = self._wav.wavelet_decompose(_input=sig[_from: _to], _level=self._level)
        return cd


if __name__ == '__main__':
    r = Raw()
    a, b = np.array([]), np.array([])
    i = 0

    for record in Variables.experiment_cases[:11]:
        tmp_a, tmp_b = r.read_data(record=record)
        if i == 0:
            a = tmp_a
            b = tmp_b
            i += 1
        else:
            a = np.concatenate((a, tmp_a), axis=0)
            b = np.concatenate((b, tmp_b), axis=0)
        print(np.shape(a))

    r.save_data(_data=a, _name='data_set', _ext='.csv')
    r.save_data(_data=b, _name='label_set', _ext='.csv')
    pass
