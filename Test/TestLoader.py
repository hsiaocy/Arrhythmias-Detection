import numpy as np
import unittest
import Private
import wfdb

class MyTestCase(unittest.TestCase):

    def test_wfdb_load_data(self, use='raw'):

        # initialization
        from Private import Variables, DataPreprocessing, Wavelet
        _data_set = []
        _label_set = []
        _abs_dir = '/Users/AppleUser/MyProjects/MyThesis/Private/mit-bih-arrhythmia-database-1.0.0/'

        # load data using wfdb pkg
        record_name = _abs_dir + Variables.experiment_cases[0]
        record = wfdb.rdrecord(record_name=record_name, channels=[0], physical=False)
        ann = wfdb.rdann(record_name=record_name, extension='atr', return_label_elements=['symbol'])

        # Make a screener to fire 5 types of annotation
        ann5_ids = np.in1d(ann.symbol, Variables.ann5)
        beats = np.array(ann.sample)[ann5_ids]
        label = np.array(ann.symbol)[ann5_ids]
        label = DataPreprocessing.str_to_int(label)
        sig = record.d_signal.ravel()

        if use == 'wavelet':
            import wavelet
            wav = Wavelet.Wavelet()
            wavelet_level = 4;

        # abandon head and tail [1: -1]
        for j, beat in enumerate(beats[1:-1]):

            # Beat width = 256 sample points
            _from, _to = beat - 128, beat + 128

            if _from < 0:
                # skip uncompleted beat
                pass
            else:
                if use == 'wavelet':
                    ca, cd, a, d = wav.wavelet_decompose(_input=sig[_from: _to], level=wavelet_level)
                    buffer = cd
                elif use == 'raw':
                    buffer = sig[_from: _to]

            # append sample to _data_set and _label_set
            _data_set = np.concatenate((_data_set, [buffer])) if _data_set != list([]) else [buffer]
            _label_set = np.concatenate((_label_set, [label[j]]), axis=-1) if _label_set != list([]) else [label[j]]

        print(_data_set.shape, _label_set.shape)
    # return data_set, label_set


if __name__ == '__main__':
    unittest.main()
