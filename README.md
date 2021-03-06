[![Platform](https://img.shields.io/badge/Platform-Tensorflow-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.6-green.svg)]()

# Arrthymias Detection Based on Electrocardiogram Using A Deep Autoencoder

AD (Arrhythmias Detection) is a repo that investigates variant autoencoder(AE) in feature extraction, and applies them to classify from Electrocardiogram(ECG) to arrhythmias.

- Situation: We found five types of ECG from MIT-BIH arrhythmias database includes "Normal", "Paced Beat", "Premature Ventricular Contraction", "Right Bundle Branch Block" and "Left Bundle Branch Block".
- Task: It's necessary to find a better way on finding the pattern from ECG and classify an ECG signal to normal or other arrhythmias.
- Actions: Now AD includes 4 AE as feature extractor, and classify by using Random Forest (can be applied with SVM or softmax classifier, too).


# Arrhythmias Data

Followed by the MIT-BIH Arrhythmias Database

- [MITDB](https://physionet.org/physiobank/database/mitdb/)
This database is described in
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
Also, more functions are updating.

Citation:
Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/content/101/23/e215.full]; 2000 (June 13).

# Requirements

Before you start testing, following requirements are needed.

- Python3.6.5
- TensorFlow1.2.0
- numpy
- scipy
- sklearn
- matplotlib
- wfdb2.2.0
- pywt

# About Data

If you need data, you could use Export/DataReader.py to read/write data to you space after you download the raw-data from the MITDB.
- [Download the ZIP file](https://storage.googleapis.com/mitdb-1.0.0.physionet.org/mit-bih-arrhythmia-database-1.0.0.zip) 

# TODO
- More detail about AD.
- More arrhythmias to be detected.
