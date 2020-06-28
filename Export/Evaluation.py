from sklearn.metrics import confusion_matrix
import numpy as np

class Evaluation:
    def __init__(self, pred, true, classes=5):
        # Initialization
        self._class = classes
        self._pred = np.array(pred)
        self._true = np.array(true)
        self.EvaMetrics = {'TP': np.zeros(shape=self._class),
                           'FP': np.zeros(shape=self._class),
                           'TN': np.zeros(shape=self._class),
                           'FN': np.zeros(shape=self._class),
                           'class_size': np.zeros(shape=self._class),
                           'Precision': 0,
                           'Recall': 0,
                           'Specificity': 0,
                           'Accuracy': 0,
                           'FPR': 0,
                           'F1-Score': 0,}
        pass

    def evaluation(self, ):

        # Calculate tp, tn, fp, fn for each class.
        for i in range(self._class):
            pred_buffer, truth_buffer = self._pred.copy(), self._true.copy()

            pred_p_idx, truth_p_idx = np.where(pred_buffer == i), np.where(truth_buffer == i)
            pred_n_idx, truth_n_idx = np.where(pred_buffer != i), np.where(truth_buffer != i)

            pred_buffer[pred_p_idx], truth_buffer[truth_p_idx] = 1, 1
            pred_buffer[pred_n_idx], truth_buffer[truth_n_idx] = 0, 0

            self.EvaMetrics['TP'][i] = np.shape(np.where((pred_buffer + truth_buffer) == 2))[1]
            self.EvaMetrics['TN'][i] = np.shape(np.where((pred_buffer + truth_buffer) == 0))[1]
            self.EvaMetrics['FP'][i] = len(pred_buffer[pred_buffer > truth_buffer])
            self.EvaMetrics['FN'][i] = len(truth_buffer[pred_buffer < truth_buffer])

        self.EvaMetrics['Precision']   = np.true_divide(self.EvaMetrics['TP'], (self.EvaMetrics['TP'] + self.EvaMetrics['FP']))
        self.EvaMetrics['Recall']      = np.true_divide(self.EvaMetrics['TP'], (self.EvaMetrics['TP'] + self.EvaMetrics['FN']))
        self.EvaMetrics['Specificity'] = np.true_divide(self.EvaMetrics['TN'], (self.EvaMetrics['TN'] + self.EvaMetrics['FP']))
        self.EvaMetrics['Accuracy']    = np.true_divide((self.EvaMetrics['TP'] + self.EvaMetrics['TN']), (self.EvaMetrics['TP'] + self.EvaMetrics['TN'] + self.EvaMetrics['FP'] + self.EvaMetrics['FN']))
        self.EvaMetrics['FPR']         = np.true_divide(self.EvaMetrics['FP'], (self.EvaMetrics['TN'] + self.EvaMetrics['FP']))
        self.EvaMetrics['F1-Score']    = np.true_divide((self.EvaMetrics['Precision'] * self.EvaMetrics['Recall']), (self.EvaMetrics['Precision'] + self.EvaMetrics['Recall'])) * 2

        return self.EvaMetrics
        pass
