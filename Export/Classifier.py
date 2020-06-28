import numpy as np




class Model:
    def __init__(self, **mydata):
        """

        :param mydata: A Dictonary consists of {train_X, train_Y, test_X, test_Y}
            train_X: training-FeatrureSet
            train_Y: training-LabelSet
            test_X: testing-FeatureSet
            test_Y: testing-LabelSet
        """
        self.MyData = dict()
        for key, item in mydata.items():
            self.MyData[str(key)] = item
        pass

    def train(self, ):
        """
        train model with training-DataSet

        """
        self._clsfir.fit(self.MyData['train_X'], self.MyData['train_Y'])
        pass

    def predict(self):
        """
        predict the testing-LabelSet

        :return: predicted testing-LabelSet
        """
        return self._clsfir.predict(self.MyData['test_X'])
        pass


class SVM(Model):
    """
    SVM classifier
    """
    def __init__(self, iter, **mydata):
        super().__init__(**mydata)

        # one-vs-one classifier, SVC
        from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
        from sklearn.svm import LinearSVC, SVC
        self._clsfir = OneVsOneClassifier(
            estimator=SVC(
                kernel='rbf',
                decision_function_shape='ovo',
                max_iter=iter,
                class_weight='balanced',
                verbose=True,
            )
        )
        pass


class RandomForest(Model):
    """
    Random Forest classifier
    """
    def __init__(self, depth, estmtr, **mydata):
        super().__init__(**mydata)

        # random forest with gini
        from sklearn.ensemble import RandomForestClassifier
        self.clsfir = RandomForestClassifier(
            max_depth=depth,
            n_estimators=estmtr,
            criterion='gini',
            class_weight='balanced',
            verbose=True,
        )


class MLP(Model):
    """
    Multi-layer Perceptron
    """
    def __init__(self, iter, **mydata):
        super().__init__(**mydata)

        # random forest with gini
        from sklearn.neural_network import MLPClassifier
        self.clsfir = MLPClassifier(max_iter=iter)
