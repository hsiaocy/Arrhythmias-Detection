import tensorflow as tf
import numpy as np

class FeatureExtractor:
    def __init__(self, ):
        self._train_feat = []
        self._test_feat = []

    def train_feat(self, ):
        pass


class PCA(FeatureExtractor):
    """
    Baseline method
    """
    def __init__(self, n_components=12, cd_len=None, theshold=0.8):
        super().__init__()
        from sklearn.decomposition import PCA
        self._pca = PCA(n_components=n_components)


        self._cd_len = [134, 209, 255, 286, 310, 330, 348, 365]
        self._theshold = theshold


    def train_feat(self, train_X, test_X):
        """
        This method train the feature extractor

        :param train_X: training-DataSet
        :param test_X: testing-DataSet
        :return: training-FeatureSet, testing-FeatureSet
        """
        for j in self._cd_len:

            # extract training set feature
            self._train_feat = self._pca.fit_transform(X=train_X[:, :j])

            # "sum(pca.explained_variance_ratio_) > theshold" means all the components are extracted
            if sum(self._pca.explained_variance_ratio_) >= self._theshold:

                # extract testing set feature
                self._test_feat = self._pca.fit_transform(X=test_X[:, :j])
                break


