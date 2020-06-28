from Export import DataHandler, DataReader, FeatureExtractor, Classifier, Evaluation, Utilise, DSD



if __name__ == '__main__':

    """
    Data Load and Pre-processing Phase
    """
    dh = DataHandler.DataHandler()
    data_set, label_set = dh.load_data()

    # normalize data
    data_set = dh.normalize_data(_data=data_set)

    # shuffle
    data_set, label_set = dh.shuffle_data(_data_set=data_set, _label_set=label_set)

    # make training set and testing set
    train_data_set, train_label_set, test_data_set, test_label_set = dh.split_data(_data_set=data_set, _label_set=label_set, _train_ratio=0.8)


    """
    Feature Extraction Phase

    # Features we got for classification : pca._train_feat, pca._test_feat
    # pca = FeatureExtractor.PCA(theshold=0.99)
    # pca.train_feat(train_X=train_data_set, test_X=test_data_set) 
    """
    # Dense Phase
    Dense = DSD.DenseLayer(feat_len=256, learning_rate=0.05, batch_size=64, num_hidden=32, sparsity=0.7, epoch=2)
    para_dict = Dense.train_feat(train_x=train_data_set)

    # Sparse Phase
    mask_dict = Utilise.create_mask(para_dict=para_dict, sparsity=0.5,)
    Sparse = DSD.SparseLayer(feat_len=256, learning_rate=0.005, batch_size=64, epoch=2, para_dict=para_dict, mask_dict=mask_dict)
    para_dict = Sparse.train_feat(train_x=train_data_set)

    # ReDense Phase
    para_dict = Utilise.restore(para_dict=para_dict, mask_dict=mask_dict)
    ReDense = DSD.ReDense(feat_len=256, learning_rate=0.05, batch_size=64, num_hidden=32, sparsity=0.7, epoch=2, para_dict=para_dict)
    train_feat, test_feat, para_dict = ReDense.train_feat(train_x=train_data_set, test_x=test_data_set)

    """
    Classification
    """
    svm = Classifier.SVM(train_X=train_data_set,
                         train_Y=train_label_set,
                         test_X=test_data_set,
                         test_Y=test_label_set,
                         iter=1000)
    svm.train()
    result = svm.predict()

    eva = Evaluation.Evaluation(pred=result, true=test_label_set)
    metrics = eva.evaluation()
    print(metrics)
    pass
