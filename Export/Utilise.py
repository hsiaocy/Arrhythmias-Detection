import numpy as np
from Export import DataReader
path = r'./Private/'


def mask_top_k(weights, bias, sparsity=0.5, filename=None):
    """

    :param weights:
    :param bias:
    :param sparsity:
    :param filename:
    :return:
    """
    k = int(len(bias) * (1 - sparsity))
    thres_w = np.reshape(np.sort(np.abs(weights))[:, -k], [-1, 1])
    thres_b = np.sort(np.abs(bias))[-k]

    mask_w_p = weights > thres_w;
    mask_w_n = weights < -thres_w;
    mask_w = mask_w_p + mask_w_n

    mask_b_p = bias > thres_b;
    mask_b_n = bias < -thres_b;
    mask_b = mask_b_p + mask_b_n

    if filename != None:
        dr = DataReader.DataReader()
        dr._dir = path
        dr.save_data(_data=weights * mask_w, _name=filename, _ext='.csv')
    return mask_w, mask_b


def prune_top_k(weights, bias, sparsity=0.5):
    """

    :param weights:
    :param bias:
    :param sparsity:
    :return:
    """
    k = int(len(bias) * (1 - sparsity))
    thres_w = np.reshape(np.sort(np.abs(weights)[:, -k]), [-1, 1])
    thres_b = np.sort(np.abs(bias))[-k]
    weights = weights[weights > thres_w]
    bias = bias[bias > thres_b]
    return weights, bias


def create_mask(para_dict, sparsity, filename='weights_mask'):
    """

    :param para_dict:
    :param sparsity:
    :param filename:
    :return:
    """
    mask_dict = dict()

    mask_w, mask_b = mask_top_k(weights=para_dict['w_enc'], bias=para_dict['b_enc'], sparsity=sparsity, filename=filename+'_encoder')
    mask_dict['mask_w_enc'] = mask_w
    mask_dict['mask_b_enc'] = mask_b

    mask_w, mask_b = mask_top_k(weights=para_dict['w_dec'], bias=para_dict['b_dec'], sparsity=sparsity, filename=filename+'_decoder')
    mask_dict['mask_w_dec'] = mask_w
    mask_dict['mask_b_dec'] = mask_b

    return mask_dict


def restore(para_dict, mask_dict):
    """

    :param para_dict:
    :param mask_dict:
    :return:
    """
    N, M = np.shape(para_dict['w_enc'])
    w_r_enc = np.ones(shape=[N, M], dtype=np.float32)
    for i in range(N):
        t = 0
        for j in range(M):
            if mask_dict['mask_w_enc'][i, j] != True:
                w_r_enc[i, j] *= 0
            elif mask_dict['mask_w_enc'][i, j] == True:
                w_r_enc[i, j] *= para_dict['w_enc'][i, t]
                t += 1

    N, M = np.shape(para_dict['w_dec'])
    w_r_dec = np.ones(shape=[N, M], dtype=np.float32)
    for i in range(N):
        t = 0
        for j in range(M):
            if mask_dict['mask_w_dec'][i, j] != True:
                w_r_dec[i, j] *= 0
            elif mask_dict['mask_w_dec'][i, j] == True:
                w_r_dec[i, j] *= para_dict['w_dec'][i, t]
                t += 1

    M = len(para_dict['b_enc'])
    b_r_enc = np.ones(shape=[M], dtype=np.float32)
    t = 0
    for j in range(M):
        if mask_dict['mask_b_enc'][j] != True:
            b_r_enc[j] *= 0
        elif mask_dict['mask_b_enc'][j] == True:
            b_r_enc[j] *= para_dict['b_enc'][t]
            t += 1

    M = len(para_dict['b_dec'])
    b_r_dec = np.ones(shape=[M], dtype=np.float32)
    t = 0
    for j in range(M):
        if mask_dict['mask_b_dec'][j] != True:
            b_r_dec[j] *= 0
        elif mask_dict['mask_b_dec'][j] == True:
            b_r_dec[j] *= para_dict['b_dec'][t]
            t += 1

    return {'w_enc': w_r_enc, 'w_dec': w_r_dec, 'b_enc': b_r_enc, 'b_dec': b_r_dec}
    pass