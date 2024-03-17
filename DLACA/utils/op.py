import tqdm
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import platform
import torch.nn.functional as F
import torch.backends


def get_entropy(feature):
    feature = F.softmax(feature, dim=-1)
    log_feature = torch.log(feature)
    entropy = torch.sum(feature * log_feature, dim=-1)
    return entropy


def get_pseudo_label(p, q, label, size, cod):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    vp, pred1 = torch.max(p, dim=-1)
    vq, pred2 = torch.max(q, dim=-1)
    p_mask = (vp >= cod)
    q_mask = (vq >= cod)
    mask1 = (pred1 == pred2)
    mask = (mask1 & p_mask) & (q_mask)


    pseudo_label = pred1 * mask
    pseudo_label[0:size] = label
    mask[0:size] = True
    return pseudo_label, mask


def get_labels(p, q, label, size, FF, cod):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    vp, pred1 = torch.max(p, dim=-1)
    vq, pred2 = torch.max(q, dim=-1)
    p_mask = (vp >= cod)
    q_mask = (vq >= cod)
    mask1 = (pred1 == pred2)
    mask = (mask1 & p_mask) & (q_mask)
    mask[0:size] = True
    mask = mask.detach()

    pred1[0:size] = label
    label = (pred1[mask]).detach()
    feature = FF[mask]
    return label, feature


def get_centroid(feature, labels, n_class):
    batch, dim = feature.shape


    ones = (torch.ones_like(labels, dtype=torch.float)).cuda()
    zeros = (torch.zeros(n_class)).cuda()
    s_n_classes = zeros.scatter_add(0, labels, ones)


    zeros = torch.zeros(n_class, dim).cuda()
    s_sum_feature = (zeros.scatter_add(0, torch.transpose(labels.repeat(dim, 1), 1, 0), feature)).cuda()


    current_s_centroid = torch.div(s_sum_feature, (s_n_classes.view(n_class, 1) + 0.0000001))
    return current_s_centroid


def get_pseudo_centroid(feature, p, q, label, size, n_class, cod):
    pseudo_label, mask = get_pseudo_label(p, q, label, size, cod)
    pseudo_s, pseudo_t1, pseudo_t2, pseudo_t3 = pseudo_label.chunk(4, dim=0)
    mask_s, mask_t1, mask_t2, mask_t3 = mask.chunk(4, dim=0)
    feature_s, feature_t1, feature_t2, feature_t3 = feature.chunk(4, dim=0)

    feature_s = feature_s[mask_s]
    feature_t1 = feature_t1[mask_t1]
    feature_t2 = feature_t2[mask_t2]
    feature_t3 = feature_t3[mask_t3]

    label_s = pseudo_s[mask_s]
    label_t1 = pseudo_t1[mask_t1]
    label_t2 = pseudo_t2[mask_t2]
    label_t3 = pseudo_t3[mask_t3]

    s_centroid = get_centroid(feature_s, label_s, n_class)
    t1_centroid = get_centroid(feature_t1, label_t1, n_class)
    t2_centroid = get_centroid(feature_t2, label_t2, n_class)
    t3_centroid = get_centroid(feature_t3, label_t3, n_class)

    centroid = torch.cat([s_centroid, t1_centroid, t2_centroid, t3_centroid], dim=0)
    c_labels = torch.arange(0, n_class)
    c_labels = torch.cat([c_labels, c_labels, c_labels, c_labels]).cuda()
    return centroid, c_labels


def get_sig_centroid(feature, p, label, size, n_class):
    pseudo_label, mask = get_sig_label(p, label, size, cod=0.8)
    pseudo_s, pseudo_t1, pseudo_t2, pseudo_t3 = pseudo_label.chunk(4, dim=0)
    mask_s, mask_t1, mask_t2, mask_t3 = mask.chunk(4, dim=0)
    feature_s, feature_t1, feature_t2, feature_t3 = feature.chunk(4, dim=0)

    feature_s = feature_s[mask_s]
    feature_t1 = feature_t1[mask_t1]
    feature_t2 = feature_t2[mask_t2]
    feature_t3 = feature_t3[mask_t3]

    label_s = pseudo_s[mask_s]
    label_t1 = pseudo_t1[mask_t1]
    label_t2 = pseudo_t2[mask_t2]
    label_t3 = pseudo_t3[mask_t3]

    s_centroid = get_centroid(feature_s, label_s, n_class)
    t1_centroid = get_centroid(feature_t1, label_t1, n_class)
    t2_centroid = get_centroid(feature_t2, label_t2, n_class)
    t3_centroid = get_centroid(feature_t3, label_t3, n_class)

    centroid = torch.cat([s_centroid, t1_centroid, t2_centroid, t3_centroid], dim=0)
    c_labels = torch.arange(0, n_class)
    c_labels = torch.cat([c_labels, c_labels, c_labels, c_labels]).cuda()
    return centroid, c_labels


def get_sig_label(p, label, size, cod):
    p = F.softmax(p, dim=-1)
    vp, pred = torch.max(p, dim=-1)
    mask = (vp >= cod)
    pseudo_label = pred * mask
    pseudo_label[0:size] = label
    mask[0:size] = True
    return pseudo_label, mask



def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("Successfully created folder")
    else:
        print("The folder already exists")


def get_split_sys():
    if platform.system().lower() == 'windows':
        split_sym = '\\'
    else:
        split_sym = '/'
    return split_sym


def requires_grad(models, flag=True):
    for p in models.parameters():
        p.requires_grad = flag


def collect_data(data_loader, max_num_features=None):
    all_data = []
    all_fault = []
    for i, data in enumerate(tqdm.tqdm(data_loader)):
        if max_num_features is not None and i >= max_num_features:
            break
        inputs_data, fault_labels, _, = data
        all_data.append(inputs_data)
        all_fault.append(fault_labels)
    all_data = (torch.cat(all_data, dim=0)).numpy()
    all_fault = (torch.cat(all_fault, dim=0)).numpy()
    return all_data, all_fault


def collect_feature(feature_extractor, data_loader, max_num_features=None):
    feature_extractor.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            inputs, labels, _, = data
            inputs = (inputs.reshape([-1, 1, 32, 32])).cuda()
            _, feature = feature_extractor(inputs)
            feature = feature.cpu()
            all_features.append(feature)
            all_labels.append(labels)
    F = (torch.cat(all_features, dim=0)).numpy()
    L = (torch.cat(all_labels, dim=0)).numpy()
    return F, L



