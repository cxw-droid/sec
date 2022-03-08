import torch
from sklearn.model_selection import StratifiedKFold
import re
import pickle

def pickle_to_d(f_name):
    with open(f_name, 'rb') as f:
        d = pickle.load(f)
    return d

def stat_num(g_dict, g_sub_label=False):
    """
    output the number of graphs of different labels
    @param g_dict:
    @param g_sub_label:
    """
    d = {}
    for k, v in g_dict.items():
        label = v['graph_sub_label'] if g_sub_label else v['label']
        d[label] = d.get(label, 0) + 1
    for k, v in d.items():
        print(f'{v} graphs of label {k}')

def sub_label(f_name, node_label, node_prop):
    """
    get specific sub node label for each node
    @param f_name: the graph.json file
    @param node_label: a list of the general node labels
    @param node_prop: a list of node properties
    @return:
    """
    sub_labels = []
    for idx, n_label in enumerate(node_label):
        if n_label == 'ProcessNode':
            try:
                _p = '\\\\' if '\\' in node_prop[idx]['exe_name'] else '/'
                n_sub_label = re.split(_p, node_prop[idx]['exe_name'])[-1]
            except:
                print(f'*** graph: {f_name}\nnode id: {idx}\n'
                               f'node label: {n_label}\nnode prop: {node_prop[idx]}\n')
                n_sub_label = n_label
        elif n_label == 'SocketChannelNode':
            l_addr = node_prop[idx]['local_inet_addr']
            r_addr = node_prop[idx]['remote_inet_addr']
            p_re = r'^(127|10|172.16|192.168)'
            if re.match(p_re, l_addr) and re.match(p_re, r_addr):
                n_sub_label = '00'
            elif re.match(p_re, l_addr) and not re.match(p_re, r_addr):
                n_sub_label = '01'
            elif not re.match(p_re, l_addr) and re.match(p_re, r_addr):
                n_sub_label = '10'
            elif not re.match(p_re, l_addr) and not re.match(p_re, r_addr):
                n_sub_label = '11'
            else:
                raise NotImplementedError
        else:
            n_sub_label = n_label
        sub_labels.append(n_sub_label)
    return sub_labels

def k_fold(dataset, folds):
    """
    to generate k fold dataset
    @param dataset:
    @param folds: the number k of folders
    @return:
    """
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices

if __name__ == '__main__':
    pass