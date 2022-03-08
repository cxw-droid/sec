import glob
import json
import random
import numpy as np
import os
import re
import shutil
from tqdm import tqdm
from utils import sub_label, pickle_to_d, stat_num

def find_files_json(dir, f_name=None, sampling=None, prefix=None):
    """
    find all the graph.json files and return all their paths
    @param dir: the fold containg all the graph.json files
    @param f_name: the name of graph json file, usually 'graph.json'
    @param sampling:
    @param prefix:
    @return:
    """
    if not sampling:
        return glob.glob(dir + '/**/' + f_name, recursive=True)
    else:  # sampling may not work for different folder structures
        num = int(prefix.split('_')[-1])
        f_benign = []
        dir_malicious = os.path.join(dir, 'malicious')
        f_malicous = glob.glob(dir_malicious + '/**/' + f_name, recursive=True)
        print(f'{len(f_malicous)} malicious files found')
        dir_benign = os.path.join(dir, 'benign')
        folders = [name for name in os.listdir(dir_benign)
                if os.path.isdir(os.path.join(dir, 'benign', name))]
        print(f'{len(folders)} folders: {folders}')
        for d_name in tqdm(folders):
            cur_dir = os.path.join(dir, 'benign', d_name)
            print(f'at folder {cur_dir}')
            if os.path.isdir(cur_dir):
                subdirs = [name for name  in os.listdir(cur_dir) if os.path.isdir(os.path.join(dir_benign, cur_dir, name))]
                if len(subdirs) > num:
                    subdirs = random.sample(subdirs, num)
                    print(f'{subdirs} sampled: {len(subdirs)}')
                else:
                    print(f'{subdirs} fully used: {len(subdirs)}')
                for sub_ in subdirs:
                    f_glob = glob.glob(os.path.join(dir_benign, cur_dir, sub_)  + '/**/' + f_name, recursive=True)
                    if not len(f_glob) == 1: print(os.path.join(dir_benign, cur_dir, sub_) + ' no graph files---')
                    f_benign.extend(f_glob)
        f_malicous = random.sample(f_malicous, len(f_benign))
        return  f_malicous + f_benign

def one_graph_json(f=None, graph_sub_label='_none'):
    """
    process one graph.json file
    @param f: the path of graph.json file
    @param graph_sub_label:
    @return:
    """
    with open(f, 'r') as json_file:
        data = json.load(json_file)
    node_id = []
    node_label = []
    node_prop = []
    edges = []
    edge_label = []
    edge_prop = []
    # if 'benign' in f and ('malicious' in f or 'anomaly' in f)\
    #         or not 'benign' in f and not 'malicious' in f and not 'anomaly' in f:
    #     raise ValueError(f + ' file path constains both benign and malicious or neither of them')

    # label = 0 if 'benign' in f else 1

    if '/a_' in f and '/b_' in f or not '/a_' in f and not '/b_' in f:
        raise  ValueError(f'{f} vs "/a_" or "/b_"')
    label = 0 if '/b_' in f else 1
    for node in data['vertices']:
        node_id.append(node['_id'])
        node_t = node['TYPE']['value']
        node_label.append(node_t)
        if node_t == 'ProcessNode':
            node_prop.append({'ref_id': node['REF_ID']['value'], 'cmd': node['CMD']['value'], 'exe_name': node['EXE_NAME']['value'], 'pid': node['PID']['value']})
        elif node_t == 'FileNode':
            node_prop.append(({'ref_id': node['REF_ID']['value'], 'file_name': node['FILENAME_SET']['value'][0]['value']}))
        elif node_t == 'SocketChannelNode':
            node_prop.append({'ref_id': node['REF_ID']['value'], 'local_inet_addr':node['LOCAL_INET_ADDR']['value'],
                              'local_port': node['LOCAL_PORT']['value'], 'remote_inet_addr': node['REMOTE_INET_ADDR']['value'],
                              'remote_port':node['REMOTE_PORT']['value']})
        else:
            raise  NotImplementedError(f'node_t is {node_t}')
    node_sub_label = sub_label(f, node_label, node_prop)
    for edge in data['edges']:
        edges.append([edge['_outV'], edge['_inV']])
        edge_label.append(edge['_label'])
    return {'node_id':node_id, 'node_label':node_label, 'node_prop': node_prop, 'node_sub_label': node_sub_label,
                'edges':edges, 'edge_label':edge_label, 'edge_prop': edge_prop, 'label':label, 'graph_sub_label':graph_sub_label}


def sample_dict(d, num):
    keys = random.sample(d.keys(), num)
    return {k:d[k] for k in keys}

def all_graphs_json(folder='../data/SEC', f_name='merged.json', sampling=-1, prefix=None, g_sub_label=False, pickle_f=None, ignore_none=True):
    """
    process all the graph.json files
    @param folder:
    @param f_name:
    @param sampling:
    @param prefix:
    @param g_sub_label:
    @param pickle_f:
    @param ignore_none:
    @return:
    """
    graphs = {}
    f_list = find_files_json(folder, f_name=f_name, sampling=1 if sampling==1 else None, prefix=prefix)
    print(f'Proecessed {len(f_list)} files')
    d_pickle = pickle_to_d(pickle_f) if g_sub_label else {}
    d_graph = {}
    for f in tqdm(f_list, desc='all graphs json files'):
        d_pickle_k = f.split('/')[-2]
        d_graph_k = 'none' if d_pickle_k not in d_pickle else d_pickle[d_pickle_k]
        if ignore_none and d_graph_k == 'none':
            continue
        if d_graph_k not in d_graph:
            d_graph[d_graph_k] = len(d_graph)
        graphs[f] = one_graph_json(f, graph_sub_label=d_graph[d_graph_k])
    if sampling == 0:
        percent = 0.07
        print(f'Sampling {int(len(f_list) * percent)} files')
        graphs_benign = {k:v for k, v in graphs.items() if v['label'] == 0}
        graphs_anomalous = {k: v for k, v in graphs.items() if v['label'] == 1}
        graphs_benign = sample_dict(graphs_benign, int(len(graphs_benign) * percent))
        graphs_anomalous = sample_dict(graphs_anomalous, int(len(graphs_anomalous) * percent))
        graphs = {**graphs_benign, **graphs_anomalous}
    print(f'd_graph: {d_graph}')
    stat_num(graphs, g_sub_label=g_sub_label)
    f_graph_json = f'graphs_{prefix}.json'
    with open(f_graph_json, 'w') as json_file:
        json.dump(graphs, json_file)
    return f_graph_json

def extract_data(f=None, prefix='SEC', n_sub_label=True, g_sub_label=False):
    """
    generate TU format dataset based on the generated json file
    @param f: the generated json file which contains all the graphs
    @param prefix:
    @param n_sub_label:
    @param g_sub_label:
    """
    # load the graphs json file
    with open(f, 'r') as json_file:
        data = json.load(json_file)

    f_dict = {'adj':'A.txt', 'label':'graph_labels.txt', 'indicator':'graph_indicator.txt', 'node_attr':'node_attributes.txt', 'node_label':'node_labels.txt', 'edge_label':'edge_labels.txt', 'edge_attr':'edge_attributes.txt'}
    path = '../data/TU/' + prefix
    if not os.path.exists(path): os.mkdir(path)

    n_label_dict = {}
    e_label_dict = {}
    adj = []
    node_label = []
    edge_label = []
    indicator = []
    label = []

    cur_node_idx = 0
    for i, item in tqdm(enumerate(data.items()), desc='extract_data'):
        g, v = item
        nodes = v['node_id']
        node_map = {node_id: idx + 1 + cur_node_idx for idx, node_id in enumerate(nodes)}  # node_id starts from 1
        for _ in nodes:
            indicator.append(i + 1)  # graph indicator starts from 1

        for src, target in v['edges']:
            adj.append([node_map[src], node_map[target]])  # node index start from 1

        if n_sub_label:
            node_labels = v['node_sub_label']
        else:
            node_labels = v['node_label']
        for n_label in node_labels:
            if not n_label in n_label_dict:
                n_label_dict[n_label] = len(n_label_dict)
            node_label.append(n_label_dict[n_label])

        if 'edge_label' in v:
            for e_label in v['edge_label']:
                if not e_label in e_label_dict:
                    e_label_dict[e_label] = len(e_label_dict)
                edge_label.append(e_label_dict[e_label])

        cur_node_idx += len(nodes)  # update cur_node_idx
        if g_sub_label:
            label.append(v['graph_sub_label'])
        else:
            label.append(v['label'])

    print(f'n_label_dict: {n_label_dict}')
    print(f'e_label_dict: {e_label_dict}')
    print(f'graph labels: {set(label)}')

    np.savetxt('{}'.format(f_dict['adj']), adj, fmt='%i', delimiter=',')  # fmt: integers
    np.savetxt('{}'.format(f_dict['node_label']), node_label, fmt='%i', delimiter=',')
    if edge_label:
        np.savetxt('{}'.format(f_dict['edge_label']), edge_label, fmt='%i', delimiter=',')
    elif os.path.exists(f_dict['edge_label']):
        os.remove(f_dict['edge_label'])
    np.savetxt('{}'.format(f_dict['indicator']), indicator, fmt='%i', delimiter=',')
    np.savetxt('{}'.format(f_dict['label']), label, fmt='%i', delimiter=',')

    raw_path = os.path.join(path, 'raw')
    if os.path.exists(raw_path):
        old_path = raw_path + '_old'
        if os.path.exists(old_path): shutil.rmtree(old_path)
        os.rename(raw_path, old_path)
    os.mkdir(raw_path)
    processed_path = os.path.join(path, 'processed')
    if os.path.exists(processed_path): shutil.rmtree(processed_path)
    for k, v in f_dict.items():
        if os.path.exists(v):
            shutil.copy(v, os.path.join(path, 'raw', prefix + '_' + v))

def main(folder=None, f_name=None, prefix=None, sampling=None, n_sub_label=False, g_sub_label=False, pickle_f=None, ignore_none=True):
    """

    @param folder: the folder containing the graph json files extracted from the database
    @param f_name: the json file name in the folder, usullay 'graph.json'
    @param prefix: the name of the dataset to generate
    @param sampling: use sampling or not
    @param n_sub_label: use specific node label or not
    @param g_sub_label: use specific graph label ('A', 'B' ...) or not
    @param pickle_f: the pickle file that contains a dictionary of json graph file types
    @param ignore_none: ignore non 'A', 'B', 'C' and 'D' label or not
    """
    print(f'folder={folder}, f_name={f_name}, prefix={prefix}, sampling={sampling}, n_sub_label={n_sub_label}, g_sub_label={g_sub_label}, pickel_f={pickle_f}')
    f_graph_json = all_graphs_json(folder=folder, f_name=f_name, sampling=sampling, prefix=prefix, g_sub_label=g_sub_label, pickle_f=pickle_f, ignore_none=ignore_none)
    extract_data(f_graph_json, prefix=prefix, n_sub_label=n_sub_label, g_sub_label=g_sub_label)
    print(f'{prefix} Done')

if __name__ == '__main__':
    # main(folder='/tmp/prov-ng/gnn/prune/linux/python/b_nd_python_csv_converted', f_name='graph.json', prefix='python_g_ignore_nn', n_sub_label=False, g_sub_label=True, pickle_f='/tmp/prov-ng/gnn/prune/linux/python/b_nd_python_csv_converted/Graph Label Provenance Graph/graph_labels.pickle')
    main(folder='/tmp/prov-ng/gnn/prune/linux/python/b_nd_python_csv_converted', f_name='graph.json', prefix='python_g_ignore', n_sub_label=True, g_sub_label=True, pickle_f='/tmp/prov-ng/gnn/prune/linux/python/b_nd_python_csv_converted/Graph Label Provenance Graph/graph_labels.pickle')