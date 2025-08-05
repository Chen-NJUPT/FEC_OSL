__author__ = 'HPC'
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
import torch as th
import copy
import random
import time
pad_length = 11           
mtu = 1000                  
mtime = 10                  
import argparse
def pad_sequence(source,pad_length,pad_value):
    if len(source) > pad_length:
        return  source[:pad_length]
    else:
        source = source + [pad_value] * (pad_length - len(source))
        return  source
    
    
def build_graph(sample, all_classes, nb_classes):
    graph = dgl.DGLGraph()
    graph.add_nodes(len(sample['nodes']))
    pad_length = len(sample['nodes'][0]['all_info'])
    all_info_matrix = np.zeros(shape=(len(sample['nodes']), pad_length),dtype =np.double)   
    for i in range(len(sample['nodes'])):
        all_info_matrix[i] = pad_sequence(sample['nodes'][i]['all_info'],pad_length,0)
    all_info_matrix = np.reshape(all_info_matrix,(-1,1,pad_length))
    
    with open("dataset/flow_data/Normalize/flow_value_scope_" + str(nb_classes) + "_" + str(all_classes - nb_classes) + ".json", 'r') as f:
        data = json.load(f)
        for key, value in data.items():
            if (value["max_value"] - value["min_value"]) != 0:
                all_info_matrix[:, 0, int(key)] -= value["min_value"]
                all_info_matrix[:, 0, int(key)] /= (value["max_value"] - value["min_value"])
            else:
                all_info_matrix[:, 0, int(key)] = 1
                
    graph.ndata['all_info'] = th.from_numpy(all_info_matrix)   
    graph.ndata['burst_id'] = th.from_numpy(np.array([node['burst_id'] for node in sample['nodes']]))  
    
    for each in sample['edges']:
        graph.add_edges(th.tensor(each[0]),th.tensor(each[1]))
        
    graph.edata['attn'] = th.from_numpy(np.random.randn(len(sample['edges']),1))
    return graph

sni_blacklist = []
concurrent_time_threshold = 1   


def build_graphs_from_flowcontainer_json(filename, time_period, concurrent_time_threshold=concurrent_time_threshold, all_classes = 8, nb_classes = 6):         
    with open(filename) as fp:
        sample = json.load(fp)
    try:
        if len(sample) == 0:
            raise BaseException('Empty graph!')
        
        for each in sample:
            if 'start_timestamp' not in each:
                each['start_timestamp'] = min(each['timestamp'])
            if 'arrive_time_delta' not in each :
                each['arrive_time_delta'] = [0] + [each['timestamp'][i]- each['timestamp'][i-1] for i in range(1, len(each['timestamp']))]
        
        sample.sort(key=lambda x : x['start_timestamp'])   
        _graphs = []
        graphs = []
        for i in range(len(sample)):
            if 'sni' in sample[i] and sample[i]['sni'] in sni_blacklist:
                continue
            if len(_graphs)== 0 :
                _graphs.append([sample[i]])
            else:
                if sample[i]['start_timestamp'] - _graphs[-1][0]['start_timestamp'] < time_period:
                    _graphs[-1].append(sample[i])   
                else:
                    _graphs.append([sample[i]])
        for S in _graphs:
            nodes = []
            edges = []
            burst = []
            last_burst = burst
            bursts = []   
            for index, flow in enumerate(S):
                flow['id'] = index 
                if len(burst) == 0:
                    burst.append(flow)
                else:
                    if ((flow['packet_length'] > 0 and burst[-1]['packet_length'] > 0) or
                         (flow['packet_length'] < 0 and burst[-1]['packet_length'] < 0)):
                        burst.append(flow)
                    else:
                        for j in range(len(burst)-1):  
                            edges.append((burst[j]['id'],burst[j + 1]['id']))
                            edges.append((burst[j + 1]['id'], burst[j]['id']))
                        if len(last_burst) > 0:
                            if len(burst) > 1:
                                edges.append((last_burst[0]['id'], burst[0]['id']))
                                edges.append((last_burst[-1]['id'], burst[-1]['id']))
                                edges.append((burst[0]['id'], last_burst[0]['id']))
                                edges.append((burst[-1]['id'], last_burst[-1]['id']))
                            elif len(burst) == 1 and last_burst != burst:         
                                edges.append((last_burst[-1]['id'], burst[0]['id']))
                                edges.append((burst[0]['id'], last_burst[-1]['id']))
                        last_burst = burst
                        burst = [flow]
                        bursts.append(last_burst)
                flow['burst_id'] = len(bursts)   
                nodes.append(flow)
            if len(burst) > 0:
                for j in range(len(burst)-1):
                    edges.append((burst[j]['id'],burst[j + 1]['id']))
                    edges.append((burst[j + 1]['id'], burst[j]['id']))
                if len(last_burst)> 0:
                    if len(burst) > 1:
                        edges.append((last_burst[0]['id'], burst[0]['id']))
                        edges.append((last_burst[-1]['id'], burst[-1]['id']))
                        edges.append((burst[0]['id'], last_burst[0]['id']))
                        edges.append((burst[-1]['id'], last_burst[-1]['id']))
                    elif len(burst) == 1 and last_burst != burst:         
                        edges.append((last_burst[-1]['id'], burst[0]['id']))
                        edges.append((burst[0]['id'], last_burst[-1]['id']))
                bursts.append(burst)
            graph = build_graph({
                'nodes': nodes,
                'edges': edges,
            }, all_classes, nb_classes)
            graphs.append(graph)
        return  graphs
    except BaseException as exp:
        info='build ill graph from {0}'.format(filename)
        return [None]
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='flow model')
    parser.add_argument('--dataset', '-d', type=str, help='dataset name', default="USTC_2016", dest='dataset')
    args = parser.parse_args()
    dataset = args.dataset
    print("using dataset",dataset)
    gs = build_graphs_from_flowcontainer_json(
        r'dataset/flow_data/' + dataset,
        time_period=1
    )[0]
    model_name = 'flow_model'
    data_path = 'data/' + model_name + '_' + dataset
    import os
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    nx_G = gs.to_networkx()
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True)
    plt.savefig("images/graph.png", format="png")
    print("finished")