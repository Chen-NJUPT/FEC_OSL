__author__ = 'HPC'

import random
import time
import gzip
import os
import sys
import dgl
import pickle
import torch as th
import numpy as np
from construct_graph import build_graphs_from_flowcontainer_json,mtu,mtime, concurrent_time_threshold
import tqdm
from tqdm import tqdm
from tqdm import trange
import argparse
import json

root_dir  =  "dataset/"
random.seed(100)
time_period= 1

class FlowContainerJSONDataset:
    def __init__(self, graph_json_directory=root_dir+"tor_session_JSON/",mode='clear',dumpData=False,usedumpData=False,dumpFilename="dataset_builder.pkl_know.gzip",cross_version=False,test_split_rate=0.1, all_classes = 8, nb_classes = 6):

        self.dumpFileName = dumpFilename
        if usedumpData==True and os.path.exists(dumpFilename):
            fp = gzip.GzipFile(dumpFilename,"rb")
            data=pickle.load(fp)
            fp.close()
            self.labelName = data['labelName']
            self.labelNameSet=data['labelNameSet']
            self.graphs = data['graphs']
            self.labelId = data['labelId']
            self.know_train_index = data['know_train_index']
            self.know_test_index = data['know_test_index']
            self.know_valid_index = data['know_valid_index']
            self.unknow_train_index = data['unknow_train_index']
            self.unknow_test_index = data['unknow_test_index']
            self.unknow_valid_index = data['unknow_valid_index']
            info ='Load dump data from {0}'.format(dumpFilename)
            
        else:
            if os.path.isdir(graph_json_directory)== False:
                info = '{0} is not a directory'.format(graph_json_directory)
                raise BaseException(info)
            assert mode in ['clear','noise','all']
            self.labelName = []
            self.labelNameSet = {}
            self.labelId = []
            self.graphs = []
            
            _labelNameSet = []  
            for _root,_dirs,_files in os.walk(graph_json_directory):
                
                if _root == graph_json_directory or len(_files) == 0:
                    continue
                _root =_root.replace("\\","/")
                packageName = _root.split("/")[-1]
                labelName=packageName
                _labelNameSet.append(labelName)
            _labelNameSet.sort()
            for i in range(len(_labelNameSet)):
                self.labelNameSet.setdefault(_labelNameSet[i], len(self.labelNameSet))

            for labelName in _labelNameSet:
                folder_path = os.path.join(graph_json_directory, labelName).replace("\\", "/")

                if not os.path.isdir(folder_path):  
                    continue

                _files = os.listdir(folder_path)  
                for index in range(len(_files)):
                    file = _files[index]
                    if file == ".DS_Store":  
                        continue

                    json_fname = os.path.join(folder_path, file).replace("\\", "/")
                    
                    gs = build_graphs_from_flowcontainer_json(json_fname, time_period=time_period, all_classes = all_classes, nb_classes = nb_classes)
                    if len(gs) < 1 or gs[0] is None:
                        continue
                    
                    self.graphs += gs
                    self.labelName += [labelName] * len(gs)  
                    self.labelId += [self.labelNameSet[labelName]] * len(gs)  

                    assert self.labelId[-1] in range(len(self.labelNameSet))
            assert len(self.graphs) == len(self.labelId)
            assert len(self.graphs) == len(self.labelName)
            info = "Build {0} graph over {1} classes, {2} graph per class. {3} flow.".format(len(self.graphs),len(self.labelNameSet),len(self.graphs)//len(self.labelNameSet),self.flowCounter)
            
            self.know_train_index = []
            self.know_valid_index = []
            self.know_test_index =  []
            self.unknow_train_index = []
            self.unknow_valid_index = []
            self.unknow_test_index =  []

            with open('dataset/byte_data/data_index_' + str(nb_classes) + "_" + str(all_classes - nb_classes) +  '.json', 'r') as json_file:
                temp = json.load(json_file)  
                self.know_train_index = temp['know_train_indices']
                self.know_valid_index = temp['know_val_indices']
                self.know_test_index =  temp['know_val_indices']
                
                self.unknow_train_index = temp['all_unknow_train_indices']
                self.unknow_valid_index = temp['all_unknow_val_indices']
                self.unknow_test_index =  temp['unknow_val_indices']
                
            if dumpData :
                self.dumpData()    
        self.class_aliasname ={}        
        labelNameSet = list(self.labelNameSet)
        
        labelNameSet.sort()             
        for i in range(len(labelNameSet)):
            self.class_aliasname.setdefault(i,labelNameSet[i])
        
        self.train_watch = 0
        self.test_watch =  0
        self.valid_watch = 0
        self.epoch_over = False

    def dumpData(self,dumpFileName=None):
        if dumpFileName == None:
            dumpFileName = self.dumpFileName
        fp = gzip.GzipFile(dumpFileName,"wb")
        pickle.dump({
                'graphs':self.graphs,
                'flowCounter':self.flowCounter,
                'labelName':self.labelName,
                'labelNameSet':self.labelNameSet,
                'labelId':self.labelId,
                'know_train_index':self.know_train_index,
                'know_valid_index':self.know_valid_index,
                'know_test_index':self.know_test_index,
                'unknow_train_index':self.unknow_train_index,
                'unknow_valid_index':self.unknow_valid_index,
                'unknow_test_index':self.unknow_test_index
                
            },file=fp,protocol=-1)
        fp.close()

    
    
    
    
    
    
    

    
    

    
    
    
    
    
    
    
    
    
    
    
    
    def reflesh(self):
        self.train_watch = 0
        
    def __next_batch(self, name, batch_size):
        graphs = []
        labels = []

        if name == 'train':
            
            remaining = len(self.train_index) - self.train_watch
            current_batch_size = min(batch_size, remaining)  

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.train_index[self.train_watch]])
                labels.append(self.labelId[self.train_index[self.train_watch]])
                self.train_watch += 1

            
            if self.train_watch >= len(self.train_index):
                self.epoch_over += 1
                self.train_watch = 0  

        elif name == 'valid':
            remaining = len(self.valid_index) - self.valid_watch
            current_batch_size = min(batch_size, remaining)

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.valid_index[self.valid_watch]])
                labels.append(self.labelId[self.valid_index[self.valid_watch]])
                self.valid_watch += 1

            if self.valid_watch >= len(self.valid_index):
                self.valid_watch = 0

        else:
            remaining = len(self.test_index) - self.test_watch
            current_batch_size = min(batch_size, remaining)

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.test_index[self.test_watch]])
                labels.append(self.labelId[self.test_index[self.test_watch]])
                self.test_watch += 1

            if self.test_watch >= len(self.test_index):
                self.test_watch = 0

        return dgl.batch(graphs), th.tensor(labels)

    def next_train_batch(self,batch_size):
        return self.__next_batch('train',batch_size)
    
    def next_valid_batch(self,batch_size):
        return self.__next_batch('valid',batch_size)
    
    def next_test_batch(self,batch_size):
        return self.__next_batch('test',batch_size)
    
    
    
    def __next_batch_know(self, name, batch_size):
        graphs = []
        labels = []

        if name == 'train':
            
            remaining = len(self.know_train_index) - self.train_watch
            current_batch_size = min(batch_size, remaining)  
            for i in range(current_batch_size):
                graphs.append(self.graphs[self.know_train_index[self.train_watch]])
                labels.append(self.labelId[self.know_train_index[self.train_watch]])
                self.train_watch += 1

            
            if self.train_watch >= len(self.know_train_index):
                self.epoch_over += 1
                self.train_watch = 0  

        elif name == 'valid':
            remaining = len(self.know_valid_index) - self.valid_watch
            current_batch_size = min(batch_size, remaining)

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.know_valid_index[self.valid_watch]])
                labels.append(self.labelId[self.know_valid_index[self.valid_watch]])
                self.valid_watch += 1

            if self.valid_watch >= len(self.know_valid_index):
                self.epoch_over += 1
                self.valid_watch = 0

        else:
            remaining = len(self.know_test_index) - self.test_watch
            current_batch_size = min(batch_size, remaining)

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.know_test_index[self.test_watch]])
                labels.append(self.labelId[self.know_test_index[self.test_watch]])
                self.test_watch += 1

            if self.test_watch >= len(self.know_test_index):
                self.epoch_over += 1
                self.valid_watch = 0

        return dgl.batch(graphs), th.tensor(labels)

    def next_train_batch_know(self,batch_size):
        return self.__next_batch_know('train', batch_size)
    
    def next_valid_batch_know(self,batch_size):
        return self.__next_batch_know('valid', batch_size)
    
    def next_test_batch_know(self,batch_size):
        return self.__next_batch_know('test', batch_size)
    
    
    def __next_batch_unknow(self, name, batch_size):
        graphs = []
        labels = []

        if name == 'train':
            
            remaining = len(self.unknow_train_index) - self.train_watch
            current_batch_size = min(batch_size, remaining)  

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.unknow_train_index[self.train_watch]])
                labels.append(self.labelId[self.unknow_train_index[self.train_watch]])
                self.train_watch += 1

            
            if self.train_watch >= len(self.unknow_train_index):
                self.epoch_over += 1
                self.train_watch = 0  

        elif name == 'valid':
            remaining = len(self.unknow_valid_index) - self.valid_watch
            current_batch_size = min(batch_size, remaining)

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.unknow_valid_index[self.valid_watch]])
                labels.append(self.labelId[self.unknow_valid_index[self.valid_watch]])
                self.valid_watch += 1

            if self.valid_watch >= len(self.unknow_valid_index):
                self.epoch_over += 1
                self.valid_watch = 0

        else:
            remaining = len(self.unknow_test_index) - self.test_watch
            current_batch_size = min(batch_size, remaining)

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.unknow_test_index[self.test_watch]])
                labels.append(self.labelId[self.unknow_test_index[self.test_watch]])
                self.test_watch += 1

            if self.test_watch >= len(self.unknow_test_index):
                self.epoch_over += 1
                self.test_watch = 0

        return dgl.batch(graphs), th.tensor(labels)

    def next_train_batch_unknow(self,batch_size):
        return self.__next_batch_unknow('train',batch_size)
    
    def next_valid_batch_unknow(self,batch_size):
        return self.__next_batch_unknow('valid',batch_size)
    
    def next_test_batch_unknow(self,batch_size):
        return self.__next_batch_unknow('test',batch_size)
    
    
    def export_wf_dataset(self,path_dir,feature_name='pkt_length'):
        
        if os.path.exists(path_dir)== False:
            os.makedirs(path_dir)
        assert  feature_name in ['pkt_length','arv_time']
        X_train =[]
        y_train =[]
        X_valid =[]
        y_valid =[]
        X_test =[]
        y_test =[]
        
        
        
        
        
        for i in self.train_index:
            X_train.append(self.graphs[i].ndata[feature_name]*mtu)
            y_train += [self.labelId[i]] * len(self.graphs[i].nodes())
        for i in self.test_index:
            X_test.append(self.graphs[i].ndata[feature_name]*mtu)
            y_test += [self.labelId[i]] * len(self.graphs[i].nodes())
        for i in self.valid_index:
            X_valid.append(self.graphs[i].ndata[feature_name]*mtu)
            y_valid += [self.labelId[i]] * len(self.graphs[i].nodes())

        
        X_train = np.concatenate(X_train)
        X_test = np.concatenate(X_test)
        X_valid =np.concatenate(X_valid)

        X_train = np.reshape(X_train,(-1,1000,1))
        X_test = np.reshape(X_test,(-1,1000,1))
        X_valid = np.reshape(X_valid,(-1,1000,1))

        with gzip.GzipFile(path_dir+"/"+"X_train_"+feature_name+".pkl","wb") as fp:
            pickle.dump(X_train,fp,-1)
        with gzip.GzipFile(path_dir+"/"+"X_valid_"+feature_name+".pkl","wb") as fp:
            pickle.dump(X_valid,fp,-1)
        with gzip.GzipFile(path_dir+"/"+"X_test_"+feature_name+".pkl","wb") as fp:
            pickle.dump(X_test,fp,-1)

        with gzip.GzipFile(path_dir+"/"+"y_train_"+feature_name+".pkl","wb") as fp:
            pickle.dump(y_train,fp,-1)
        with gzip.GzipFile(path_dir+"/"+"y_valid_"+feature_name+".pkl","wb") as fp:
            pickle.dump(y_valid,fp,-1)
        with gzip.GzipFile(path_dir+"/"+"y_test_"+feature_name+".pkl","wb") as fp:
            pickle.dump(y_test,fp,-1)

        print('export {0} flows'.format(X_train.shape[0]))
        assert  X_train.shape[0] ==len(y_train)
        assert  X_valid.shape[0] ==len(y_valid)
        assert  X_test.shape[0] ==len(y_test)

    def export_flow_dataset(self,path_dir,feature_name='pkt_length'):
        print('export to flow format')
        if os.path.exists(path_dir) == False:
            os.makedirs(path_dir)
        assert  feature_name in ['pkt_length','arv_time']
        flowCounter = 0
        for i in range(len(self.labelName)):
            
            package_name = self.labelName[i]
            
            feature_matrix =mtu * self.graphs[i].ndata[feature_name]
            fp = open(path_dir+package_name+".num","a")
            for j in range(feature_matrix.shape[0]):
                feature = ";"+"\t".join([str(int(feature_matrix[j][0][i__])) for i__ in range(feature_matrix.shape[2])])+"\t;\n"
                fp.writelines(feature)
                flowCounter+=1
            fp.close()
        print('export {0} flows'.format(flowCounter))
    @property
    def flowCounter(self):
        flowcounter = 0
        for i in range(len(self.labelName)):
            
            flowcounter += self.graphs[i].ndata['all_info'].shape[0]
        return  flowcounter


if __name__ == '__main__':

    '''
    '''
    

    parser = argparse.ArgumentParser(description='flow model')
    parser.add_argument('--dataset', '-d', type=str, help='dataset name', required=True, dest='dataset')
    args = parser.parse_args()
    dataset = args.dataset
    print("dataset:",dataset)

    graph_json_directory = os.path.join("../../../","dataset",dataset)
    if not os.path.exists(graph_json_directory):
        print(f"data_path {graph_json_directory} does not exist!")
        raise 1
    print("graph_json_directory:",graph_json_directory)
    
    data_path = './data/flow' + '_' + dataset
    dumpFilename = data_path+"/dataset_builder.pkl.gzip"
    if not os.path.exists(data_path):
            os.makedirs(data_path)
    print("dumpFilename:",dumpFilename)

    dataset = FlowContainerJSONDataset(mode='clear',
                      dumpData=True,usedumpData=False,
                      dumpFilename=dumpFilename,
                      cross_version=False,
                      test_split_rate=0.1,
                      graph_json_directory = graph_json_directory)
    del dataset

    
    
    
    
    
    
    
