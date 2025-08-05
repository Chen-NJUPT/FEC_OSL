__author__ = 'HPC'
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torch_kmeans import KMeans
from torch.utils.checkpoint import checkpoint
import dgl
from dgl.nn.pytorch import GraphConv,TAGConv
import timm.models.vision_transformer
from functools import partial

base_dir = os.path.dirname(os.path.abspath(__file__))
print("base_dir = ", base_dir)
byte_extract_path = os.path.join(base_dir, 'byte_features_extract')
sys.path.append(byte_extract_path)
flow_extract_path = os.path.join(base_dir, 'flow_features_extract')
sys.path.append(flow_extract_path)
energy_model_path = os.path.join(base_dir, 'energy_model')
sys.path.append(energy_model_path)

class MultiLevelConv1DFingerprinting(nn.Module):
    def __init__(self,feature_width=32,nb_classes=None):
        #初始化
        super(MultiLevelConv1DFingerprinting,self).__init__()
        filter_num = ['None', 32, 64, 128, 256] 
        kernel_size = ['None', 3, 3, 3, 3]
        conv_stride_size = ['None', 1, 1, 1, 1] 
        pool_stride_size = ['None', 2, 2, 2, 2] 
        pool_size = ['None', 2, 2, 2, 2] 
        self._1Conv1D = nn.Conv1d(stride=conv_stride_size[1],
                                 kernel_size=kernel_size[1],
                                 in_channels=1,
                                 out_channels=filter_num[1],
                                 padding=kernel_size[1]//2)
        self._2BatchNormalization = nn.BatchNorm1d(filter_num[1])
        self._3ELU = nn.ELU(alpha=1.0)

        self._4Conv1D = nn.Conv1d(in_channels=filter_num[1],
                                  out_channels=filter_num[1],
                                  kernel_size = kernel_size[1],
                                  stride=conv_stride_size[1],
                                  padding=kernel_size[1]//2)
        self._5BatchNormalization = nn.BatchNorm1d(filter_num[1])
        self._6ELU = nn.ELU(alpha=1.0)
        self._7MaxPooling1D=nn.MaxPool1d(stride=pool_stride_size[1],
                                         kernel_size=pool_size[1],padding=1)
        self._8Dropout = nn.Dropout(p=0.1)

        self._9Conv1D = nn.Conv1d(in_channels=filter_num[1],
                                  out_channels=filter_num[2],
                                  kernel_size = kernel_size[2],
                                  stride=conv_stride_size[2],
                                  padding=kernel_size[2]//2)
        self._10BatchNormalization = nn.BatchNorm1d(filter_num[2])
        self._11Relu = nn.ReLU()

        self._12Conv1D = nn.Conv1d(in_channels=filter_num[2],
                                  out_channels=filter_num[2],
                                  kernel_size = kernel_size[2],
                                  stride=conv_stride_size[2],
                                  padding=kernel_size[2]//2)
        self._13BatchNormalization = nn.BatchNorm1d(filter_num[2])
        self._14Relu = nn.ReLU()

        self._15MaxPooling1D=nn.MaxPool1d(kernel_size=pool_size[2],
                                          stride=pool_stride_size[2],
                                          padding=pool_size[2]//2)
        self._16Dropout = nn.Dropout(p=0.1)

        self._17Conv1D = nn.Conv1d(in_channels=filter_num[2],
                                  out_channels=filter_num[3],
                                  kernel_size = kernel_size[3],
                                  stride=conv_stride_size[3],
                                  padding=kernel_size[3]//2)
        self._18BatchNormalization = nn.BatchNorm1d(filter_num[3])
        self._19Relu = nn.ReLU()
        
        self._20Conv1D = nn.Conv1d(in_channels=filter_num[3],
                                  out_channels=filter_num[3],
                                  kernel_size = kernel_size[3],
                                  stride=conv_stride_size[3],
                                  padding=kernel_size[3]//2)
        self._21BatchNormalization = nn.BatchNorm1d(filter_num[3])
        self._22Relu = nn.ReLU()

        self._23MaxPooling1D=nn.MaxPool1d(kernel_size=pool_size[3],
                                          stride=pool_stride_size[3],
                                          padding=pool_size[3]//2)
        self._24Dropout = nn.Dropout(p=0.1)

        self._25Conv1D = nn.Conv1d(in_channels=filter_num[3],
                                  out_channels=filter_num[4],
                                  kernel_size = kernel_size[4],
                                  stride=conv_stride_size[4],
                                  padding=kernel_size[4]//4)
        self._26BatchNormalization = nn.BatchNorm1d(filter_num[4])
        self._27Relu = nn.ReLU()
        self._28Conv1D = nn.Conv1d(in_channels=filter_num[4],
                                  out_channels=filter_num[4],
                                  kernel_size = kernel_size[4],
                                  stride=conv_stride_size[4],
                                  padding=kernel_size[4]//2)
        self._29BatchNormalization = nn.BatchNorm1d(filter_num[4])
        self._30Relu = nn.ReLU()
        self._31MaxPooling1D=nn.MaxPool1d(kernel_size=pool_size[4],
                                          stride=pool_stride_size[4],
                                          padding=pool_size[4]//2)
        self._32Dropout = nn.Dropout(p=0.1)

        self._33Flattern = nn.Flatten()
        self._34Dense = nn.Sequential(nn.Linear(in_features=256,out_features=feature_width), nn.ReLU(True))  # f = 4
        
        if nb_classes!= None:
            self._35Sigmoid = nn.Sequential(nn.Linear(in_features=feature_width,out_features=nb_classes),nn.Sigmoid())
        else:
            self._35Sigmoid = None

    def forward(self, x):
        x = self._1Conv1D(x)
        x = self._2BatchNormalization(x)
        x = self._3ELU(x)

        x = checkpoint(self._4Conv1D, x)

        x = self._5BatchNormalization(x)
        x = checkpoint(self._6ELU, x)

        x = checkpoint(self._7MaxPooling1D, x)
        x = self._8Dropout(x)

        x = checkpoint(self._9Conv1D, x)
        x = self._10BatchNormalization(x)
        x = checkpoint(self._11Relu, x)

        x = checkpoint(self._12Conv1D, x)
        x = self._13BatchNormalization(x)
        x = checkpoint(self._14Relu, x)

        x = checkpoint(self._15MaxPooling1D, x)
        x = self._16Dropout(x)

        x = checkpoint(self._17Conv1D, x)
        x = self._18BatchNormalization(x)
        x = checkpoint(self._19Relu, x)

        x = checkpoint(self._20Conv1D, x)
        x = self._21BatchNormalization(x)
        x = checkpoint(self._22Relu, x)

        x = checkpoint(self._23MaxPooling1D, x)
        x = self._24Dropout(x)

        x = checkpoint(self._25Conv1D, x)
        x = self._26BatchNormalization(x)
        x = checkpoint(self._27Relu, x)
        
        x = checkpoint(self._28Conv1D, x)
        x = self._29BatchNormalization(x)
        x = checkpoint(self._30Relu, x)

        x = checkpoint(self._31MaxPooling1D, x)
        x = self._32Dropout(x)

        x = self._33Flattern(x)
        x = self._34Dense(x)

        # sigmoid
        if self._35Sigmoid!=None:
            x = checkpoint(self._35Sigmoid, x)
            
        # Softmax
        if self._35Softmax!=None:
            x = checkpoint(self._35Softmax, x)
        
        return  x

class FlowFeatureExtractor(nn.Module):
    def __init__(self, nb_classes=8, nb_layers=2, latent_feature_length=32,use_gpu=False,device="cpu",layer_type='TAGCN'):
        assert  layer_type in ['GCN', 'GAT', 'TAGCN']
        super(FlowFeatureExtractor, self).__init__()
        self.nb_classes = nb_classes
        self.nb_layers = nb_layers
        self.layer_type = layer_type
        self.latent_feature_length = latent_feature_length
        
        self.pkt_length_fextractor = MultiLevelConv1DFingerprinting(self.latent_feature_length)
        self.arv_time_fextractor = MultiLevelConv1DFingerprinting(self.latent_feature_length)
        self.fs_fextractor = MultiLevelConv1DFingerprinting(self.latent_feature_length)
        if use_gpu:
            self.arv_time_fextractor = self.arv_time_fextractor.cuda(device)
            self.pkt_length_fextractor = self.pkt_length_fextractor.cuda(device)
            self.fs_fextractor = self.fs_fextractor.cuda(device)
        self.use_gpu = use_gpu
        self.device = device
        self.layers=[]
        head_nums = [1] + [int(1.5**(nb_layers-i)) for i in range(nb_layers)]
        for i in range(nb_layers):
            if layer_type =='GCN':
                layer = GraphConv(in_feats=self.latent_feature_length * int(1.6**i),out_feats=self.latent_feature_length * int(1.6**(i+1)), allow_zero_in_degree=True)
            elif layer_type =='GAT':
                print('Build GAT : in_feats={0}，out_feats={1},num_heads={2}'.format(self.latent_feature_length * int(1.6**i)*head_nums[i],self.latent_feature_length  * int(1.6**(i+1)),head_nums[i+1]))
                # pass
            elif layer_type == 'TAGCN':
                layer = TAGConv(in_feats=self.latent_feature_length * int(1.6**i), 
                          out_feats=self.latent_feature_length * int(1.6**(i+1)), 
                          k=1)
            if use_gpu :
                layer = layer.to(torch.device(device))
            self.layers.append(layer)
        
        self.classify = nn.Linear(in_features=(self.latent_feature_length * int(1.6 ** nb_layers)), out_features=nb_classes)
    
    def forward(self, g):
        all_info_matrix = g.ndata['all_info'].float()
        fgnet_matrix = self.fs_fextractor(all_info_matrix)

        for layer in self.layers:
            fgnet_matrix = layer(g, fgnet_matrix.to(torch.device(self.device)))
            if self.layer_type =='GAT':
                fgnet_matrix = torch.flatten(fgnet_matrix,1)
        g.ndata['fgnet'] = fgnet_matrix
        fgnet_matrix = dgl.mean_nodes(g,'fgnet')
        return  fgnet_matrix
    
    def forward_classify(self, g):
        all_info_matrix = g.ndata['all_info'].float()
        fgnet_matrix = self.fs_fextractor(all_info_matrix)
        for layer in self.layers:
            fgnet_matrix = layer(g, fgnet_matrix.to(torch.device(self.device)))
            if self.layer_type =='GAT':
                fgnet_matrix = torch.flatten(fgnet_matrix,1)
        g.ndata['fgnet'] = fgnet_matrix
        fgnet_matrix = dgl.mean_nodes(g,'fgnet')
        return  self.classify(fgnet_matrix)
        
class HeaderPatchEmbed(nn.Module):
    def __init__(self, img_size=40, patch_size=2, in_chans=1, embed_dim=96):
        super().__init__()
        img_size = (int(img_size / 5), img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) # 计算总的patch数量
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class HeaderTrafficTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super(HeaderTrafficTransformer, self).__init__(**kwargs)

        self.patch_embed = HeaderPatchEmbed(img_size=kwargs['img_size'], patch_size=kwargs['patch_size'],
                                        in_chans=kwargs['in_chans'], embed_dim=kwargs['embed_dim'])
        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        self.fc_norm = norm_layer(embed_dim)
        del self.norm

    # 包头
    def forward_packet_features(self, x, i):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        cls_pos = self.pos_embed[:, :1, :]
        packet_pos = self.pos_embed[:, i*20+1:i*20+21, :]

        pos_all = torch.cat((cls_pos, packet_pos), dim=1)

        x = x + pos_all
        x = self.pos_drop(x)

        sigh = 0
        for blk in self.blocks:
            if sigh == 0:
                sigh = 1
                x = blk(x)
            else:
                x = checkpoint(blk, x)

        cls = x[:, :1, :]

        x = x[:, 1:, :]
        x = x.reshape(B, 2, 10, -1).mean(axis=1)
        x = torch.cat((cls, x), dim=1)

        self.fc_norm(x)

        return x

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, 5, -1)
        for i in range(5):
            packet_x = x[:, :, i, :]
            packet_x = packet_x.reshape(B, C, -1, 20)
            packet_x = self.forward_packet_features(packet_x, i)
            if i == 0:
                new_x = packet_x
            else:
                new_x = torch.cat((new_x, packet_x), dim=1)
        x = new_x

        sigh = 0
        for blk in self.blocks:
            if sigh == 0:
                sigh = 1
                x = blk(x)
            else:
                x = checkpoint(blk, x)
        x = x.reshape(B, 5, 11, -1)[:, :, 0, :]
        outcome = self.fc_norm(x)
        return outcome

class PayloadPatchEmbed(nn.Module):
    def __init__(self, img_size=40, patch_size=2, in_chans=1, embed_dim=96):
        super().__init__()
        img_size = (int(img_size / 5), img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class PayloadTrafficTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super(PayloadTrafficTransformer, self).__init__(**kwargs)

        self.patch_embed = PayloadPatchEmbed(img_size=kwargs['img_size'], patch_size=kwargs['patch_size'],
                                         in_chans=kwargs['in_chans'], embed_dim=kwargs['embed_dim'])

        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        self.fc_norm = norm_layer(embed_dim)
        del self.fc_norm

    # 载荷
    def forward_packet_features(self, x, i):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        cls_pos = self.pos_embed[:, :1, :]
        packet_pos = self.pos_embed[:, i*80+1:i*80+81, :]
        pos_all = torch.cat((cls_pos, packet_pos), dim=1)
        
        x = x + pos_all
        x = self.pos_drop(x)

        sigh = 0
        for blk in self.blocks:
            if sigh == 0:
                sigh = 1
                x = blk(x)
            else:
                x = checkpoint(blk, x)

        cls = x[:, :1, :]

        x = x[:, 1:, :]
        x = x.reshape(B, 4, 20, -1).mean(axis=1)
        x = torch.cat((cls, x), dim=1)

        self.fc_norm(x)

        return x

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, 5, -1)
        for i in range(5):
            packet_x = x[:, :, i, :]
            packet_x = packet_x.reshape(B, C, -1, 40) 
            packet_x = self.forward_packet_features(packet_x, i)

            if i == 0:
                new_x = packet_x
            else:
                new_x = torch.cat((new_x, packet_x), dim=1)
        x = new_x

        sigh = 0
        for blk in self.blocks:
            if sigh == 0:
                sigh = 1
                x = blk(x)
            else:
                x = checkpoint(blk, x)
        x = x.reshape(B, 5, 21, -1)[:, :, 0, :]

        outcome = self.fc_norm(x)
        del new_x,x
        return outcome
        
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, input_dim * 2)

        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim * 2, input_dim)

        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(input_dim, int(input_dim / 2))

        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(int(input_dim / 2), num_classes + 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = checkpoint(self.fc2, x)
        x = checkpoint(self.relu2, x)
        x = checkpoint(self.fc3, x)
        x = checkpoint(self.relu3, x)
        x = checkpoint(self.fc4, x)
        return x

class ODCModel(nn.Module):
    
    def __init__(self, input_dim=256, hidden_dim1=512, hidden_dim2=512, output_dim=2):
        super(ODCModel, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_dim1, kernel_size=3, stride=1, padding=1),  # (64, 256) -> (64, hidden_dim1)
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim1, out_channels=hidden_dim2, kernel_size=3, stride=1, padding=1),  # (64, hidden_dim1) -> (64, hidden_dim2)
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim2, out_channels=hidden_dim1, kernel_size=3, stride=1, padding=1),  # (64, hidden_dim2) -> (64, hidden_dim1)
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim1, out_channels=input_dim, kernel_size=3, stride=1, padding=1),  # (64, hidden_dim1) -> (64, input_dim)
            nn.ReLU()
        )

        self.cls_head = nn.Linear(input_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        
        self.memory_bank = None
        self.momentum = 0.5
        self.num_classes = output_dim
        
        self.loss_weight = torch.ones((self.num_classes, ),
                                      dtype=torch.float32).cuda()
        self.loss_weight /= self.loss_weight.sum()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
  
    def forward(self, x):
        x = self.extract_feat(x)
        cls_score = self.cls_head(x)
        
        return [cls_score]
        
    def loss(self, features, centroids, cls_score, pseudo_labels):
        losses = dict()
        loss_cen = self.center_loss(features, centroids, pseudo_labels)
        loss_con = self.contrastive_loss(features, pseudo_labels)
        loss_cro = self.criterion(cls_score[0], pseudo_labels)
        losses['loss'] = loss_cro
        losses['acc'] = self._accuracy(cls_score[0], pseudo_labels)

        return losses
        
    def center_loss(self, features, centroids, labels):
        features = features.view(features.size(0), -1)
        target_centers = centroids[labels]
        loss = (features - target_centers).pow(2).sum(1).mean()
        return loss

    def contrastive_loss(self, features, labels, margin=1.0):
        distances = (features.unsqueeze(1) - features.unsqueeze(0)).pow(2).sum(2)
        labels_eq = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        positive_loss = distances * labels_eq.float()
        negative_loss = (margin - distances).clamp(min=0) * (1 - labels_eq.float())
        loss = positive_loss.sum() + negative_loss.sum()
        return loss / (len(labels) * len(labels))

    def extract_feat(self, inputs):
        # 直接使用输入特征，不需要调整形状
        x = inputs.unsqueeze(1)
        x = self.backbone(x)
        x = x.mean(dim=2)
        return x
    
    def _accuracy(self, pred, target, topk=1):
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
            return_single = True
        else:
            return_single = False

        maxk = max(topk)
        _, pred_label = pred.topk(maxk, dim=1)
        pred_label = pred_label.t()
        correct = pred_label.eq(target.contiguous().view(1,-1).expand_as(pred_label))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(
                0, keepdim=True)
            res.append(correct_k.mul_(100.0 / pred.size(0)))
        return res[0] if return_single else res
    
    def parse_losses(self, losses):
        total_loss = 0.0
        total_loss = losses['loss'].mean() 

        return total_loss

    def initialize_memory_bank(self, data, num_clusters):
        self.memory_bank = {
            'features': torch.zeros((len(data), 160), dtype=torch.float32),  
            'labels': torch.zeros((len(data), ), dtype=torch.long),
            'centroids': torch.zeros((num_clusters, 160), dtype=torch.float32).cuda()
        }
        
        kmeans = KMeans(n_clusters = num_clusters, num_init = num_clusters,)
                
        data = data.unsqueeze(0)
        kmeans_result = kmeans(data)
        data = data.squeeze(0)
        labels = kmeans_result.labels.squeeze(0)
        self.memory_bank['features'].copy_(data.float())
        self.memory_bank['labels'].copy_(labels.long())
        self.set_reweight(labels, 0.5)
        kmeans = KMeans(n_clusters=num_clusters, n_init=num_clusters, tol=1e-4)
        kmeans.fit(data)  # 直接对256维特征进行聚类
        assert isinstance(kmeans.labels_, np.ndarray)
        labels = kmeans.labels_.astype(np.int64)
        labels_tensor = torch.from_numpy(labels).long()
        data_tensor = torch.from_numpy(data)

        # 将特征和标签存入记忆库中
        self.memory_bank['features'].copy_(data_tensor)
        self.memory_bank['labels'].copy_(labels_tensor)

        # 更新损失权重
        self.set_reweight(labels_tensor, 0.5)
        
        # 将聚类中心存入记忆库中
        centroids = self._compute_centroids()
        self.memory_bank['centroids'].copy_(centroids)

        # print(self.memory_bank['centroids'])
        unique_labels, counts = torch.unique(labels, return_counts=True)
        print("Initial KMeans Clustering Result:")
        for label, count in zip(unique_labels, counts):
            print(f"Cluster {label}: {count} samples")
        return labels
        
    def update_memory_bank_samples(self, ind, feature):
        with torch.no_grad():
            feature_tensor = feature.clone().detach().to(dtype=torch.float32)
            feature_norm = feature_tensor / (feature_tensor.norm(dim=1, keepdim=True) + 1e-10)

            old_features = self.memory_bank['features'][ind].clone().detach().to(dtype=torch.float32)
            feature_norm = feature_norm.to(old_features.device)
            feature_new = (1 - self.momentum) * old_features + self.momentum * feature_norm
            feature_norm = feature_new / (feature_new.norm(dim=1, keepdim=True) + 1e-10)
            self.memory_bank['features'][ind] = feature_norm.to(self.memory_bank['features'].device)
            
            change_ratio = self._reassign_pseudo_labels()

            return change_ratio

    def _reassign_pseudo_labels(self):
        # 重新分配伪标签
        feature_norm = self.memory_bank['features'].clone().detach().to(dtype=torch.float32).to(next(self.parameters()).device)
        centroids = self.memory_bank['centroids'].clone().detach().to(dtype=torch.float32).to(next(self.parameters()).device)
        similarity_to_centroids = torch.mm(centroids, feature_norm.t())
        new_labels = similarity_to_centroids.argmax(dim=0)

        new_labels = new_labels.to(self.memory_bank['labels'].device)
        change_ratio = (new_labels != self.memory_bank['labels']).sum() / float(new_labels.shape[0])
        self.change_ratio = change_ratio
        self.memory_bank['labels'] = new_labels

        return change_ratio

    def _compute_centroids(self):
        label_bank_np = self.memory_bank['labels'].numpy()
        argl = np.argsort(label_bank_np)
        sortl = label_bank_np[argl]
        diff_pos = np.where(sortl[1:] - sortl[:-1] != 0)[0] + 1
        start = np.insert(diff_pos, 0, 0)
        end = np.insert(diff_pos, len(diff_pos), len(label_bank_np))
        class_start = sortl[start]

        centroids = self.memory_bank['centroids']
        for i, st, ed in zip(class_start, start, end):
            centroids[i, :] = self.memory_bank['features'][argl[st:ed], :].mean(dim=0)
        return centroids

    def _compute_centroids_ind(self, cinds):
        num = len(cinds)
        centroids = torch.zeros((num, 160), dtype=torch.float32)
        for i, c in enumerate(cinds):
            ind = np.where(self.memory_bank['labels'].numpy() == c)[0]
            centroids[i, :] = self.memory_bank['features'][ind, :].mean(dim=0)
        return centroids

    def update_centroids(self, cinds=None):
        if cinds is None:
            center = self._compute_centroids()
            self.memory_bank['centroids'].copy_(center)
        else:
            center = self._compute_centroids_ind(cinds)
            self.memory_bank['centroids'][cinds, :] = center.to(self.memory_bank['centroids'].device)

    def deal_with_small_clusters(self):
        histogram = np.bincount(self.memory_bank['labels'], minlength=self.num_classes)
        small_clusters = np.where(histogram <20)[0].tolist()  # 找出样本数小于50的小聚类, hwx
        if len(small_clusters) == 0:
            return
        for s in small_clusters:
            idx = np.where(self.memory_bank['labels'] == s)[0]
            if len(idx) > 0:
                inclusion = torch.from_numpy(np.setdiff1d(np.arange(self.num_classes), np.array(small_clusters), assume_unique=True)).cuda()

                target_idx = torch.mm(self.memory_bank['centroids'][inclusion, :], self.memory_bank['features'][idx, :].cuda().permute(1, 0)).argmax(dim=0)
                target = inclusion[target_idx]

                self.memory_bank['labels'][idx] = target.clone().detach().to(dtype=torch.long).to(self.memory_bank['labels'].device)

        self._redirect_empty_clusters(small_clusters)

    def _redirect_empty_clusters(self, empty_clusters):

        for e in empty_clusters:
            max_cluster = np.bincount(self.memory_bank['labels'], minlength=self.num_classes).argmax().item()  

            sub_cluster1_idx, sub_cluster2_idx = self._partition_max_cluster(max_cluster)  
            
            self.memory_bank['labels'][sub_cluster2_idx] = e  

            self.update_centroids([max_cluster, e])

    def _partition_max_cluster(self, max_cluster):
        max_cluster_idx = np.where(self.memory_bank['labels'] == max_cluster)[0]
        if len(max_cluster_idx) >= 2:
            max_cluster_features = self.memory_bank['features'][max_cluster_idx, :]

            kmeans = KMeans(n_clusters=2) 
            result = kmeans(max_cluster_features.unsqueeze(0))
            label = result.labels.squeeze(0)
            sub_cluster1_idx = max_cluster_idx[label == 0]
            sub_cluster2_idx = max_cluster_idx[label == 1]
            return sub_cluster1_idx, sub_cluster2_idx
        else:
            return max_cluster_idx, torch.tensor([])

    def set_reweight(self, labels=None, reweight_pow=0.5):
        if labels is None:
            if isinstance(self.memory_bank['labels'], torch.Tensor) and self.memory_bank['labels'].is_cuda:
                labels = self.memory_bank['labels'].cpu().numpy()
            else:
                labels = self.memory_bank['labels']
        histogram = np.bincount(labels.cpu().numpy(), minlength=self.num_classes).astype(np.float32)
        inv_histogram = (1. / (histogram + 1e-10))**reweight_pow
        weight = inv_histogram / inv_histogram.sum()

        weight_tensor = torch.from_numpy(weight).to(self.loss_weight.device)
        self.loss_weight.copy_(weight_tensor)
        self.criterion = nn.CrossEntropyLoss(weight=self.loss_weight).to(next(self.parameters()).device)

# FEC-OSL
class FECOSLModel(nn.Module):
    def __init__(self, nb_classes, use_gpu=False, device = "cpu", ratio=0.75, new_classes=1):
        super(FECOSLModel, self).__init__()
        self.byte_header_extractor = HeaderTrafficTransformer(
            img_size=20,  # header 矩阵
            patch_size=2, in_chans=1, embed_dim=96, depth=4, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.byte_payload_extractor = PayloadTrafficTransformer(
            img_size=40,  # payload 矩阵
            patch_size=2, in_chans=1, embed_dim=96, depth=4, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        self.flow_extractor = FlowFeatureExtractor(
            nb_classes = nb_classes, nb_layers=2, latent_feature_length=32, use_gpu=use_gpu, device=device,layer_type='TAGCN'
        )

        self.energy_classifier = MLP(256, int(nb_classes * ratio))
        self.odc_classifier = ODCModel(256, 512, 512, nb_classes - int(nb_classes * ratio) - new_classes)
        self.is_init_odc = False

    def forward(self, data_loader_head, data_loader_payload, data_loader_flow, flag):
        if(flag == 'energy'):
            x1 = self.byte_header_extractor.forward_features(data_loader_head).mean(dim=1)
            x2 = self.byte_payload_extractor.forward_features(data_loader_payload).mean(dim=1)  # 改5
            x3 = self.flow_extractor(data_loader_flow)
            x4 = torch.cat((x1, x2, x3), dim=1)
           
            logits = self.energy_classifier(x4)
            del x1, x2, x3 
            return x4, logits
        
        elif(flag == 'odc'):
            x1 = self.byte_header_extractor.forward_features(data_loader_head).mean(dim=1)
            x2 = self.byte_payload_extractor.forward_features(data_loader_payload).mean(dim=1)
            x3 = self.flow_extractor(data_loader_flow)
            x4 = torch.cat((x1, x2, x3), dim=1)   # 融合特征
            with torch.no_grad():
                logits = self.energy_classifier(x4)

            cls_score = self.odc_classifier(x4)
            del x1, x2, x3
            return x4, cls_score
        

