import torch.nn.functional as F
from torch import nn
import dgl.nn as dglnn
import config as config


class GraphConvModel(nn.Module):

    def __init__(self,in_feats, hidden_dim, n_layers,fanouts, n_classes,h_dropout,dnn_unit_dropout,save_path=config.gcn_model_path,
                 loss_path=config.gcn_loss_path,acc_path=config.gcn_acc_path,confusion_path=config.gcn_confusion_path,norm='both',activation=F.relu):
        '''
        :param in_feats: input feature dimension
        :param hidden_dim: hidden layer dimension
        :param n_layers:Sampling layers
        :param fanouts the maximum number of sampled neighbor nodes per layer
        :param n_classes:number of categories
        :param h_dropout: hidden layer dropout rate
        :param dnn_unit_dropout:DNN dropout rate
        :param save_path:model storage path
        :param loss_path: training loss save path
        :param acc_path： training accuracy save path 
        :param confusion_path: confusion matrix save path
        :param norm: normalization parameter
        :param activation:activation function default=relu
        '''
        super(GraphConvModel, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.fanouts=fanouts
        self.n_classes = n_classes
        self.h_dropout=h_dropout
        self.dnn_unit_dropout=dnn_unit_dropout
        self.save_path=save_path
        self.loss_path = loss_path
        self.acc_path = acc_path
        self.confusion_path = confusion_path
        self.norm=norm
        self.activation = activation
        self.dropout = nn.Dropout(h_dropout)

        self.linear_list=[]
        for i in range(config.DNN_layers):
            if i==0:
                self.linear_list.append(nn.Linear(self.hidden_dim,config.DNN_hidden_dim[i]))
            else:
                self.linear_list.append(nn.Linear(config.DNN_hidden_dim[i-1], config.DNN_hidden_dim[i]))
            self.linear_list.append(nn.LeakyReLU ())
            self.linear_list.append(nn.Dropout (self.dnn_unit_dropout))

        self.DNN = nn.Sequential (*self.linear_list)

        self.layers = nn.ModuleList()

        # build multiple layers
        self.layers.append(dglnn.GraphConv(in_feats=self.in_feats,
                                           out_feats=self.hidden_dim,
                                           norm=self.norm,
                                           activation=self.activation))
        for l in range(1, (self.n_layers)):
            self.layers.append(dglnn.GraphConv(in_feats=self.hidden_dim,
                                               out_feats=self.hidden_dim,
                                               norm=self.norm,
                                               activation=self.activation))

    def forward(self, blocks, features):
        h = features
        for l, (layer, block) in enumerate (zip (self.layers, blocks)):
            h = layer (block, h)
            h = self.dropout (h)
        return self.DNN (h)
