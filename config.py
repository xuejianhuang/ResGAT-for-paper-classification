import torch

#Graph data store file
base_path="./data"
graph_file="graph_pre.bin"
label_file='labels_pre.pkl'
features_file="features_pre.npy"

#Checkpoint save settings
resume=False  #Whether to start training from a checkpoint
checkpoint_path='./checkpoint/ckpt.pth'  #Checkpoint save path
checkpoint_epoch=5          #Save the current training state every how many epochs



#visdom configuration
server='localhost'
port=8097



#GAT model parameters
GAT_layers=3                #number of sampling layers
GAT_fanouts=[10,10,10]       #the maximum number of sampled neighbor nodes per layer
GAT_hidden_dim=256      #hidden layer dimension
GAT_f_dropout=0.1      #feature dropout rate
GAT_atte_dropout=0.1     #attention dropout rate
GAT_atte_num_heads=1      #the number of multi-headed attention
gat_model_path="./results/models/gat_"+str(GAT_layers)+"_"+str(GAT_atte_num_heads)+".pkl"  #model storage path
gat_loss_path='./results/gat_'+str(GAT_layers)+"_"+str(GAT_atte_num_heads)+"_loss.csv"      #training loss save path
gat_acc_path='./results/gat_'+str(GAT_layers)+"_"+str(GAT_atte_num_heads)+"_acc.csv"    #training accuracy save path
gat_confusion_path='./results/gat_'+str(GAT_layers)+"_"+str(GAT_atte_num_heads)+'_confusin.csv'  #confusion matrix save path

#GCN model parameters
GCN_layers=3                #Number of sampling layers
GCN_fanouts=[10,10,10]       #The maximum number of sampled neighbor nodes per layer
GCN_hidden_dim=256       #hidden layer dimension
GCN_h_dropout=0.1       #Hidden layer dropout rate
gcn_model_path="./results/models/gcn_"+str(GCN_layers)+".pkl"  #model storage path
gcn_loss_path='./results/gcn_'+str(GCN_layers)+"_loss.csv"      #training loss save path
gcn_acc_path='./results/gcn_'+str(GCN_layers)+"_acc.csv"    #training accuracy save path
gcn_confusion_path='./results/gcn_'+str(GCN_layers)+'_confusin.csv'  #confusion matrix save path


#SAGE model parameters
SAGE_layers=3                #Number of sampling layers
SAGE_fanouts=[10,10,10]       #The maximum number of sampled neighbor nodes per layer
SAGE_hidden_dim=256      #hidden layer dimension
SAGE_f_dropout=0.1        #feature dropout rate
sage_model_path="./results/models/sage_"+str(SAGE_layers)+".pkl"  #model storage path
sage_loss_path='./results/sage_'+str(SAGE_layers)+"_loss.csv"      #training loss save path
sage_acc_path='./results/sage_'+str(SAGE_layers)+"_acc.csv"    #training accuracy save path
sage_confusion_path='./results/sage_'+str(SAGE_layers)+'_confusin.csv'  #confusion matrix save path




n_classes=23       #number of categories
in_feats=300       #input feature dimension

label=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q', 'R', 'S', 'T', 'U', 'V', 'W']


#fully connected layer parameters
DNN_layers=5         #fully connected layers
DNN_hidden_dim =[512,256,128,64,n_classes]  #hidden layer dimension
DNN_dropout =0.1     #dropout rate



#training parameters
learning_rate=0.001    #learning rate
weight_decay=0     #L2 regularization parameter
epochs=100            #maximum training epochs
early_stop_cnt=20      #early stopping patience
batch_size=2048       #batch size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   #Use GPU or CPU for training
num_worker=16
