import torch

random_seed=22
#model storage path
dnn_model_path="base_dnn.pkl"
lstm_model_path="base_lstm.pkl"

#Text classification model data file (labeled points)
feature_path = '../data/train_nodes_have_label_feature.npy'
label_path = '../data/train_nodes_have_label_l.npy'

#DNN model parameters
DNN_layers=5         #fully connected layers
DNN_hidden_dim =[512,256,128,64,23]  #Dimensions of the hidden layer of the fully connected layer
DNN_dropout =0.1     #droupot rate
DNN_model_path="./results/models/dnn_"+str(DNN_layers)+".pkl"  #model storage path
DNN_loss_path='./results/dnn_'+str(DNN_layers)+"_loss.csv"      #training loss save path
DNN_acc_path='./results/dnn_'+str(DNN_layers)+"_acc.csv"    #training accuracy save path

#LSTM model parameters
lstm_hidden_size=128
lstm_layers=2
lstm_dropout=0.1
bidirectional=False
lstm_model_path="./results/models/lstm_"+str(lstm_layers)+".pkl"  #model storage path
lstm_loss_path='./results/lstm_'+str(lstm_layers)+"_loss.csv"      #training loss save path
lstm_acc_path='./results/lstm_'+str(lstm_layers)+"_acc.csv"    #training accuracy save path




#训练参数
learning_rate=0.001    #learning rate
weight_decay=0     #L2 regularization parameter
epochs=100           #maximum training epochs
early_stop_cnt=20   #early stopping patience
batch_size=2048       #batch size
device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   #Use GPU or CPU for training
num_worker=16

