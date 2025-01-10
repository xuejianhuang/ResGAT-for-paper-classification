# Paper Automatic Classification Based on Graph Neural Network
  With the development of natural language processing techniques, scholars have proposed various academic paper classification methods based on text analysis. In early research, the focus was mainly on feature engineering, where representation features of papers were extracted from their content, titles, abstracts, keywords, etc. These features were then used to build classification models using traditional machine learning algorithms such as Naive Bayes (NB), k-Nearest Neighbors (KNN), Support Vector Machines (SVM), and others. In recent years, with the advancement of deep learning techniques, text classification based on models like Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), Transformer, etc., has become a hot research topic. Although significant progress has been made in academic paper classification based on text analysis, this approach only considers the intrinsic features of papers such as titles, abstracts, and keywords, without incorporating the features of reference papers and citations. As a result, the accuracy of classification is not satisfactory. Therefore, in this experiment, we construct a paper relationship graph based on citation relationships and propose an automatic academic paper classification model using Graph Neural Networks (GNN) to improve the accuracy of automatic paper classification.
  * Programming language: python, pytorch, sklearn
# Dataset Description
  The data used in this experiment is a paper relationship graph generated based on Microsoft Academic Literature. In this graph, the nodes represent individual papers, and the edges represent citation relationships between the papers. The original dataset consists of 3,063,061 papers and 29,168,650 edges, covering a total of 23 categories.
You can download the dataset from the following link: https://pan.baidu.com/s/1G7CtmsZ-thsMW2zjVhH2IQ
Extraction code: isc3
# Experiment model
### Traditional graph neural network model
   * GCN
   * GraphSAGE
   * GAT 
### Traditional Text Classification Model
   * Naive Bayes
   * KNN
   * SVM
   * LSTM
   * Transformer
# File directory description
  * baseline: Catalog of traditional text classification models
      - results：Baseline model experiment results directory
      - baseline_config.py：baseline configuration file
      - DataSampler.py：Data Sampling Class
      - DNN.py：Naive Bayesian model
      - KNN.py: KNN model
      - SVM.py：SVM model
      - LSTM.py：LSTM model
      - Transformer.py:Transformer model
  * checkpoint: Checkpoint save directory, in order to realize the recovery of interruption during training
  * pre_data: Data preprocessing and statistical analysis code
  * results: GNN model experiment results save directory
  * confusin.py: Visualize the confusion matrix
  * ResGAT.py: GAT and its improved model implementation class, the number of multi-head attention and residual structure can be configured through parameters
  * GCN.py: GCN model implementation class
  * main.py: main file
  * SAGE.py：GraphSAGE model implementation class
  * utils.py: Toolkit class
  * visualize.py: visdom dynamic result visualization class
 # Main dependent packages
   * numpy=1.21.4
   * pandas=1.2.0
   * matplotlib=3.3.2
   * sklearn=0.24.1
   * dgl=0.7.2
   * pytorch=1.9.1
   * visdom=0.1.8.9
# Experimental results
  The improved GAT model based on multi-head attention mechanism and residual connection has the highest accuracy rate, reaching 61%.

# Citation
If you find this project helps your research, please kindly consider citing our paper in your publications.
```
@article{huang2024resgat,
  title={ResGAT: an improved graph neural network based on multi-head attention mechanism and residual network for paper classification},
  author={Huang, Xuejian and Wu, Zhibin and Wang, Gensheng and Li, Zhipeng and Luo, Yuansheng and Wu, Xiaofang},
  journal={Scientometrics},
  volume={129},
  number={2},
  pages={1015--1036},
  year={2024},
  publisher={Springer}
}
```
