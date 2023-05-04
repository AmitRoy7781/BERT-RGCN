# Investigating Knowlege Graph  Completion with Pre-trained Language model and Graph Neural Networks

A repository containing the implementation of Bert embeddings with RGCN for the link prediction task in knowledge graph

<img src="bert_rgcn.png" width="600">

**Abstract** 

In recent years, two lines of research in deep learning are very gaining interest namely large language models and Graph Neural Networks. The earlier one has shown reasonable performance in different types of NLP tasks and capturing the structural relationship with the graph convolution operation is the key reason behind the success of GNNs. Predicting links in knowledge graphs is a very important task for knowledge completion and information retrieval. Considering the knowledge graph have text data in their entities and relations and also provides the structural information in the form of a triplet, several state-of-the-art approaches have separately employed BERT-like large language models and GNNs to predict links knowledge graph. In this study, we aim at investigating how the combination of embeddings from the large language models followed by a graph convolution network can solve the link prediction task. We performed experiments on three real-world knowledge graph datasets and demonstrated our results.


**Main Parameters:**

```
--test        Boolean flag to train the BERT-RGCN or load pretrained model for inference
--dataset     Name of the dataset
--emb_type    type of embedding used bert or scratch
--long_text   Use long text for the entities and relations or short text
--max_relations   Maximum number of relations to be used
--max_tokens      Maximum number of tokens to be considered for the embedding
--graph-batch-size Batch size of the graph during training
--graph-split-size Ratio for spliting the links in graph 
--negative-sample Ratio of sampling negative edges
--n-epochs  Number of training epochs
--evaluate-every Evaluate and save the model after how many epochs
--dropout       Dropout probability
--gpu           Cuda Device ID
--n-bases       Number of bases used for basis-decomposition
--hid-dim       Number of dimension of the hidden representation
--regularization  Hyperparameter for the weight decay
--grad-norm       Clip the gradient within which range
```


**Basic Usage:**

Run the python notebook with appropriate parameter changes.

