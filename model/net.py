import torch
import torch.nn as nn
import numpy as np

class GKNNNet(nn.Module):
    def __init__(self, params):
        super(GKNNNet, self).__init__()
        print("initializing Gaussian Kernel Nearest Neighbour Net with params: ")
        for key, param in params.__dict__.items():
            print("{}: {}".format(key, param))
        self.cnn = CNN(params)

    def forward(self, x):
        x = self.cnn(x)
        return x

class CNN(nn.Module):
    def __init__(self, params):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(params.x_dim, params.embedding_dimension)

    def forward(self, x):
        x = self.fc1(x)
        return x
    
def get_class_probabilities(x, c, l, w, params):
    '''
    Return class probabilities for a single training example, based on the embedding vector and a vector of gaussian kernel 
    centres c, one for each training sample, except the current sample
    labels: vector with labels associated with class centres c
    w: weights of each sample

    Dim(c) = (m - 1, embedding_dimension)
    Dim(l) = (m - 1,)
    Dim(w) = (m - 1,)
    Dim(x) = (embedding_dimension,)
    Dim(x) = (batch_size, embedding_dimension) -> not yet implemented
    '''
    # assert x.shape == (params.batch_size, params.embedding_dimension)
    assert x.shape == (params.embedding_dimension,)
    assert c.shape[0] == l.shape[0]

    ## add weights and sum
    denominator = torch.exp(-(x - c) / (2 * params.global_scale ** 2))

    for class_i in range(params.num_classes):
        c_match = c[l==class_i,]







    

