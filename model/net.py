import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self, params):
        print("initializing Net with params: ")
        for key, value in params.dict.items():
            print("{}: {}".format(key,value))
        super(Net, self).__init__()
        self.params = params
        self.fc1 = nn.Linear(params.x_dim, params.embedding_dimension)
        self.fc2 = nn.Linear(params.embedding_dimension, params.embedding_dimension)
        self.fc3 = nn.Linear(params.embedding_dimension, params.embedding_dimension)
        self.fcw = nn.Linear(params.embedding_dimension, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        z = F.relu(self.fc3(x))

        out = {"z": z}

        if self.params.use_weights:
            w = self.fcw(x)
            out["w"] = w

        return out
    
def get_class_probs(x, c, l, w, params):
    '''
    TODO: vectorize over l
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
    # print(x.shape)
    # print(c.shape)
    # print(l.shape)
    # print(w.shape)
    assert x.shape == (params.embedding_dimension,)
    assert c.shape[0] == l.shape[0]

    if w is None:
        w = torch.ones_like(c)

    p = torch.zeros((params.num_classes,))

    denominator = torch.sum(w * (torch.exp(-((x - c)**2) / (2 * params.global_scale ** 2))))

    for class_i in range(params.num_classes):
        c_match = c[l==class_i,:]
        w_match = w[l==class_i,:]
        enumerator = torch.sum(w_match * (torch.exp(-((x - c_match)**2) / (2 * params.global_scale ** 2))))
        p[class_i] = enumerator / denominator

    p += 1e-6 # for numeric stability

    return p


def loss_fn(p, t, params):
    """
    Loss of predictions
    dim(p): (num_classes,)
    t: integer, index of p
    """
    assert p.shape == (params.num_classes,)
    return -torch.log(p[t])








    

