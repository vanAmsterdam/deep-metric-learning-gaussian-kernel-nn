import torch 
from torch import optim
from torch.autograd import Variable
from torch.distributions import Normal
import numpy as np
from simulatedata import simulate_data

from model.net import Net, get_class_probs, loss_fn

import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--foo", default="bar")

def train(data, model, optimizer, storage, args, params, epoch):

    print("training epoch {}".format(epoch), end="")

    x = Variable(data["x"])
    l = Variable(data["target"])

    out = model(x)
    out_z = out["z"]

    if params.use_weights:
        out_w = out["w"]
    else:
        out_w = torch.ones_like(l)

    if (epoch == 1) or (epoch % params.c_update_interval == 0):
        print(" updating centres".format(epoch), end = "")
        c = Variable(out["z"])
        storage["c"] = c
    else:
        # TODO find a way to get this c into global scope between training iterations
        c = storage["c"]

    model.train()
    loss = torch.zeros((1,))
    
    for i in torch.arange(x.shape[0]):
        include = np.delete(np.arange(x.shape[0]), i)
        p = get_class_probs(out_z[i,:], c[include,:], l[include], out_w[include], params)
        loss += loss_fn(p, l[i], params)

    print(", loss: {}".format(loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



if __name__ == "__main__":
    # Load the parameters from json file
    args = parser.parse_args()

    params = utils.Params("params.json")

    model = Net(params)
    optimizer = optim.SGD(model.parameters(), params.lr)

    x, t = simulate_data(params)
    data = {"x": x, "target": t}
    storage = {}

    for epoch in range(params.epochs):
        train(data, model, optimizer, storage, args, params, epoch+1)
