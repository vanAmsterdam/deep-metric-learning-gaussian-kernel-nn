import torch 
from torch import optim
import numpy as np

from model.net import GKNNNet

import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--foo", default="bar")


if __name__ == "__main__":
    # Load the parameters from json file
    args = parser.parse_args()

    params = utils.Params("params.json")

    model = GKNNNet(params)
    optimizer = optim.SGD(model.parameters(), params.lr)

    x = torch.zeros((10, params.x_dim))

    print(model(x))
