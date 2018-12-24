import torch
import numpy as np


def simulate_data(params):
    t = torch.randint(params.num_classes, (params.num_samples,1))
    # x_noise = params.x_noise * torch.rand((params.num_samples, params.num_classes))
    x_noise = torch.normal(torch.zeros((params.num_samples, params.num_classes)),
                           torch.ones((params.num_samples, params.num_classes)))
    x = torch.zeros_like(x_noise)
    x.scatter_(1, t, 1)
    x += params.x_noise * x_noise

    return x, t.squeeze()

