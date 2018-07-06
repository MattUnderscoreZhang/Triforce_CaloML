import torch
from torch.autograd import Variable
import torch.nn.functional as F
import math, pdb
import numpy as np

def weighted_mse_loss(pred,target,weights):
    sqerr = (pred-target)**2
    sqerr = sqerr * weights
    loss = torch.mean(sqerr, dim=0)
    return loss

def combinedLossFunction(output, data, term_weights):
    # classification loss: cross entropy
    loss_class = term_weights['classification'] * F.cross_entropy(output['classification'], Variable(data['classID'].cuda()))
    # regression loss: mse
    truth_energy = Variable(data['energy'].cuda())
    # use per-event weights for energy to emphasize lower energies
    event_weights = 1.0 / torch.log(truth_energy)
    loss_energy = term_weights['energy_regression'] * weighted_mse_loss(output['energy_regression'], truth_energy, event_weights)
    loss_eta = term_weights['eta_regression'] * F.mse_loss(output['eta_regression'], Variable(data['eta'].cuda()))
    return {"total": loss_class+loss_energy+loss_eta, "classification": loss_class, "energy": loss_energy, "eta": loss_eta}

def classificationOnlyLossFunction(output, data, term_weights):
    # classification loss: cross entropy
    loss_class = term_weights['classification'] * F.cross_entropy(output['classification'], Variable(data['classID'].cuda()))
    zero = Variable(torch.from_numpy(np.array([0])))
    return {"total": loss_class, "classification": loss_class, "energy": zero, "eta": zero}
