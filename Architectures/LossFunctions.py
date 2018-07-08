import torch
from torch.autograd import Variable
import torch.nn.functional as F
import math, pdb
import numpy as np
from Loader import transforms

def weighted_mse_loss(pred,target,weights):
    sqerr = (pred-target)**2
    sqerr = sqerr * weights
    loss = torch.mean(sqerr, dim=0)
    return loss

def combinedLossFunction(output, data, term_weights):
    # classification loss: cross entropy
    loss_class = term_weights['classification'] * F.cross_entropy(output['classification'], Variable(data['classID'].cuda()))
    # regression loss: mse
    reg_energy, target_energy = transforms.reg_target_energy_for_loss(output, data)
    loss_energy = term_weights['energy_regression'] * F.mse_loss(reg_energy, target_energy)
    loss_eta = term_weights['eta_regression'] * F.mse_loss(output['eta_regression'], transforms.target_eta_for_loss(data))
    loss_phi = term_weights['phi_regression'] * F.mse_loss(output['phi_regression'], transforms.target_phi_for_loss(data))
    return {"total": loss_class+loss_energy+loss_eta+loss_phi, "classification": loss_class, "energy": loss_energy, "eta": loss_eta, "phi": loss_phi}

def classificationOnlyLossFunction(output, data, term_weights):
    # classification loss: cross entropy
    loss_class = term_weights['classification'] * F.cross_entropy(output['classification'], Variable(data['classID'].cuda()))
    zero = Variable(torch.from_numpy(np.array([0])))
    return {"total": loss_class, "classification": loss_class, "energy": zero, "eta": zero, "phi": zero}
