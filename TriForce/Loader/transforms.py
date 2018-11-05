import torch
from torch.autograd import Variable
import pdb

# returns reg_energy, target_energy for MSE loss function (on gpu)
def reg_target_energy_for_loss(output, data):
    truth_energy = Variable(data['energy'].cuda())
    reco_energy = Variable((data['ECAL_E'] + data['HCAL_E']).cuda())
    denom = torch.log(truth_energy)
    reg_energy = output['energy_regression'] / denom
    target_energy = (truth_energy - reco_energy) / (10.0 * denom)
    return reg_energy, target_energy

# returns target_eta for MSE loss function (on gpu)
def target_eta_for_loss(data):
    return Variable((data['eta'] - data['recoEta']).cuda())*10.0

# returns target_phi for MSE loss function (on gpu)
def target_phi_for_loss(data):
    return Variable((data['phi'] - data['recoPhi']).cuda())*10.0

# returns predicted energy from regression (assumes reg_energy is a Tensor on cpu)
def pred_energy_from_reg(reg_energy, data):
    return (reg_energy * 10.0) + data['ECAL_E'] + data['HCAL_E']

# returns predicted eta from regression (assumes reg_eta is a Tensor on cpu)
def pred_eta_from_reg(reg_eta, data):
    return (reg_eta / 10.0) + data['recoEta']

# returns predicted phi from regression (assumes reg_phi is a Tensor on cpu)
def pred_phi_from_reg(reg_phi, data):
    return (reg_phi / 10.0) + data['recoPhi']

