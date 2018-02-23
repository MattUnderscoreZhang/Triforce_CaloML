import torch
import pdb

# This module has to be here for dependency reasons

# train model
def train(model, ECALs, HCALs, truth):
    model.net.train()
    model.optimizer.zero_grad()
    # pdb.set_trace()
    outputs = model.net(ECALs)# , HCALs)
    loss = model.lossFunction(outputs, truth)
    loss.backward()
    model.optimizer.step()
    _, predicted = torch.max(outputs.data, 1)
    try:
        accuracy = (predicted == truth.data).sum()/truth.shape[0]
    except:
        accuracy = 0 # ignore accuracy for energy regression
    return (loss.data[0], accuracy)

# evaluate model
def eval(model, ECALs, HCALs, truth):
    model.net.eval()
    outputs = model.net(ECALs, HCALs)
    loss = model.lossFunction(outputs, truth)
    _, predicted = torch.max(outputs.data, 1)
    try:
        accuracy = (predicted == truth.data).sum()/truth.shape[0]
    except:
        accuracy = 0 # ignore accuracy for energy regression
    return (loss.data[0], accuracy)
