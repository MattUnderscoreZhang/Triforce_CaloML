import torch

# This module has to be here for dependency reasons

# train model
def train(model, ECALs, HCALs, truth):
    model.net.train()
    model.optimizer.zero_grad()
    outputs = model.net(ECALs)
    loss = model.lossFunction(outputs, truth)
    loss.backward()
    model.optimizer.step()
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == truth.data).sum()/truth.shape[0]
    return (loss.data[0], accuracy)

# evaluate model
def eval(model, ECALs, HCALs, truth):
    model.net.eval()
    outputs = model.net(ECALs)
    loss = model.lossFunction(outputs, truth)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == truth.data).sum()/truth.shape[0]
    return (loss.data[0], accuracy)
