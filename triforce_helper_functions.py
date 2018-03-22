import torch

# This module has to be here for dependency reasons

# train model
def train(model, ECALs, HCALs, truth):
    model.net.train()
    model.optimizer.zero_grad()
    outputs = model.net(ECALs, HCALs)
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
    try:
       model.net.eval()
       outputs = model.net(ECALs, HCALs)
       loss = model.lossFunction(outputs, truth)
       _, predicted = torch.max(outputs.data, 1)
    except:
       model.eval()
       outputs = model(ECALs, HCALs)
    try:
        accuracy = (predicted == truth.data).sum()/truth.shape[0]
    except:
        accuracy = 0 # ignore accuracy for energy regression
    # relative diff mean and sigma, for regression
    try:
        reldiff = 100.0*(truth.data - outputs.data)/truth.data
        mean = torch.mean(reldiff)
        sigma = torch.std(reldiff)
    except:
        mean = 0
        sigma = 0
    try:
       returndata = (loss.data[0], accuracy, outputs.data, truth.data, mean, sigma)
    except:
       returndata = (0, accuracy, outputs.data, truth.data, mean, sigma)
    return returndata
