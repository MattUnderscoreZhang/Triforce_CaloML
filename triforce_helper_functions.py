import torch
import pdb

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
        signal_accuracy, background_accuracy = sgl_bkgd_acc(predicted, truth.data)
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
        returndata = (loss.data[0], accuracy, outputs.data, truth.data, mean, sigma, signal_accuracy, background_accuracy)
    except:
        pdb.set_trace()
        returndata = (0, accuracy, outputs.data, truth.data, mean, sigma)
    return returndata

def sgl_bkgd_acc(predicted, truth): 
    """
    Considering 'predicted' and 'truth' are both in Tensor format
    sgl = 1
    bkgd = 0

    Return signal accuracy and background accuracy
    """
    truth_sgl = truth.nonzero() # indices of non-zero elements in truth
    truth_bkgd = (truth == 0).nonzero() # indices of zero elements in truth
    correct_sgl = 0
    correct_bkgd = 0
    for i in range(truth_sgl.shape[0]): 
        if predicted[truth_sgl[i]][0] == truth[truth_sgl[i]][0]: 
            correct_sgl += 1
    for i in range(truth_bkgd.shape[0]): 
        if predicted[truth_bkgd[i]][0] == truth[truth_bkgd[i]][0]: 
            correct_bkgd += 1

    return float(correct_sgl / truth_sgl.shape[0]), float(correct_bkgd / truth_bkgd.shape[0])
