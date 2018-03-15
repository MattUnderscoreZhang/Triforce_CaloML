import torch
from torch.autograd import Variable
from triforce_helper_functions import eval,eval_net
import numpy as np
############
# Analysis #
############

class Analyzer():
    def analyze(self, tools, testLoader, out_file):
        [classifier, regressor, GAN] = tools
        classifier_test_loss = 0
        regressor_test_loss = 0
        GAN_test_loss = 0
        n_test_batches = 0
        classifier_test_accuracy = 0
        GAN_test_accuracy = 0
        regressor_test_mean = 0
        regressor_test_sigma = 0
        regressor_pred = []
        regressor_true = []

        for data in testLoader:
            ECALs, HCALs, ys, energies = data
            ECALs, HCALs, ys, energies = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda()), Variable(energies.cuda())
            if (classifier != None): classifier_outputs = eval(classifier, ECALs, HCALs, ys)
            else: classifier_outputs = (0, 0,0,0,0,0)
            if (regressor != None): regressor_output = eval(regressor, ECALs, HCALs, energies)
            else: regressor_output = (0,0,0,0,0,0)
            if (GAN != None): GAN_outputs = eval(GAN, ECALs, HCALs, ys)
            else: GAN_outputs = (0, 0,0,0,0,0)
            classifier_test_loss += classifier_outputs[0]
            regressor_test_loss += regressor_output[0]
            GAN_test_loss += GAN_outputs[0]
            classifier_test_accuracy += classifier_outputs[1]
            GAN_test_accuracy += GAN_outputs[1]
            regressor_test_mean += regressor_output[-2]
            regressor_test_sigma += regressor_output[-1]
            regressor_pred.append(regressor_output[2])
            regressor_true.append(regressor_output[3])
            n_test_batches += 1
        classifier_test_loss /= n_test_batches
        GAN_test_loss /= n_test_batches
        classifier_test_accuracy /= n_test_batches
        GAN_test_accuracy /= n_test_batches
        regressor_test_mean /= n_test_batches
        regressor_test_sigma /= n_test_batches
        pred = np.concatenate(regressor_pred).ravel()
        true = np.concatenate(regressor_true).ravel()
        print('test accuracy: (C) %.4f, (G) %.4f' % (classifier_test_accuracy, GAN_test_accuracy))
        if (classifier != None): out_file.create_dataset("classifier_test_accuracy", data=classifier_test_accuracy)
        if (regressor != None): out_file.create_dataset("regressor_pred", data=pred)
        if (regressor != None): out_file.create_dataset("regressor_true", data=true)
        if (GAN != None): out_file.create_dataset("GAN_test_accuracy", data=GAN_test_accuracy)
        
    def analyze_net(self, nets, testLoader, out_file):
        [classifier, regressor, GAN] = nets
        classifier_test_loss = 0
        regressor_test_loss = 0
        GAN_test_loss = 0
        n_test_batches = 0
        classifier_test_accuracy = 0
        GAN_test_accuracy = 0
        regressor_test_mean = 0
        regressor_test_sigma = 0
        regressor_outputs = []
        for data in testLoader:
            ECALs, HCALs, ys, energies = data
            ECALs, HCALs, ys, energies = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda()), Variable(energies.cuda())
            if (classifier != None): classifier_outputs = eval_net(classifier, ECALs, HCALs, ys)
            else: classifier_outputs = (0, 0,0,0)
            if (regressor != None): regressor_output = eval_net(regressor, ECALs, HCALs, energies)
            else: regressor_output = (0, 0,0,0)
            regressor_outputs.append(regressor_output)
            if (GAN != None): GAN_outputs = eval_net(GAN, ECALs, HCALs, ys)
            else: GAN_outputs = (0, 0,0,0)
            classifier_test_loss += classifier_outputs[0]
            regressor_test_loss += regressor_output[0]
            regressor_test_mean += regressor_output[2]
            regressor_test_sigma += regressor_output[3]
            GAN_test_loss += GAN_outputs[0]
            classifier_test_accuracy += classifier_outputs[1]
            GAN_test_accuracy += GAN_outputs[1]
            n_test_batches += 1
        classifier_test_loss /= n_test_batches
        GAN_test_loss /= n_test_batches
        classifier_test_accuracy /= n_test_batches
        GAN_test_accuracy /= n_test_batches
        regressor_test_mean /= n_test_batches
        regressor_test_sigma /= n_test_batches
        print('test accuracy: (C) %.4f, (G) %.4f' % (classifier_test_accuracy, GAN_test_accuracy))
        return [classifier_outputs,regressor_outputs,GAN_outputs]
    
    
     
