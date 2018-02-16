import torch
from torch.autograd import Variable
from triforce_helper_functions import eval

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
        for data in testLoader:
            ECALs, HCALs, ys, energies = data
            ECALs, HCALs, ys, energies = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda()), Variable(energies.cuda())
            if (classifier != None): classifier_outputs = eval(classifier, ECALs, HCALs, ys)
            else: classifier_outputs = (0, 0)
            if (regressor != None): regressor_outputs = eval(regressor, ECALs, HCALs, energies)
            else: regressor_outputs = (0, 0)
            if (GAN != None): GAN_outputs = eval(GAN, ECALs, HCALs, ys)
            else: GAN_outputs = (0, 0)
            classifier_test_loss += classifier_outputs[0]
            regressor_test_loss += regressor_outputs[0]
            GAN_test_loss += GAN_outputs[0]
            classifier_test_accuracy += classifier_outputs[1]
            GAN_test_accuracy += GAN_outputs[1]
            n_test_batches += 1
        classifier_test_loss /= n_test_batches
        GAN_test_loss /= n_test_batches
        classifier_test_accuracy /= n_test_batches
        GAN_test_accuracy /= n_test_batches
        print('test accuracy: (C) %.4f, (G) %.4f' % (classifier_test_accuracy, GAN_test_accuracy))
        if (classifier != None): out_file.create_dataset("classifier_test_accuracy", data=classifier_test_accuracy)
        if (GAN != None): out_file.create_dataset("GAN_test_accuracy", data=GAN_test_accuracy)
