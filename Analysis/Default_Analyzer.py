import torch
from torch.autograd import Variable

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
        regressor_test_accuracy = 0
        GAN_test_accuracy = 0
        for data in testLoader:
            ECALs, HCALs, ys = data
            ECALs, HCALs, ys = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda())
            if (classifier != None): classifier_outputs = classifier.eval(ECALs, HCALs, ys)
            else: classifier_outputs = (0, 0)
            if (regressor != None): regressor_outputs = regressor.eval(ECALs, HCALs, ys)
            else: regressor_outputs = (0, 0)
            if (GAN != None): GAN_outputs = GAN.eval(ECALs, HCALs, ys)
            else: GAN_outputs = (0, 0)
            classifier_test_loss += classifier_outputs[0]
            regressor_test_loss += regressor_outputs[0]
            GAN_test_loss += GAN_outputs[0]
            classifier_test_accuracy += classifier_outputs[1]
            regressor_test_accuracy += regressor_outputs[1]
            GAN_test_accuracy += GAN_outputs[1]
            n_test_batches += 1
        classifier_test_loss /= n_test_batches
        regressor_test_loss /= n_test_batches
        GAN_test_loss /= n_test_batches
        classifier_test_accuracy /= n_test_batches
        regressor_test_accuracy /= n_test_batches
        GAN_test_accuracy /= n_test_batches
        print('test accuracy: (C) %.4f, (R) %.4f, (G) %.4f' % (classifier_test_accuracy, regressor_test_accuracy, GAN_test_accuracy))
        out_file.create_dataset("classifier_test_accuracy", data=classifier_test_accuracy)
        out_file.create_dataset("regressor_test_accuracy", data=regressor_test_accuracy)
        out_file.create_dataset("GAN_test_accuracy", data=GAN_test_accuracy)
