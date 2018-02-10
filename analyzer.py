import torch

############
# Analysis #
############

class Default_Analyzer():
    def analyze(self, classifier, testLoader, out_file):
        correct = 0
        total = 0
        classifier.eval() # set to evaluation mode (turns off dropout)
        for data in testLoader:
            ECALs, HCALs, ys = data
            outputs = classifier.test(ECALs, ys)
            _, predicted = torch.max(outputs.data, 1)
            total += ys.size(0)
            correct += (predicted == ys.data).sum()
        print('Accuracy of the network on test samples: %f %%' % (100 * float(correct) / total))
        out_file.create_dataset("outputs", data=np.array(outputs.data))
        out_file.create_dataset("test_accuracy", data=np.array([100*float(correct)/total]))
