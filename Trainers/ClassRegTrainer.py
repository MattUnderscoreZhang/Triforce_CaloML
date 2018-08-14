import torch
from torch.autograd import Variable
import os, sys
sys.dont_write_bytecode = True # prevent the creation of .pyc files
from Loader import transforms

#########################################
# Classification and regression trainer #
#########################################

class ClassRegTrainer:

    def __init__(self, combined_classifier):
        self.reset()
        self.previous_total_test_loss = 0
        self.previous_epoch_total_test_loss = 0
        self.delta_loss_below_threshold_count = 0
        self.stat_name = ['class_reg_loss', 'class_loss', 'reg_energy_loss', 'reg_eta_loss', 'reg_phi_loss', 'class_acc', 'class_sig_acc', 'class_bkg_acc', 'reg_energy_bias', 'reg_energy_res', 'reg_eta_diff', 'reg_eta_std', 'reg_phi_diff', 'reg_phi_std']
        self.history = historyData()
        self.model = combined_classifier

    def reset(self):
        self.microbatch_n = 0
        self.model.optimizer.zero_grad()

    def sgl_bkgd_acc(self, predicted, truth): 
        truth_sgl = truth.nonzero() # indices of non-zero elements in truth
        truth_bkgd = (truth == 0).nonzero() # indices of zero elements in truth
        correct_sgl_frac = 0
        correct_bkgd_frac = 0
        if len(truth_sgl) > 0:
            correct_sgl = 0
            for i in range(truth_sgl.shape[0]):
                if predicted[truth_sgl[i]][0] == truth[truth_sgl[i]][0]:
                    correct_sgl += 1
            correct_sgl_frac = float(correct_sgl / truth_sgl.shape[0])
        if len(truth_bkgd) > 0:
            correct_bkgd = 0
            for i in range(truth_bkgd.shape[0]):
                if predicted[truth_bkgd[i]][0] == truth[truth_bkgd[i]][0]:
                    correct_bkgd += 1
            correct_bkgd_frac = float(correct_bkgd / truth_bkgd.shape[0])
        return correct_sgl_frac, correct_bkgd_frac # signal acc, bkg acc

    def class_reg_eval(self,event_data, do_training=False, store_reg_results=False):
        if do_training:
            self.model.net.train()
        else:
            self.model.net.eval()
        outputs = self.model.net(event_data)
        return_event_data = {}
        # classification
        truth_class = Variable(event_data["classID"].cuda())
        class_reg_loss = self.model.lossFunction(outputs, event_data, options['lossTermWeights'])
        if do_training:
            microbatch_norm_loss = {}
            for key in class_reg_loss:
                microbatch_norm_loss[key] = class_reg_loss[key] / options['nMicroBatchesInMiniBatch']
            microbatch_norm_loss["total"].backward()
            self.microbatch_n += 1
            if (self.microbatch_n >= options['nMicroBatchesInMiniBatch']):
                self.microbatch_n = 0
                self.model.optimizer.step()
        _, predicted_class = torch.max(outputs['classification'], 1) # max index in each event
        class_sig_acc, class_bkg_acc = self.sgl_bkgd_acc(predicted_class.data, truth_class.data)
        # regression outputs. move first to cpu
        pred_energy = transforms.pred_energy_from_reg(outputs['energy_regression'].data.cpu(), event_data)
        truth_energy = event_data["energy"]
        reldiff_energy = 100.0*(truth_energy - pred_energy)/truth_energy
        pred_eta = transforms.pred_eta_from_reg(outputs['eta_regression'].data.cpu(), event_data)
        diff_eta = event_data["eta"] - pred_eta
        pred_phi = transforms.pred_phi_from_reg(outputs['phi_regression'].data.cpu(), event_data)
        diff_phi = event_data["phi"] - pred_phi
        # return values
        return_event_data["class_reg_loss"] = class_reg_loss["total"].data[0]
        return_event_data["class_loss"] = class_reg_loss["classification"].data[0]
        return_event_data["reg_energy_loss"] = class_reg_loss["energy"].data[0]
        return_event_data["reg_eta_loss"] = class_reg_loss["eta"].data[0]
        return_event_data["reg_phi_loss"] = class_reg_loss["phi"].data[0]
        return_event_data["class_acc"] = (predicted_class.data == truth_class.data).sum()/truth_class.shape[0]
        return_event_data["class_raw_prediction"] = outputs['classification'].data.cpu().numpy()[:,1] # getting the second number for 2-class classification
        return_event_data["class_prediction"] = predicted_class.data.cpu().numpy()
        return_event_data["class_truth"] = truth_class.data.cpu().numpy()
        return_event_data["class_sig_acc"] = class_sig_acc
        return_event_data["class_bkg_acc"] = class_bkg_acc
        return_event_data["reg_energy_bias"] = torch.mean(reldiff_energy)
        return_event_data["reg_energy_res"] = torch.std(reldiff_energy)
        return_event_data["reg_eta_diff"] = torch.mean(diff_eta)
        return_event_data["reg_eta_std"] = torch.std(diff_eta)
        return_event_data["reg_phi_diff"] = torch.mean(diff_phi)
        return_event_data["reg_phi_std"] = torch.std(diff_phi)
        return_event_data["energy"] = event_data["energy"].numpy()
        return_event_data["eta"] = event_data["eta"].numpy()
        return_event_data["openingAngle"] = event_data["openingAngle"].numpy()
        if store_reg_results:
            return_event_data["reg_energy_prediction"] = pred_energy.numpy()
            return_event_data["reg_eta_prediction"] = pred_eta.numpy()
            return_event_data["reg_phi_prediction"] = pred_phi.numpy()
            ECAL = event_data["ECAL"]
            return_event_data["ECAL_E"] = torch.sum(ECAL.view(ECAL.shape[0], -1), dim=1).view(-1).numpy()
            HCAL = event_data["HCAL"]
            return_event_data["HCAL_E"] = torch.sum(HCAL.view(HCAL.shape[0], -1), dim=1).view(-1).numpy()
            return_event_data["pdgID"] = event_data["pdgID"].numpy()
            return_event_data["phi"] = event_data["phi"].numpy()
            return_event_data["recoEta"] = event_data["recoEta"].numpy()
            return_event_data["recoPhi"] = event_data["recoPhi"].numpy()
        return return_event_data

    def class_reg_train(self, event_data):
        return self.class_reg_eval(event_data, do_training=True)

    def update_batch_history(self, data, train_or_test, minibatch_n):
        if train_or_test == TRAIN:
            eval_results = self.class_reg_train(data)
        else:
            eval_results = self.class_reg_eval(data)
        for stat in range(len(self.stat_name)):
            self.history[stat][train_or_test][BATCH][minibatch_n] += (eval_results[self.stat_name[stat]] / options['nMicroBatchesInMiniBatch'])

    def update_epoch_history(self):
        for stat in range(len(self.stat_name)):
            for split in [TRAIN, TEST]:
                self.history[stat][split][EPOCH].append(self.history[stat][split][BATCH][-1])

    def print_stats(self, timescale):
        print_prefix = "epoch " if timescale == EPOCH else ""
        for split in [TRAIN, TEST]:
            if timescale == EPOCH and split == TRAIN: continue
            print(print_prefix + split_name[split] + ' sample')
            for stat in range(len(self.stat_name)):
                if self.stat_name[stat] in options['print_metrics']:
                    print('  ' + self.stat_name[stat] + ':\t %8.4f' % (self.history[stat][split][timescale][-1]))
            print()

    def should_i_stop(self, timescale):
        if not options['earlyStopping']: return False
        total_test_loss = self.history[CLASS_LOSS][TEST][timescale][-1]
        if timescale == BATCH:
            relative_delta_loss = 1 if self.previous_total_test_loss==0 else (self.previous_total_test_loss - total_test_loss)/(self.previous_total_test_loss)
            self.previous_total_test_loss = total_test_loss
            if (relative_delta_loss < options['relativeDeltaLossThreshold']): self.delta_loss_below_threshold_count += 1
            if (self.delta_loss_below_threshold_count >= options['relativeDeltaLossNumber']): return True
            else: self.delta_loss_below_threshold_count = 0
        elif timescale == EPOCH:
            relative_delta_loss = 1 if self.previous_epoch_total_test_loss==0 else (self.previous_epoch_total_test_loss - epoch_total_test_loss)/(self.previous_epoch_total_test_loss)
            self.previous_epoch_total_test_loss = total_test_loss
            if (relative_delta_loss < options['relativeDeltaLossThreshold']): return True
        return False

    def train(self):

        minibatch_n = 0
        end_training = False

        for epoch in range(options['nEpochs']):

            train_or_test = TRAIN
            trainIter = iter(trainLoader)
            testIter = iter(testLoader)
            break_loop = False
            while True:
                self.reset()
                for _ in range(options['nMicroBatchesInMiniBatch']):
                    try:
                        if train_or_test == TRAIN:
                            data = next(trainIter)
                        else:
                            data = next(testIter)
                        self.update_batch_history(data, train_or_test, minibatch_n)
                    except StopIteration:
                        break_loop = True
                if break_loop:
                    break
                if train_or_test == TEST:
                    print('-------------------------------')
                    print('epoch %d, batch %d' % (epoch+1, minibatch_n))
                    self.print_stats(BATCH)
                    minibatch_n += 1
                if self.should_i_stop(BATCH): end_training = True
                if train_or_test == TEST:
                    train_or_test = TRAIN
                else:
                    train_or_test = TEST

            # end of epoch
            self.update_epoch_history()
            print('-------------------------------')
            self.print_stats(EPOCH)
            if self.should_i_stop(EPOCH): end_training = True

            # save results
            # should these be state_dicts?
            if options['saveFinalModel'] and (options['saveModelEveryNEpochs'] > 0) and ((epoch+1) % options['saveModelEveryNEpochs'] == 0):
                if not os.path.exists(options['outPath']): os.makedirs(options['outPath'])
                torch.save(self.model.net, options['outPath']+"saved_classifier_epoch_"+str(epoch)+".pt")
                if discriminator != None: torch.save(discriminator.net, options['outPath']+"saved_discriminator_epoch_"+str(epoch)+".pt")
                if generator != None: torch.save(generator.net, options['outPath']+"saved_generator_epoch_"+str(epoch)+".pt")

            if end_training: break
