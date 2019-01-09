import os,sys
import h5py as h5
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

if len(sys.argv) < 2:
    print('usage: python do_regression_plots.py <input_file> [output_dir]')
    sys.exit()

infile = sys.argv[1]
outdir = ''
if len(sys.argv) < 3:
    outdir = os.path.dirname(infile)
else:
    outdir = sys.argv[2]

with h5.File(infile,'r') as f:
    try:
        reg_loss_history_test = f['regressor_loss_history_test'][:]
        reg_loss_history_train = f['regressor_loss_history_train'][:]
    except:
        reg_loss_history_test = None
        reg_loss_history_train = None
    reg_pred = f['regressor_pred'][:].reshape(-1)
    reg_true = f['regressor_true'][:].reshape(-1)
    try:
        reg_meandiff_history_test = f['regressor_meandiff_history_test'][:]
        reg_sigmadiff_history_test = f['regressor_sigmadiff_history_test'][:]
    except:
        reg_meandiff_history_test = None
        reg_sigmadiff_history_test = None
        
try:
    # plot loss vs training update
    n_epochs = reg_loss_history_test.shape[0]
    updates_per_epoch = int(reg_loss_history_train.shape[0]/n_epochs)

    reg_loss_history_train_epochs = np.sum(reg_loss_history_train.reshape(n_epochs,updates_per_epoch),axis=1)/updates_per_epoch

    plt.plot(reg_loss_history_train_epochs,label='Train loss',marker='o')
    plt.plot(reg_loss_history_test,label='Test loss',marker='o')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('%s/loss_vs_epoch.eps'%(outdir))
    plt.clf()

    # plot test loss only to be more legible
    plt.plot(reg_loss_history_test,label='Test loss',marker='o')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('%s/test_loss_vs_epoch.eps'%(outdir))
    plt.clf()
except:
    print 'loss history not present'

try:
    plt.plot(reg_meandiff_history_test,label='Test meandiff',marker='o')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('%s/test_meandiff_vs_epoch.eps'%(outdir))
    plt.clf()
except:
    print 'regressor_meandiff_history_test not present'

try:
    plt.plot(reg_sigmadiff_history_test,label='Test sigmadiff',marker='o')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('%s/test_sigmadiff_vs_epoch.eps'%(outdir))
    plt.clf()
except:
    print 'regressor_sigmadiff_history_test not present'

fine_bins = np.arange(0,501,5)
plt.hist2d(reg_true,reg_pred,bins=fine_bins,norm=colors.LogNorm())
plt.xlabel('True Energy [GeV]')
plt.ylabel('Predicted Energy [GeV]')
plt.savefig('%s/true_vs_pred_E.eps'%(outdir))
plt.savefig('%s/true_vs_pred_E.pdf'%(outdir))
plt.savefig('%s/true_vs_pred_E.png'%(outdir))
plt.clf()

reldiff = (reg_true - reg_pred) / reg_true * 100.
plt.hist(reldiff,bins=np.arange(-30,31,2))
plt.xlabel('(True - Pred)/True Energy [%]')
plt.savefig('%s/reldiff_E.eps'%(outdir))
plt.clf()

coarse_bins = np.arange(0,501,25)
coarse_bin_centers = np.arange(12.5,501,25)
reldiff_means = []
reldiff_sigmas = []
for i in range(len(coarse_bins)-1):
    bin_lower = coarse_bins[i]
    bin_upper = coarse_bins[i+1]
    indices = (reg_true > bin_lower) & (reg_true < bin_upper)
    reldiff_bin = reldiff[indices]
    plt.hist(reldiff_bin,bins=np.arange(-30,31,2))
    plt.xlabel('(True - Pred)/True Energy [%]')
    plt.savefig('%s/reldiff_E_bin%d.eps'%(outdir,i))
    plt.clf()
    reldiff_means.append(np.mean(reldiff_bin))
    reldiff_sigmas.append(np.std(reldiff_bin))

plt.plot(coarse_bin_centers,reldiff_means,marker='o')
plt.xlabel('True Energy [GeV]')
plt.ylabel('Relative Bias [%]')
plt.savefig('%s/mean_reldiff_vs_E.eps'%(outdir))
plt.clf()

plt.plot(coarse_bin_centers,reldiff_sigmas,marker='o')
plt.xlabel('True Energy [GeV]')
plt.ylabel('Relative Resolution [%]')
plt.savefig('%s/sigma_reldiff_vs_E.eps'%(outdir))
plt.clf()

