import matplotlib
matplotlib.use('Agg') # NOQA
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

# particle_name = 'Electron'
# particle_name = 'Photon'
particle_name = 'Pi0'
# particle_name = 'Charged Pion'

if particle_name is 'Electron':
    input_files = [
        ('Output/EleLin/results.h5', 'Linear Regression'),
        ('Output/EleXgb/results.h5', 'XGBoost Baseline'),
        ('/home/matt/Projects/calo/FinalResults/CLIC/EleChPi/TriForce/ChPi_DNN/validation_results.h5', 'DNN'),
        ('/home/matt/Projects/calo/FinalResults/CLIC/EleChPi/TriForce/ChPi_CNN/validation_results.h5', 'CNN'),
        ('/home/matt/Projects/calo/FinalResults/CLIC/EleChPi/TriForce/ChPi_GN/validation_results.h5', 'GN')
        ]
    outdir = 'Output/Plots/Ele/'
    outlabel = 'Ele_angles'
    pdgID = 11

if particle_name is 'Charged Pion':
    input_files = [
        ('Output/ChPiLin/results.h5', 'Linear Regression'),
        ('Output/ChPiXgb/results.h5', 'XGBoost Baseline'),
        ('/home/matt/Projects/calo/FinalResults/CLIC/EleChPi/TriForce/ChPi_DNN/validation_results.h5', 'DNN'),
        ('/home/matt/Projects/calo/FinalResults/CLIC/EleChPi/TriForce/ChPi_CNN/validation_results.h5', 'CNN'),
        ('/home/matt/Projects/calo/FinalResults/CLIC/EleChPi/TriForce/ChPi_GN/validation_results.h5', 'GN')
        ]
    outdir = 'Output/Plots/ChPi/'
    outlabel = 'ChPi_angles'
    pdgID = 211

if particle_name is 'Photon':
    input_files = [
        ('Output/GammaLin/results.h5', 'Linear Regression'),
        ('Output/GammaXgb/results.h5', 'XGBoost Baseline'),
        ('/home/matt/Projects/calo/FinalResults/CLIC/GammaPi0/TriForce/Output_DNN_4_512_0.0002_0.04/validation_results.h5', 'DNN'),
        ('/home/matt/Projects/calo/FinalResults/CLIC/GammaPi0/TriForce/Output_CNN_4_512_0.0004_0.12/validation_results.h5', 'CNN'),
        ('/home/matt/Projects/calo/FinalResults/CLIC/GammaPi0/TriForce/Output_GN_0_1024_0.0001_0.01/validation_results.h5', 'GN')
        ]
    outdir = 'Output/Plots/Gamma/'
    outlabel = 'Gamma_angles'
    pdgID = 22

if particle_name is 'Pi0':
    input_files = [
        # ('Output/Pi0Lin/results.h5', 'Linear Regression'),
        ('Output/Pi0Xgb/results.h5', 'XGBoost Baseline'),
        ('/home/matt/Projects/calo/FinalResults/Regression/paper_data/paper_Pi0Fixed/results_xgb.h5', 'XGBoost Baseline Paper'),
        ('/home/matt/Projects/calo/Training/TriForce/Output/RegTest/validation_results.h5', 'DNN'),
        ('/home/matt/Projects/calo/FinalResults/Regression/paper_data/paper_Pi0Fixed/results_dnn.h5', 'DNN Paper'),
        # ('/home/matt/Projects/calo/FinalResults/CLIC/GammaPi0/TriForce/Output_DNN_4_512_0.0002_0.04/validation_results.h5', 'DNN'),
        # ('/home/matt/Projects/calo/FinalResults/CLIC/GammaPi0/TriForce/Output_CNN_4_512_0.0004_0.12/validation_results.h5', 'CNN'),
        # ('/home/matt/Projects/calo/FinalResults/CLIC/GammaPi0/TriForce/Output_GN_0_1024_0.0001_0.01/validation_results.h5', 'GN')
        ]
    outdir = 'Output/Plots/Pi0/'
    outlabel = 'Pi0_angles'
    pdgID = 111

results_dict = OrderedDict()

coarse_bins = np.arange(0, 501, 25)
coarse_bin_centers = np.arange(12.5, 501, 25)

fine_bins = np.arange(10, 501, 5)
fine_bin_centers = np.arange(12.5, 501, 5)


def res_func(E, a, b, c):
    # resolution as a function of energy
    # parameters a,b,c to be determined in a fit
    return np.sqrt((a**2 / E) + b**2 + (c / E)**2)


# from PDG, Table 35.8
# http://pdg.lbl.gov/2017/reviews/rpp2017-rev-particle-detectors-accel.pdf
# ATLAS results consistent with fig 35 of https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/PERF-2013-05/
# LCD numbers from https://github.com/delphes/delphes/blob/master/cards/delphes_card_CLICdet_Stage1.tcl#L529
atlas_res = [0.1, 0.004, 0.3]
lcd_res = [0.156, 0.01, 0.0]
cms_res = [0.03, 0.005, 0.2]


for (infile, label) in input_files:
    results = []
    with h5.File(infile, 'r') as f:
        reg_pred_name = 'regressor_pred' if 'regressor_pred' in f.keys() else 'reg_energy_prediction'
        reg_true_name = 'regressor_true' if 'regressor_true' in f.keys() else 'reg_energy_truth' if 'reg_energy_truth' in f.keys() else 'energy'
        has_pdgID = 'pdgID' in f.keys()
        if has_pdgID:
            reg_pred = np.array(f[reg_pred_name])[np.array(f['pdgID']) == pdgID].reshape(-1)
            reg_true = np.array(f[reg_true_name])[np.array(f['pdgID']) == pdgID].reshape(-1)
        else:
            reg_pred = f[reg_pred_name][:].reshape(-1)
            reg_true = f[reg_true_name][:].reshape(-1)
    reldiff = (reg_true - reg_pred) / reg_true * 100.

    reldiff_means = binned_statistic(reg_true, reldiff, statistic='mean', bins=coarse_bins).statistic
    reldiff_sigmas = binned_statistic(reg_true, reldiff, statistic=np.std, bins=coarse_bins).statistic

    # fit for resolution versus energy
    params, cov = curve_fit(res_func, coarse_bin_centers, reldiff_sigmas, bounds=(0, 100000))
    print(label, 'params:', params)
    results_dict[label] = [reldiff_means, reldiff_sigmas, params]
    print(label, reldiff_sigmas[0], reldiff_sigmas[-1])


# plot reldiff means
fig = plt.Figure()
ax = plt.gca()
ax.set_xlim(0, 500)
# ax.set_ylim(-20, 30)
ax.set_ylim(-30, 5)
# ax.set_ylim(-10, 10)
# ax.set_ylim(-10, 5)
# ax.set_ylim(-2, 10)
plt.xlabel('True Energy [GeV]')
plt.ylabel('Relative Bias [%]')
plt.title('%s Energy Regression' % (particle_name))
for label, results in results_dict.items():
    mark = 'v' if 'Linear Reg' in label else 'o'
    if 'XGBoost' in label:
        mark = 's'
    plt.plot(coarse_bin_centers, results[0], marker=mark, label=label)
plt.legend(loc='best')
plt.grid(True)
plt.savefig('%s/bias_vs_E_%s.eps' % (outdir, outlabel))
plt.clf()

# plot reldiff means, zoomed
fig = plt.Figure()
ax = plt.gca()
ax.set_xlim(0, 500)
ax.set_ylim(-2, 2)
plt.xlabel('True Energy [GeV]')
plt.ylabel('Relative Bias [%]')
plt.title('%s Energy Regression' % (particle_name))
for label, results in results_dict.items():
    mark = 'v' if 'Linear Reg' in label else 'o'
    if 'XGBoost' in label:
        mark = 's'
    plt.plot(coarse_bin_centers, results[0], marker=mark, label=label)
plt.legend(loc='best')
plt.grid(True)
plt.savefig('%s/bias_vs_E_%s_zoom.eps' % (outdir, outlabel))
plt.clf()


# plot reldiff sigmas, log scale
fig = plt.Figure()
ax = plt.gca()
ax.set_xlim(0, 500)
ax.set_ylim(0.5, 50)
# ax.set_ylim(0.5, 200)
plt.xlabel('True Energy [GeV]')
plt.ylabel('Relative Resolution [%]')
plt.title('%s Energy Regression' % (particle_name))
for label, results in results_dict.items():
    mark = 'v' if 'Linear Reg' in label else 'o'
    if 'XGBoost' in label:
        mark = 's'
    plt.plot(coarse_bin_centers, results[1], marker=mark, label=label)
plt.legend(loc='best')
plt.yscale('log')
plt.grid(True, which='both')
plt.savefig('%s/res_vs_E_%s.eps' % (outdir, outlabel))
plt.clf()


# plot reldiff sigmas with fit results, log scale
fig = plt.Figure()
ax = plt.gca()
ax.set_xlim(0, 500)
ax.set_ylim(0.5, 100)
# ax.set_ylim(2.0, 100)
# ax.set_ylim(0.5, 200)
plt.xlabel('True Energy [GeV]')
plt.ylabel('Relative Resolution [%]')
plt.title('%s Energy Regression' % (particle_name))
for label, results in results_dict.items():
    mark = 'v' if 'Linear Reg' in label else 'o'
    if 'XGBoost' in label:
        mark = 's'
    func_vals = res_func(fine_bin_centers, *results[2])
    func_plot = plt.plot(fine_bin_centers, func_vals)
    color = func_plot[0].get_color()
    scatter = plt.scatter(coarse_bin_centers, results[1], marker=mark, label=label, color=color)
# atlas_res_vals = res_func(fine_bin_centers, *atlas_res)*100.
# atlas_res_plot = plt.plot(fine_bin_centers, atlas_res_vals, label='$%.1f\%% / \sqrt{E} \oplus %.1f\%% \oplus %.1f / E$ (ATLAS)'%(100.0*atlas_res[0],100.0*atlas_res[1],atlas_res[2]))
lcd_res_vals = res_func(fine_bin_centers, *lcd_res)*100.
lcd_res_plot = plt.plot(fine_bin_centers, lcd_res_vals, linestyle='--',label='$%.1f\%% / \sqrt{E} \oplus %.1f\%% \oplus %.1f / E$ (LCD)'%(100.0*lcd_res[0],100.0*lcd_res[1],lcd_res[2]))
# cms_res_vals = res_func(fine_bin_centers, *cms_res)*100.
# cms_res_plot = plt.plot(fine_bin_centers, cms_res_vals, label=label='$%.1f\%% / \sqrt{E} \oplus %.1f\%% \oplus %.1f / E$ (CMS)'%(100.0*cms_res[0],100.0*cms_res[1],cms_res[2]))
plt.legend(loc='best')
plt.yscale('log')
plt.grid(True,which='both')
plt.savefig('%s/res_vs_E_%s_fits.eps'%(outdir,outlabel))
plt.clf()
