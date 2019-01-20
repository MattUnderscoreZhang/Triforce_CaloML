import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

input_files = [

              # # ### XGB, DNN, and CNN 
              # ('Output/Reg_EleFixed_LinReg_SumsOnly/results.h5', 'Linear Regression'),
              # ('Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/results.h5', 'XGBoost Baseline'),
              # ('Output/Reg_EleFixed_DNNCellsSumSkip2Init_ecal51_hcal11_2hid_512nodes_relu_learnrate0p001_L2reg0p01_dropout0p2_5epNoStop/results.h5', 'DNN'),
              # ('Output/Reg_EleFixed_CNNCellsSumSkip_ecal51_hcal11_ecalconv3_hcalconv3_learnrate0p001_L2reg0p01_dropout0p2_5epNoStop/results.h5', 'CNN'),

              ]

outdir = 'Output/'

particle_name = 'Electron'
#particle_name = 'Photon'
#particle_name = 'Pi0'
#particle_name = 'Charged Pion'
#particle_name = 'All Particles,'
#particle_name = 'Electron / Photon'
#particle_name = 'Electron / Pi0'

outlabel = 'Ele_xgb_angles'

results_dict = OrderedDict()

coarse_bins = np.arange(0,501,25)
coarse_bin_centers = np.arange(12.5,501,25)

fine_bins = np.arange(10,501,5)
fine_bin_centers = np.arange(12.5,501,5)

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

for (infile,label) in input_files:
    results = []
    with h5.File(infile,'r') as f:
        reg_pred_name = 'reg_energy_prediction'
        reg_true_name = 'energy'
        reg_pred = f[reg_pred_name][:].reshape(-1)
        reg_true = f[reg_true_name][:].reshape(-1)
    reldiff = (reg_true - reg_pred) / reg_true * 100.

    reldiff_means = binned_statistic(reg_true, reldiff, statistic='mean', bins=coarse_bins).statistic
    reldiff_sigmas = binned_statistic(reg_true, reldiff, statistic=np.std, bins=coarse_bins).statistic
    
    # fit for resolution versus energy
    params, cov = curve_fit(res_func, coarse_bin_centers, reldiff_sigmas, 
                            bounds=(0,100000) )
    print label, 'params:', params
    results_dict[label] = [reldiff_means,reldiff_sigmas,params]
    print label,reldiff_sigmas[0],reldiff_sigmas[-1]


# plot reldiff means
fig = plt.Figure()
ax = plt.gca()
ax.set_xlim(0,500)
#ax.set_ylim(-20,30)
ax.set_ylim(-30,5)
#ax.set_ylim(-10,10)
#ax.set_ylim(-10,5)
#ax.set_ylim(-2,10)
plt.xlabel('True Energy [GeV]')
plt.ylabel('Relative Bias [%]')
plt.title('%s Energy Regression'%(particle_name))
for label, results in results_dict.items():
    mark = 'v' if 'Linear Reg' in label else 'o'
    if 'XGBoost' in label: mark = 's'
    plt.plot(coarse_bin_centers,results[0],marker=mark,label=label)
plt.legend(loc='best')
plt.grid(True)
plt.savefig('%s/bias_vs_E_%s.pdf'%(outdir,outlabel))
plt.clf()

# plot reldiff means, zoomed
fig = plt.Figure()
ax = plt.gca()
ax.set_xlim(0,500)
ax.set_ylim(-2,2)
plt.xlabel('True Energy [GeV]')
plt.ylabel('Relative Bias [%]')
plt.title('%s Energy Regression'%(particle_name))
for label, results in results_dict.items():
    mark = 'v' if 'Linear Reg' in label else 'o'
    if 'XGBoost' in label: mark = 's'
    plt.plot(coarse_bin_centers,results[0],marker=mark,label=label)
plt.legend(loc='best')
plt.grid(True)
plt.savefig('%s/bias_vs_E_%s_zoom.pdf'%(outdir,outlabel))
plt.clf()


# plot reldiff sigmas, log scale
fig = plt.Figure()
ax = plt.gca()
ax.set_xlim(0,500)
ax.set_ylim(0.5,50)
#ax.set_ylim(0.5,200)
plt.xlabel('True Energy [GeV]')
plt.ylabel('Relative Resolution [%]')
plt.title('%s Energy Regression'%(particle_name))
for label, results in results_dict.items():
    mark = 'v' if 'Linear Reg' in label else 'o'
    if 'XGBoost' in label: mark = 's'
    plt.plot(coarse_bin_centers,results[1],marker=mark,label=label)
plt.legend(loc='best')
plt.yscale('log')
plt.grid(True,which='both')
plt.savefig('%s/res_vs_E_%s.pdf'%(outdir,outlabel))
plt.clf()


# plot reldiff sigmas with fit results, log scale
fig = plt.Figure()
ax = plt.gca()
ax.set_xlim(0,500)
ax.set_ylim(0.5,100)
#ax.set_ylim(2.0,100)
#ax.set_ylim(0.5,200)
plt.xlabel('True Energy [GeV]')
plt.ylabel('Relative Resolution [%]')
plt.title('%s Energy Regression'%(particle_name))
for label, results in results_dict.items():
    mark = 'v' if 'Linear Reg' in label else 'o'
    if 'XGBoost' in label: mark = 's'
    func_vals = res_func(fine_bin_centers, *results[2])
    func_plot = plt.plot(fine_bin_centers,func_vals)
    color = func_plot[0].get_color()
    scatter = plt.scatter(coarse_bin_centers,results[1],marker=mark,label=label,color=color)
#atlas_res_vals = res_func(fine_bin_centers, *atlas_res)*100.
#atlas_res_plot = plt.plot(fine_bin_centers, atlas_res_vals, label='$%.1f\%% / \sqrt{E} \oplus %.1f\%% \oplus %.1f / E$ (ATLAS)'%(100.0*atlas_res[0],100.0*atlas_res[1],atlas_res[2]))
lcd_res_vals = res_func(fine_bin_centers, *lcd_res)*100.
lcd_res_plot = plt.plot(fine_bin_centers, lcd_res_vals, linestyle='--',label='$%.1f\%% / \sqrt{E} \oplus %.1f\%% \oplus %.1f / E$ (LCD)'%(100.0*lcd_res[0],100.0*lcd_res[1],lcd_res[2]))
#cms_res_vals = res_func(fine_bin_centers, *cms_res)*100.
#cms_res_plot = plt.plot(fine_bin_centers, cms_res_vals, label=label='$%.1f\%% / \sqrt{E} \oplus %.1f\%% \oplus %.1f / E$ (CMS)'%(100.0*cms_res[0],100.0*cms_res[1],cms_res[2]))
plt.legend(loc='best')
plt.yscale('log')
plt.grid(True,which='both')
plt.savefig('%s/res_vs_E_%s_fits.pdf'%(outdir,outlabel))
plt.savefig('%s/res_vs_E_%s_fits.png'%(outdir,outlabel))
plt.clf()
