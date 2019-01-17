from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import os

import h5py
import numpy as np

#####################
# PLOTTING FUNCTION #
#####################

def scatter3d(x, y, z, values, jets=None, eventVector=None, size=25, colorsMap='jet', saveName='Plots/Test.eps'):

    cm = plt.get_cmap(colorsMap) # set up color map
    cNorm = matplotlib.colors.Normalize(vmin=min(values), vmax=max(values)) # normalize to min and max of data
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm) # map data using color scale
    sizes = pow(values*100,2)/max(values)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, s=sizes, c=scalarMap.to_rgba(values)) # plot 3D grid
    ax.set_title("Energy in calorimeter")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Layer")

    ax.set_xlim3d(0, size-1)
    ax.set_ylim3d(0, size-1)
    ax.set_zlim3d(0, size-1)

    scalarMap.set_array(values)
    fig.colorbar(scalarMap) # plot color bar

    # dimensions for ECAL
    # CHECKPOINT - make sure HCAL doesn't plot any jets
    EventMinRadius = 1.5
    SliceCellRSize = 0.00636
    SliceCellZSize = 0.0051
    SliceCellPhiSize = 0.0051

    # also plot each protojet
    if jets is not [[0, 0], [0, 0]] and jets is not None:
        for jet in jets:
            # CHECKPOINT - geometry may be different (last layers thicker?)
            Ri = EventMinRadius # R to beginning of first layer
            Rf = EventMinRadius + size*SliceCellRSize # R to end of last layer
            dPhi = jet[0] - eventVector[0] # difference between phi of jet and event
            etaJet = jet[1] # eta is inf in +z and 0 straight up
            etaEvent = eventVector[1] # eta is inf in +z and 0 straight up
            thetaJet = 2*np.arctan(np.exp(-etaJet)) # theta is 0 in +z and 90 deg straight up
            thetaEvent = 2*np.arctan(np.exp(-etaEvent)) # theta is 0 in +z and 90 deg straight up
            # treating local ([x], [y], [z]) as ([Rphi], [z], [r])
            x1 = (size-1)/2 + np.tan(dPhi) / SliceCellPhiSize
            x2 = (size-1)/2 + np.tan(dPhi) / SliceCellPhiSize # huh, I guess the nsub algorithm can never capture a slope in x
            y1 = (size-1)/2 + (Ri/np.tan(thetaJet) - Ri/np.tan(thetaEvent)) / SliceCellZSize
            y2 = (size-1)/2 + (Rf/np.tan(thetaJet) - Rf/np.tan(thetaEvent)) / SliceCellZSize # or any jet not coming directly from origin
            ax.plot([x1, x2], [y1, y2], [0, size])

    plt.show()
    # plt.savefig(saveName, bbox_inches="tight")

#########################
# CHOOSE EVENTS TO PLOT #
#########################

data = h5py.File("/data/LCD/V3/littleData/Pi0/Pi0_0.h5")
events = [0, 1, 2, 3, 4]

saveDirectory = "Plots/NSub2/"
if not os.path.exists(saveDirectory): os.makedirs(saveDirectory)

# plot nsub jets
for eventN in events:

    eventVector = (60, 0, 0)
    # convert to (eta, phi)
    (px, py, pz) = eventVector
    pr = np.sqrt(px*px + py*py)
    EventPhi = np.arctan(py / px)
    p = np.sqrt(pr*pr + pz*pz)
    EventEta = 0.5 * np.log((p + pz) / (p - pz))
    eventVector = (EventEta, EventPhi)

    ECAL = data['ECAL']
    jet1 = data['N_Subjettiness/bestJets1']
    jet2 = data['N_Subjettiness/bestJets2']
    x, y, z = np.nonzero(ECAL[eventN]) # first event
    E = ECAL[eventN][np.nonzero(ECAL[eventN])]
    if len(E)>0: scatter3d(x, y, z, E, (jet1[eventN], jet2[eventN]), eventVector, size=25, saveName=saveDirectory+"ECAL_event"+str(eventN)+".eps")

    # HCAL = data['HCAL/HCAL']
    # x, y, z = np.nonzero(HCAL[eventN]) # first event
    # E = HCAL[eventN][np.nonzero(HCAL[eventN])]
    # if len(E)>0: scatter3d(x, y , z, E, size=5, saveName=saveDirectory+"HCAL_event"+str(eventN)+".eps")

# # plot averages
# ECAL_average = np.average(data['ECAL/ECAL'], axis=0)
# x, y, z = np.nonzero(ECAL_average) # first event
# E = ECAL_average[np.nonzero(ECAL_average)]
# scatter3d(x, y, z, E, size=25, saveName=saveDirectory+"ECAL_average.eps")

# HCAL_average = np.average(data['HCAL/HCAL'], axis=0)
# x, y, z = np.nonzero(HCAL_average) # first event
# E = HCAL_average[np.nonzero(HCAL_average)]
# scatter3d(x, y , z, E, size=5, saveName=saveDirectory+"HCAL_average.eps")
