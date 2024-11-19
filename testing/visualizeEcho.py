## A simple script that visualizes a DICOM file in Python
## @author: Robin Willems

import os
import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread

def rgb2gray(rgb):# applies the formula to convert RBG into brightness
    return np.dot(rgb, [0.2989, 0.5870, 0.1140])

CurrenDir = os.path.realpath(os.path.dirname(__file__)) # Current directory
folder   = "input/" # Specify the folder where the dicom file is stored
filename = "AP4CH.dcm" # Name of the dicom file to be loaded
saveFig  = False       # True if you want to save the figure, False otherwise     
framenr  = 54          # The framenumber we want to visualize 
clicking = True

# Load the Echo file
dicom = dcmread(os.path.join(CurrenDir,folder,filename))

ds = dicom
dx = ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX/100 # Convert [cm] to [m]
dy = ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaY/100

img = ds.pixel_array[framenr,:,:,:]
img_gray = rgb2gray(img)    

plt.imshow(img_gray, cmap="gray", aspect='auto')
plt.title(f"Echocardiogram view")
if clicking:
    dPix = np.array(plt.ginput(-1, timeout=0))
    X    = dPix*np.array([dx,dy])
if saveFig:
    plt.savefig(os.path.join(CurrenDir,"EchoView"))
plt.close()
