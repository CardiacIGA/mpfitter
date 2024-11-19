import scipy.io as sc
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, 'modules')
import ebdm.fit_toolbox_EBDM_2D as fit
from geomdl import BSpline
from scipy.spatial.transform import Rotation as R
from pydicom import dcmread

def rgb2gray(rgb):
    # applies the formula to convert RBG into brightness
    return np.dot(rgb, [0.2989, 0.5870, 0.1140])
#%%

# % Script for manually extracting data points of echocardiograms by
# % detecting the edges. Input: DICOM files, output: set of data points
# %% Giving the input parameters
patient = 3;                # Patientnr
seqnr   = 6;                # Sequence nr of the echo
N_pts   = 20;               # Amount of data points per edge
#set figure size
plt.rcParams["figure.figsize"] = [20, 20]
plt.rcParams["figure.autolayout"] = True

# %% Extracting relevant data

dirdicom = 'data\\p' + str(patient).zfill(2) + '\seq' + str(seqnr).zfill(2)
listfiles = []
for file in [f for f in os.listdir(dirdicom) if os.path.isfile(os.path.join(dirdicom, f))]:
    listfiles.append(file)

#make output figure directory
diroutput = dirdicom + '\\data\\input'
os.makedirs(diroutput, exist_ok=True)

#%%

dp = {}
for file in listfiles:
    view = file[-5:-3]

    ds = dcmread(dirdicom+'\\'+file)

    dx = ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
    dy = ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaY
   
    framenr = int(file[-2:])
    img = ds.pixel_array[framenr,:,:,:]
    img_gray = rgb2gray(img)    
    
    plt.imshow(img_gray, cmap="gray", aspect='auto')
    #plt.ax.set_aspect('auto')
    #plt.axis('auto')
    coordinates_input_in = np.array(plt.ginput(-1, timeout=0))
    dp[view + '_in'] =coordinates_input_in*np.array([dx,dy])/100
    plt.close()
    
    plt.imshow(img_gray, cmap="gray")
    plt.scatter(coordinates_input_in[:,0],coordinates_input_in[:,1], facecolors='none', edgecolors='g')
    coordinates_input_out = np.array(plt.ginput(-1, timeout=0))
    dp[view + '_out'] =coordinates_input_out*np.array([dx,dy])/100
    plt.close()
    
    plt.imshow(img_gray, cmap="gray")
    plt.scatter(coordinates_input_in[:,0],coordinates_input_in[:,1], facecolors='none', edgecolors='g')
    plt.scatter(coordinates_input_out[:,0],coordinates_input_out[:,1], facecolors='none', edgecolors='y')
    plt.savefig(diroutput+'\\pts_'+view)
    plt.close()



#%% -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:32:32 2022

@author: lexve

Combining the 3 different views to generate one overall 3D point cloud
"""

#%% Input the patient and sequence nr

# State the amount of fits that need to be taken for p1 on a2 and a4
sapltes     = 100 


#%% Construct splines to use as interpolation between data points
curves = {}
max_iter    = 15
min_error   = 1e-7
max_ref     = 10
saplte_size = 200


for view in ['a2', 'a4']:
    for wall in ['out', 'in']:
        dp_view = dp[view + '_' + wall]
        dp_view = np.c_[dp_view, np.zeros(len(dp_view))]
        # Create the spline with beginning and end point based on the data points
        curve = BSpline.Curve()
        cps = [list(dp_view[0,:]), list((dp_view[0,:] + dp_view[1,:]) / 2 + np.array([-0.005, -0.005, 0])), list(dp_view[1,:])]
        curve.degree        = 2
        curve.ctrlpts       = cps
        curve.knotvector    = [0,0,0,1,1,1]
        curve.saplte_size   = saplte_size
        curve.evaluate()
        curve.evalptsnp = np.array(curve.evalpts)
        
        # Run the EBDM with the input parameters
        curve = fit.EBDM(curve, dp_view, max_iter, min_error, max_ref)
        # print('curve' + view + wall + 'created')
        curves[view + '_' + wall] = curve.evalptsnp

for file in ['p1_out', 'p1_in']:
    dp_view = dp[file]
    dp_view = np.c_[dp_view, np.zeros(len(dp_view))]
    dp_view[:, [1, 0]] = dp_view[:, [0, 1]]                       # Switch x-y columns to match orientation
    # Create the spline with beginning and end point based on the data points
    curve = BSpline.Curve()
    ymaxarg = np.argmax(dp_view[:,1])
    ymax = np.max(dp_view[:,1])
    ymin = np.min(dp_view[:,1])
    xmax = np.max(dp_view[:,0])
    xmin = np.min(dp_view[:,0])
    
    cps = [list(dp_view[ymaxarg,:]), [xmax,ymax,0], [xmax,ymin,0], [xmin,ymin,0], [xmin,ymax,0], list(dp_view[ymaxarg,:])]
    curve.degree        = 2
    curve.ctrlpts       = cps
    curve.knotvector    = [0,0,0,1,2,3,4,4,4]
    curve.saplte_size   = saplte_size
    curve.evaluate()
    curve.evalptsnp = np.array(curve.evalpts)
    
    # Run the EBDM with the input parameters
    max_iter    = 30
    min_error   = 1e-7
    max_ref     = 1
    # 0
    curve = fit.EBDM(curve, dp_view, max_iter, min_error, max_ref)
    curves[file] = curve.evalptsnp
    dp[file] = dp_view
    







#%% Combine the 3 different views to 1 point cloud (a4 view is used as the reference points). 
# It is assumed that the point of view for the a2 and a4 is similar, but the a2 view is 
# tilted 45 deg. The p1 view is perpendicular to both a2 and a4 view

'''Set the a4 view in x-z plane'''
dp, curves = fit.transforma(dp, 'a4', curves)

'''Set the a2 view with an angle of 45 deg '''
dp, curves = fit.transforma(dp, 'a2', curves)

# Determine the rotation matrix
r = R.from_quat([0, 0, np.sin(np.pi/8), np.cos(np.pi/8)])
# for i in range(len(dp['a2_in'])):
#     dp['a2_out'][i] = np.dot(dp['a2_out'][i], r.as_matrix())
#     dp['a2_in'][i]  = np.dot(dp['a2_in'][i] , r.as_matrix())    
# for i in range(len(curves['a2_out'])):
#     curves['a2_out'][i] = np.dot(curves['a2_out'][i], r.as_matrix())
#     curves['a2_in'][i] = np.dot(curves['a2_in'][i], r.as_matrix())

dp['a2_out']     = np.dot(dp['a2_out'], r.as_matrix())
dp['a2_in']      = np.dot(dp['a2_in'],  r.as_matrix())
curves['a2_out'] = np.dot(curves['a2_out'], r.as_matrix())
curves['a2_in']  = np.dot(curves['a2_in'], r.as_matrix())

'''Set up the short axis view'''
heights = []
for view in ['a4_out', 'a2_out', 'a4_in', 'a4_out']:
    min_height = np.min(dp[view][0:2, 2])
    heights.append(min_height)

height = np.min( heights )

errors      = []
trans       = []
z           = np.linspace(0.2*height, 0.8*height,sapltes)

# First center it around the z-axis
x_low   = np.min(curves['p1_out'][:,0])
x_up    = np.max(curves['p1_out'][:,0])
y_low   = np.min(curves['p1_out'][:,1])
y_up    = np.max(curves['p1_out'][:,1])
translation = - np.array([(x_up + x_low)/2, (y_up + y_low)/2, 0])
view = 'p1'
dp[view + '_in']        += translation
dp[view + '_out']       += translation
curves[view + '_in']    += translation
curves[view + '_out']   += translation

points = curves['p1_out']
# Divide data points in x > 0 and x < 0
negative_rows = []
for i in range(len(points)):
    if points[i,0] < 0:
        negative_rows.append(i)
# negative_rows = np.array(negative_rows)
points2 = points[negative_rows,:]
points1 = np.delete(points,negative_rows, axis = 0)

# Find the curve points corresponding to the closest points to a4 and a2
p1_a2_out   = fit.find_points_p1a(curves['p1_out'], 2)
p1_a4_out   = fit.find_points_p1a(curves['p1_out'], 4)
p1_a2_in    = fit.find_points_p1a(curves['p1_in'],  2)
p1_a4_in    = fit.find_points_p1a(curves['p1_in'],  4)

# Find centers of the closest a2 and a4 points
a2_center = (p1_a2_out[0] + p1_a2_out[1]) / 2
a4_center = (p1_a4_out[0] + p1_a4_out[1]) / 2
# Overall center
p1_center = (a2_center + a4_center) / 2

for i in z:
    # Find the x-y values for the a2 and a4 curves at current z
    a_data = {}
    a_data['a2_in']  = fit.find_points_z(curves['a2_in'],  i)
    a_data['a2_out'] = fit.find_points_z(curves['a2_out'], i)
    a_data['a4_in']  = fit.find_points_z(curves['a4_in'],  i)
    a_data['a4_out'] = fit.find_points_z(curves['a4_out'], i)
    
    # Find centers of the a2 and a4 points
    a2_center = (a_data['a2_out'][0] + a_data['a2_out'][1]) / 2
    a4_center = (a_data['a4_out'][0] + a_data['a4_out'][1]) / 2
    # Overall center
    a_center  = (a2_center + a4_center) / 2
    
    # Compute translation of the p1 slice
    translation = a_center - p1_center
    trans.append(translation)
    p1_out    = curves['p1_out'] + translation
    p1_in     = curves['p1_in']  + translation
    
    # Compute error between the p1 slice and the a4 and a2 slices
    # Find again the points closest on the curves to the slices
    p1_data = {}
    p1_data['a2_out']   = fit.find_points_p1a(p1_out, 2)
    p1_data['a4_out']   = fit.find_points_p1a(p1_out, 4)
    p1_data['a2_in']    = fit.find_points_p1a(p1_in,  2)
    p1_data['a4_in']    = fit.find_points_p1a(p1_in,  4)
    
    error = []
    for view in ['a2', 'a4']:
        for wall in ['in', 'out']:
            j = view + '_' + wall
            for i in [0,1]:
                # Consider the error the distance between the relevant points
                error_sec = np.sqrt( (p1_data[j][i][0] - a_data[j][i][0])**2 + (p1_data[j][i][1] - a_data[j][i][1])**2)
                error.append(error_sec)
    error = np.average(np.array(error)) *1000
    errors.append(error)

# Find the height for which the error is minimal
index = np.argmin(errors)
# Translate p1 data points with the corresponding translation
translation_p1 = trans[index]
dp['p1_in']        += translation_p1
dp['p1_out']       += translation_p1

# Rotate the whole set of data points and flatten it at the top, based on the a4 view
dp, curves = fit.rotate_set(dp, curves)

'''Export the data'''
# Create one matrix for the inner side and one matrix for the outer side
dp_in  = [dp['p1_in'] , dp['a2_in']  , dp['a4_in']]
dp_in = np.concatenate(dp_in)
dp_out = [dp['p1_out'] , dp['a2_out'] , dp['a4_out']]
dp_out = np.concatenate(dp_out)

# Other saving functions
# np.save("dp_s5", dp)
# np.save("curves_s5", curves)
# for view in ['a2', 'a4', 'p1']:
#     dp_sec = np.concatenate([dp[view + '_in'], dp[view + '_out']], axis = 0)
#     np.save(view + '.npy', dp_sec)

# Slightly rotate the data for better usage in the EBDM
r = R.from_quat([0, 0, -np.sin(np.pi/16), np.cos(np.pi/16)])
for i in range(len(dp_in)):
    dp_in[i]  = np.dot(dp_in[i],  r.as_matrix())
    dp_out[i] = np.dot(dp_out[i], r.as_matrix())

# Translate all data points to have the highest data point at z = 0
z_max = np.max([np.max(dp_in[:,2]), np.max(dp_out[:,2])])
translation = np.array([0,0,-z_max])
dp_in += translation 
dp_out += translation 

# Save in directory
np.save(dirdicom + '\\data\\input\\dp_in', dp_in)
np.save(dirdicom + '\\data\\input\\dp_out', dp_out)

#%% Post processing for visualisation
plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})
plt.plot(z * 1000, errors, lw = 1.5, color = '#0077BB')
plt.xlabel('Height [mm]')
plt.ylabel('Average error [mm]')
plt.grid()
# plt.savefig('p1_errorcon.eps', format = 'eps')

blue    = '#0077BB'
cyan    = '#33BBEE'
teal    = '#009988'
orange  = '#EE7733'
red     = '#CC3311'
magenta = '#EE3377'
grey    = '#BBBBBB'

fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
ax = fig.add_subplot(projection='3d')
ax.set_xlim3d(-0.04,0.04)
ax.set_ylim3d(-0.04,0.04)
ax.set_zlim3d(-0.08, 0)
ax.view_init(elev= 15 , azim= -120)

for i in ['a4_in', 'a4_out']:
    if i == 'a4_in':
        ax.scatter(dp[i][:,0], dp[i][:,1], dp[i][:,2], color = 'black', linewidth = 0.1, zorder = 5, label = '4 chamber view')
    else:
        ax.scatter(dp[i][:,0], dp[i][:,1], dp[i][:,2], color = 'black', linewidth = 0.1, zorder = 5)  
for i in ['a2_in', 'a2_out']:
    if i == 'a2_in':
        ax.scatter(dp[i][:,0], dp[i][:,1], dp[i][:,2], color= red, linewidth = 0.1, zorder = 5, label = '2 chamber view')
    else:
        ax.scatter(dp[i][:,0], dp[i][:,1], dp[i][:,2], color= red, linewidth = 0.1, zorder = 5)
for i in ['p1_in', 'p1_out']:
    if i == 'p1_in':
        ax.scatter(dp[i][:,0], dp[i][:,1], dp[i][:,2], color=blue, linewidth = 0.1, zorder = 5, label = 'Short axis view')
    else:
        ax.scatter(dp[i][:,0], dp[i][:,1], dp[i][:,2], color= blue, linewidth = 0.1, zorder = 5)    
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.axes.zaxis.set_ticklabels([])
plt.legend()
plt.show()
plt.savefig('3D_sliced.eps', format = 'eps')


plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})

n = 'a4_out'
plt.plot(curves[n][:,0], curves[n][:,2], color = '#0077BB', lw = 1.5, label = 'Fitted outer spline')
plt.scatter(dp[n][:,0], dp[n][:,2], color = 'black', label = 'Outer data points')

n = 'a4_in'
plt.plot(curves[n][:,0], curves[n][:,2], color = '#CC3311', lw = 1.5, label = 'Fitted inner spline')
plt.scatter(dp[n][:,0], dp[n][:,2], color = 'grey', label = 'Inner data points')
plt.legend(loc = 1)
plt.ylim(-0.005,0.12)
plt.show()
plt.savefig('a4_spline_fit.eps', format = 'eps')





































