# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:46:02 2022

@author: lexve

Merged fitting algorithm toolbox for 2D splines on data points. Based on the GSA and CPBD methods, merged and applied to splines
"""

import math
from geomdl import BSpline, operations
import numpy as np
import splipy as sp
from scipy.spatial.transform import Rotation as R

class data_ctrlpt_class:
    _coordinate     = np.array([0,0,0])
    _velocity       = np.array([0,0,0])
    _acceleration   = np.array([0,0,0])
    _force          = np.array([0,0,0])
    _weight         = 0
    _ref_index      = 0

class data_point:
    _coordinate         = np.array([0,0,0])
    _ref_index          = 0
    _ref_frac           = 0
    _point_curve_ref    = np.array([0,0,0])
    _error_vec          = np.array([0,0,0])
    _lsq                = 0
    _weights            = []
    _lower_index        = 0
    _upper_index        = 0

class data_section:
    _lower  = 0
    _upper  = 0
    _list_lsq   = []

def transforma(dp_tot, view, curves):
    # Transform the data points to the x-z plane
    dpin        = xztrans(dp_tot[view + '_in'],  0)
    dpout       = xztrans(dp_tot[view + '_out'], 0)
    curve_in    = xztrans(curves[view + '_in'],  1)
    curve_out   = xztrans(curves[view + '_out'], 1)
    
    # Translate the data points to have the lowest point (outer wall) at z = 0, based on the interpolation
    translationz = - np.array([0, 0, np.min(curve_out[:,2])])
                              

    # Translate the data so that the center of the data points in x-direction is the z-axis
    x_low   = np.min(curves[view + '_out'][:,0])
    x_up    = np.max(curves[view + '_out'][:,0]) 
    translationx = - np.array([(x_up + x_low)/2,0,0])
    translation = translationx + translationz
    
    dp_tot[view + '_in']    = dpin + translation
    dp_tot[view + '_out']   = dpout + translation
    curves[view + '_in']    = curve_in + translation
    curves[view + '_out']    = curve_out + translation
    
    return dp_tot, curves

def xztrans(dp, i):
    if i == 0:
        dp = np.c_[dp, np.zeros(len(dp))]         # Add a column of zeros to create 3D coordinates
    dp[:, [2, 1]] = dp[:, [1, 2]]                       # Switch y-z column to work in x-z plane
    return dp

def find_points_z(points, z):
    '''Linear interpolate for points p and reference height z'''
    # Divide data points in x > 0 and x < 0
    negative_rows = []
    for i in range(len(points)):
        if points[i,0] < 0:
            negative_rows.append(i)
    points2 = points[negative_rows,:]
    points1 = np.delete(points,negative_rows, axis = 0)
    
    p1 = find_point_z(points1, z)
    p2 = find_point_z(points2, z)

    return [p1,p2]

def find_point_z(points, z):
    p1 = find_nearest(points[:,2], z)
    # Perform linear interpolation on reference height
    if points[p1,2] > z:
        p2 = p1 - 1
    else:
        p2 = p1 + 1
        p1, p2 = p2, p1
    
    point1 = points[p1,:]
    point2 = points[p2,:]
    
    dV = point1 - point2
    zx = dV[0] / dV[2]
    zy = dV[1] / dV[2]
    
    dz = z - point2[2]
    dx = dz * zx
    dy = dz * zy
    
    dI = np.array([dx,dy,dz])
    P = point2 + dI
    return P

def find_points_p1a(points, view):
     # Divide data points in x > 0 and x < 0
     negative_rows = []
     for i in range(len(points)):
         if points[i,0] < 0:
             negative_rows.append(i)
     points2 = points[negative_rows,:]
     points1 = np.delete(points,negative_rows, axis = 0)
     
     if view == 2:
         p1 = find_point_p1_xy(points1)
         p2 = find_point_p1_xy(points2)
     elif view == 4:
         p1 = find_point_p1_x(points1)
         p2 = find_point_p1_x(points2)

     return [p1,p2]

def rotate_set(dp, curves):

    dp_ref = dp['a4_out']
    if dp_ref[0,0] > dp_ref[1,0]:
        p1 = dp_ref[0,:]
        p2 = dp_ref[1,:]
    else:
        p1 = dp_ref[1,:]
        p2 = dp_ref[0,:]
    
    # determine vector and angle between highest points
    vec = p1 - p2
    alpha = np.arctan(vec[2]/vec[0])
    if vec[2] < 0:
        alpha  = - alpha
    r = R.from_rotvec(alpha * np.array([0, 1, 0]))
    
    for view in ['a2', 'a4', 'p1']:
        for wall in ['out', 'in']:
            tag = view + '_' + wall
            for i in range(len(dp[tag])):
                dp[tag][i,:] = np.dot(dp[tag][i,:], r.as_matrix())
            for i in range(len(curves[tag])):
                curves[tag][i,:] = np.dot(curves[tag][i,:], r.as_matrix())
    
    # Trim the data points to a certain height to prevent large inequalities in data densities
    # First find the maximal height
    max_z = dp['a2_out'][0,2]
    for view in ['a2', 'a4']:
        for wall in ['out', 'in']:
            for i in [0,1]:
                height = dp[view + '_' + wall][i,2]
                if height < max_z:
                    max_z = height
    
    # Remove all data points with z higher than max_z
    for view in ['a2', 'a4', 'p1']:
        for wall in ['out', 'in']:
            tag = view + '_' + wall
            delete_rows = []
            for i in range(len(dp[tag])):
                if dp[tag][i,2] > max_z:
                    delete_rows.append(i)
            dp[tag] = np.delete(dp[tag], delete_rows, 0)              
    
    # Translate the data set to set the center around the z-axis and the top data point at z = 0
    index = np.argmin(curves['a4_out'][:,2])
    translationxy = curves['a4_out'][index,:]
    translation = np.array([translationxy[0], translationxy[1], max_z])
    
    for view in ['a2', 'a4', 'p1']:
        for wall in ['out', 'in']:
            tag = view + '_' + wall
            for i in range(len(dp[tag])):
                dp[tag][i,:] = dp[tag][i,:] - translation
    
    return dp, curves

def find_point_p1_xy(points):
    dp = points[:,0] + points[:,1]
    index = np.argmin(np.abs(dp))
    return points[index,:]
    
def find_point_p1_x(points):
    index = np.argmin(np.abs(points[:,1]))
    return points[index,:]
    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def EBDM(init_curve, data_points, t_max, min_error, max_ctrlpt_addition):
    sample_dataset, avgerror_init = gather_sample_data(init_curve, data_points)
    curve_list = []
    error_list = []
    curve_list.append(init_curve)
    error_list.append(avgerror_init)
    
    # loop over addition of control points
    for ictrl in range(max_ctrlpt_addition + 1):
    # for factor in factorrange:
        # perform GSA for set amount of max iterations (curve is returned if it is within the error margin)
        for it in range(t_max):
            
            # Copmute weights of data points based on sections
            sample_dataset = EBDM_weights(init_curve, sample_dataset)
            
            # Calculate control point displacement
            new_ctrlpts = EBDM_disp_ctrlpts(init_curve, sample_dataset,it,t_max)      
            
            # New curve
            curve = BSpline.Curve()
            curve.degree        = init_curve.degree
            curve.ctrlpts       = new_ctrlpts
            curve.knotvector    = init_curve.knotvector    
            curve.sample_size   = init_curve.sample_size
            curve.evaluate()
            curve.evalptsnp     = np.array(curve.evalpts)
            curve_list.append(curve)
            
            init_curve = curve
            # Error evaluation
            sample_dataset, avgerror = gather_sample_data(curve, data_points)
            # check error criterion 
            if avgerror < min_error:
                return curve_list[-1]
            
            # check if curve is better than previous iteration
            elif avgerror < avgerror_init:
                avgerror_init = avgerror
                init_curve = curve
            error_list.append(avgerror_init)
        
        # add a control point if it is still not within the margin and is not the final iteration
        if avgerror > min_error and ictrl != max_ctrlpt_addition:
            ctrlpts_dataset = gather_data_cps(init_curve)
            ctrlpt_loc = ctrlpt_location(sample_dataset, ctrlpts_dataset, init_curve)
            operations.insert_knot(init_curve, [ctrlpt_loc], [1])
            init_curve.sample_size   = init_curve.sample_size
            init_curve.evaluate()
            init_curve.evalptsnp = np.array(init_curve.evalpts)
         
        sample_dataset = EBDM_weights(init_curve, sample_dataset)   
         
    return curve_list[-1]

def gather_data_cps(curve):
    ctrlpts_dataset = []
    # Find closest curve point to control point
    for i in range(curve.ctrlpts_size):
        ctrlpt_data = data_ctrlpt_class()
        
        # Compute minimal distance point to curve and store its index
        dist = 1e10
        for j in range(len(curve.evalpts)):
            # find closest curve point to control point
            dist_ref = distance(curve.evalpts[j],curve.ctrlpts[i])
            if dist_ref < dist or dist == 0 : 
                dist = dist_ref
                curve_index = j
            ctrlpt_data._ref_index = curve_index            
        ctrlpts_dataset.append(ctrlpt_data)
    return ctrlpts_dataset 

def gather_sample_data(curve, points):
    """ Store all other variables relevant for data point:
        index and coordinate of closest curve point, distance and vector to curve and lsq. Also outputs the overall average lsq
    Can be optimized by not running for all evalpts --> approach closest one"""
    error_tot = 0
    sample_dataset = []
    ref_indices = []
    for i in range(len(points)):
        sample_data = data_point()
        
        # Compute minimal distance point to curve and store its index
        dist, ref_curve_point = distance_point_curve(points,curve,i)
        
        # Vector curve to point
        vec_Xi_Pi = np.array(points[i]) - np.array(curve.evalpts[ref_curve_point])
        
        # Store error data per data point 
        sample_data._coordinate      = points[i]
        sample_data._ref_index       = ref_curve_point
        sample_data._point_curve_ref = curve.evalpts[ref_curve_point]
        sample_data._error_vec       = vec_Xi_Pi
        sample_data._lsq             = dist
        sample_data._ref_frac        = ref_curve_point / (curve.sample_size - 1)
        sample_data._weights         = []
        # append to whole list of sample data and error calculation
        sample_dataset.append(sample_data)
        error_tot += dist
        ref_indices.append(ref_curve_point)
        
    # sort the list based on ref_index of the curve
    for i in range(len(sample_dataset)):
        # find minimal index
        min_idx = i
        # Find the minimum element in remaining 
    
        for j in range(i+1, len(ref_indices)):
            if ref_indices[min_idx] > ref_indices[j]:
                min_idx = j
                  
        # Swap the found minimum element with 
        # the first element        
        sample_dataset[i], sample_dataset[min_idx] = sample_dataset[min_idx], sample_dataset[i]
        ref_indices[i], ref_indices[min_idx] = ref_indices[min_idx], ref_indices[i]
    
    # compute upper and lower boundaries for the weight
    for i in range(len(points)):
        # check if it is there is only one data point
        if i == 0 and len(points) == 1:
            sample_dataset[i]._lower_index = 0
            sample_dataset[i]._upper_index = 2*curve.sample_size - 1
        
        # Else for the first data point
        elif i == 0:
            sample_dataset[i]._lower_index = 0
            sample_dataset[i]._upper_index = sample_dataset[i+1]._ref_index + sample_dataset[i]._ref_index
        
        # For the final data point
        elif i == len(points) - 1:
            sample_dataset[i]._lower_index = sample_dataset[i-1]._upper_index
            sample_dataset[i]._upper_index = 2*curve.sample_size - 1
        
        # For all other data points
        else:
            sample_dataset[i]._lower_index = sample_dataset[i-1]._upper_index
            sample_dataset[i]._upper_index = sample_dataset[i+1]._ref_index + sample_dataset[i]._ref_index
    
    error_avg = error_tot/len(points)
    return sample_dataset, error_avg

def EBDM_weights(curve, sample_dataset):
    basis = sp.BSplineBasis(order = curve.order, knots = curve.knotvector)
    t = np.linspace(0,curve.knotvector[-1], 2*curve.sample_size)
    Nic = basis.evaluate(t) # Compute values for all shape functions at all evaluation points
    # compute the weights of each data point based on the uniformity and shape function
    for c in range(curve.ctrlpts_size):
        # Define total area under shape function
        Ntot = 0
        dXi = t[1] - t[0]
        for i in range(len(t)-1):
            dN = (Nic[i,c] + Nic[i+1,c]) / 2 * dXi
            Ntot += dN
        
        # loop over all data points      
        for i in range(len(sample_dataset)):
            # only compute weights if control point is not first or last (since these do not move)
            if c == 0 or c == curve.ctrlpts_size - 1:
                sample_dataset[i]._weights.append(0)
            else:
                weight = 0
                # print('indices are ', np.linspace(sample_dataset[i]._lower_index, sample_dataset[i]._upper_index - 1, sample_dataset[i]._upper_index - sample_dataset[i]._lower_index))                
                for j in np.linspace(sample_dataset[i]._lower_index, sample_dataset[i]._upper_index - 1, sample_dataset[i]._upper_index - sample_dataset[i]._lower_index):
                    # print('ds = ',j,'factor =', curve.ctrlpts_size)
                    j = int(j)
                    weight += (Nic[j,c] + Nic[j+1,c]) / 2 * dXi #/ Nmax
                # print('weight point ', i,' =', weight)
                sample_dataset[i]._weights.append(weight/Ntot) 
                
    return sample_dataset

def EBDM_disp_ctrlpts(curve, sample_dataset,it,t_max):
    # compute new coordinates control points based on displacement sections by data points
    new_ctrlpts = []
    for c in range(curve.ctrlpts_size):
        # keep the first and last control point at the same location
        if c == 0 or c == curve.ctrlpts_size - 1:
            new_ctrlpt = curve.ctrlpts[c]
            new_ctrlpts.append(new_ctrlpt)
        else:    
            displacement = np.array([0,0,0])
            for i in range(len(sample_dataset)):
                displacement = displacement + sample_dataset[i]._error_vec * sample_dataset[i]._weights[c] #* Nic[i,c]
            new_ctrlpt = curve.ctrlpts[c] + displacement#*math.e**(-it/t_max)
            new_ctrlpts.append(list(new_ctrlpt))
    
    return new_ctrlpts       

def ctrlpt_location(sample_dataset,ctrlpts_dataset, curve):
    sections    = []
    lsq         = []
    
    # define the sections by setting the lower and upper boundary index
    for i in range(len(ctrlpts_dataset)-1):
        section = data_section()
        section._lower = ctrlpts_dataset[i]._ref_index        
        section._upper = ctrlpts_dataset[i+1]._ref_index
        sections.append(section)
    
    # add the lsq per section and add them up in a seperate list 'lsq'
    for i in range(len(sections)):
        lsq_point = 0
        for j in range(len(sample_dataset)):
            # check if data point is within the range of 
            if sample_dataset[j]._ref_index >= sections[i]._lower and sample_dataset[j]._ref_index < sections[i]._upper:
                sections[i]._list_lsq.append(sample_dataset[j]._lsq)
                lsq_point += sample_dataset[j]._lsq
        lsq.append(lsq_point)
    
    # find the maximal value in list lsq and return its centre index
    index = np.argmax(lsq, axis=0)
    # compute factor
    factor = (sections[index]._upper + sections[index]._lower) / 2
    # normalize over the length of the curve
    factor = factor / len(curve.evalpts)
    
    return factor

def distance_point_curve(points,curve,i):
    if i == 0:
        dist = 0
        ref_curve_point = 0
    elif i == len(points) - 1:
        dist = 0
        ref_curve_point = len(curve.evalpts) - 1
    else:
        nodes = np.asarray(curve.evalptsnp)
        dist_2 = np.sum((nodes - points[i])**2, axis=1)
        ref_curve_point = np.argmin(dist_2)
        dist            = np.min(dist_2)

    return dist, ref_curve_point

def distance(point1, point2):
    distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
    return distance





















