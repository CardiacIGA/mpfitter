# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:38:28 2022

@author: lexve

Modify verts and faces to stl files
"""

from nutils import export
import numpy as np

cps = np.load('cps.npy')
cps_faces = np.load('cps_faces.npy')

verts = np.load('verts.npy')
faces = np.load('faces.npy')


export.vtk('output\\Controlnet', cps_faces, cps) # Where conn is 2D and cps 3D