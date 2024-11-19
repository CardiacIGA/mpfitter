import numpy as np
from geomdl import BSpline
from vtnurbs.construct import MultipatchSurface
from ebdm.toolbox import surf_list_init, calc_connection
import ebdm.EBDM_3D_v1 as fit
_ = np.newaxis
## 1) Define the multipatch surface (4 patches)
# G──────────────H──────────────I
# │              │              │
# │              │              │
# │      3       │       2      │
# │              │              │
# │              │              │
# D──────────────E──────────────F
# │              │              │
# │              │              │
# │      0       │       1      │    y▲
# │              │              │     │
# │              │              │     │
# A──────────────B──────────────C     o───►
#                                    z    x
## Patches: 0,1,2,3
## Patch vertices: A, B C, D, E, F, G, H, I

# Following input is identical for Nutils multipatch input. However, it assumes that only quadratic B-splines are used! (Does not support cubic)
# patchverts : Patch vertices coordinates in 3D space (Does not contain repeated values!)
# patchcon   : Patch connectivity array of the individual patches
# cps        : Control points of the patches (concatenated and contains repeated cps)
# w          : Weights of the corresponding control points (also concatenated and contains repeated values)
# nelems     : Dict containing the number of elements for each boundary {(0,1): 1, (0,2): 2, (1,3): 2, ..}
# pnelems    : Dict containing the number of elements for each patch {0: (u,v), 1: (u,v), ..}
patchverts = np.array([[0,0,0], # A
                       [1,0,0], # B
                       [2,0,0], # C
                       [0,1,0], # D
                       [1,1,0], # E
                       [2,1,0], # F
                       [0,2,0], # G
                       [1,2,0], # H
                       [2,2,0]])# I

patchcon   = np.array([[0,1,3,4], # 0 (A,B,D,E)
                       [1,2,4,5], # 1 (B,C,E,F)
                       [4,5,7,8], # 2 (E,F,H,I)
                       [3,4,6,7]]) # 3 (D,E,G,H)

cps        = np.array([patchverts[0], [0.5, 0  , 0], patchverts[1],
                       [ 0, 0.5, 0 ], [0.5, 0.5, 0], [ 1, 0.5, 0 ],
                       patchverts[3], [0.5, 1  , 0], patchverts[4],
                       
                       patchverts[1], [1.5, 0  , 0], patchverts[2],
                       [ 1, 0.5, 0 ], [1.5, 0.5, 0], [ 2, 0.5, 0 ],
                       patchverts[4], [1.5, 1  , 0], patchverts[5],
                       
                       patchverts[4], [1.5, 1  , 0], patchverts[5],
                       [ 1, 1.5, 0 ], [1.5, 1.5, 0], [ 2, 1.5, 0 ],
                       patchverts[7], [1.5, 2  , 0], patchverts[8],
                       
                       patchverts[3], [0.5, 1  , 0], patchverts[4],
                       [ 0, 1.5, 0 ], [0.5, 1.5, 0], [ 1, 1.5, 0 ],
                       patchverts[6], [0.5, 2  , 0], patchverts[7]])

w          = np.ones(len(cps))
nelems     = {None : 1}     #{(0,1):1, (1,2):1, (3,4):1, (4,5):1, (6,7):1, (7,8): 1,
                            # (0,3):1, (3,6):1, (1,4):1, (4,7):1, (2,5):1, (5,8): 1} 
pnelems    = {None : (1,1)} #{f"patch{i}" : (1,1) for i in range(len(patchverts))}
multipatch_surface = MultipatchSurface("MultiPatchSurface", "surface", patchverts, patchcon, cps, w, nelems, pnelems)

delta = 0.04
multipatch_bspline_surface, multipatch_nurbs_surface = surf_list_init(multipatch_surface, delta)


## Translate the raw (Nutils) geometry and topology data to geomdl format (NURBS module)



## 2) Define some arbitrary (sparse) point cloud data (diagonal line x-y with arch in z direction)
npoints = 15 # nr of points
Ldomain = 2 # Length of the domain
transZ  = 0 # translation in Z-direction
x = np.linspace(0,Ldomain,npoints)
y = x.copy()
z = np.sin( x*np.pi/Ldomain ) + transZ
pointcloud = np.concatenate([ x[:,_], y[:,_], z[:,_] ], axis=1) 



##-------------------------------------------------------------##
## 2) Fit the 3D multipatch surface to the 3D point-cloud ##
##-------------------------------------------------------------##


# Give the input parameters for the algorithm
t_max       = 15        # Amount of iterations the EBDM runs per refinement
max_error   = 1e-5      # Error criterion for which the surfaces can be exported (average distance from data point to surfaces in meters)
max_refs    = 0         # state how many refinements can be done on the surfaces
cont_relax  = 0 #0.9         # Continuity relaxation factor (interpatch continuity)
connection  = calc_connection(patchcon)
#connectivity = np.array([ [0,1,2,3], [2,3,4,5], [6,7,4,5], [0,1,6,7], [1,7,3,5] ])


# Run the EBDM for both the outer and inner shell separately, with similar settings
surfs_list_out, continuities_out, errors_list_out, con_dict_out = fit.EBDM_3D(multipatch_bspline_surface, 
                                                                              pointcloud, 
                                                                              t_max, max_error, max_refs, connection, cont_relax)

#fit.export_finalsurfs(surfs_list_out[-1], surfs_list_in[-1], dp_tot, path, 'pdf')
fit.plot_total([surfs_list_out[-1]], data_points=pointcloud, evalpts=10)