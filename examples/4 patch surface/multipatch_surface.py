import numpy as np
from geomdl import BSpline
from vtnurbs.construct import MultipatchSurface
from ebdm.toolbox import init_multipatch_surfaces, update_multipatch_object
import ebdm.EBDM_3D as fit
from ebdm.postprocess import Plotting
import os
directC=os.path.realpath(os.path.dirname(__file__)) # Current directory
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

patchcon   = np.array([[0,1,3,4],  # 0 (A,B,D,E)
                       [1,2,4,5],  # 1 (B,C,E,F)
                       [4,5,7,8],  # 2 (E,F,H,I)
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
boundary_names = { 'side'   : [(0,'left') ,(0,'bottom'),(1,'top'),(1,'left'),
                               (2,'right'),(2,'top'), (3,'bottom'),(3,'right') ],
                   'top'    : [(2,'right')   ,(3,'right')   ],
                   'bottom' : [(0,'left'),(1,'left')],
                   'left'   : [(0,'bottom')  ,(3,'bottom')  ],
                   'right'  : [(1,'top') ,(2,'top') ]}                       
## 1) Create the multipatchsurface object instance, found in the vtnurbs module
multipatch_surface = MultipatchSurface("MultiPatchSurface", "surface", patchverts, patchcon, cps, w, nelems, pnelems, bnames=boundary_names)

## 2) Specify the constraints of the surface fit, if any
# constraints dict, specify in what direction specific boundaries are constraint
constraints = {}#{#"left"      : ("vector", np.array([0,0,0])),
               #"right"     : ("vector", np.array([0,0,0])),
               #"bottom"    : ("plane", dict(normal=np.array([0,1,0]), point=np.array([0,-1,0]))),
               #None        : ("normal", "fixed")}
               #"bottom"  : ("plane", dict(normal=np.array([0,1,0]), point=np.array([0,-1,0])))} # Constrain the side boundary to move only in normal direction
#             {boundary : (  type  ,       value      )}
## Constraints types:
# Boundary:
# - 'side', 'left, ..                     (Nutils named boundaries, assigned with the topo.withboundary() function)
# - 'patch0-left' or 'patch0-umin',   ..  (Indivual patch boundaries, either use the nutils convention 'left', 'right', or the splipy convention 'umin','umax etc.')
# - 'patch1-patch2', 'patch2-patch3', ..  (Patch interfaces)
# - 'patch0', 'patch1', ..                (Entire patch) 
# - None                                  (All control points that have not been constrained yet)  

# Type:   
# - 'vector'         , np.array
# - 'plane'          , dict(normal=np.array, point=np.array) # Also projects points that are not in plane, into plane (hard)
# - 'plane-parallel' , dict(normal=np.array, point=np.array) # This plane type does not force points into the plane, but displaces points outside of plane parallel to plane
# - 'normal'         , 'fixed' (use normal of initial geometry) or 'update' (update normal every iteration based on fitted geometry)

## 3) Convert vtnurbs multipatch surface to a list of splipy surface objects -> This format is used for the ebdm fitting.
multipatch_nurbs_surface = init_multipatch_surfaces(multipatch_surface, constraints=constraints)

## 4) Define some arbitrary (sparse) point cloud data (diagonal line x-y with arch in z direction)
# npoints = 16 # nr of points
# Ldomain = 2 # Length of the domain
# transZ  = 0 # translation in Z-direction
# x = np.linspace(0,Ldomain,npoints)
# y = x.copy()
# z = np.sin( x*np.pi/Ldomain ) + transZ
# pointcloud = np.concatenate([ x[:,_], y[:,_], z[:,_] ], axis=1) 
#pointcloud = np.array([ [0.25,0.5,0.5], [0.75,0.5,1], [1.5,0.5,0.4], [0.5,1.5,-0.3], [1.5,1.5,0.7] ])
#pointcloud = np.array([ [0.5,0.5,0.5], [1.5,0.5,0.4], [0.5,1.5,-0.3], [1.5,1.5,0.7] ])

# b) Dense pointcloud option
# x, y = np.meshgrid(x,y)
# x = x.ravel()
# y = y.ravel()
# z =  np.sin( x*np.pi/Ldomain )*np.sin( y*np.pi/Ldomain ) + transZ
# pointcloud = np.concatenate([ x[:,_], y[:,_], z[:,_] ], axis=1)

# c) Asymmetric point cloud (sparse)
npoints = 16 # nr of points
Ldomain = 2 # Length of the domain
transZ  = 0 # translation in Z-direction
x = np.linspace(0,Ldomain,npoints)
y = np.ones(len(x))
x = np.concatenate([x, x])
y = np.concatenate([ 0.2*y, 1.8*y ])
z = np.sin( x*np.pi/Ldomain ) + transZ
pointcloud = np.concatenate([ x[:,_], y[:,_], z[:,_] ], axis=1)

# d) Patch interface point cloud
# npoints = 8 # nr of points
# Ldomain = 2 # Length of the domain
# transZ  = 1 # translation in Z-direction
# y = np.linspace(0,Ldomain,npoints)
# x = np.ones(len(y))
# z = transZ*np.ones(len(y))
# pointcloud = np.concatenate([ x[:,_], y[:,_], z[:,_] ], axis=1)



## 5) Fit the 3D multipatch surface to the 3D point-cloud
# Initialize the EBDM algorithm
EBDM = fit.EBDM(*multipatch_nurbs_surface, datapoints=pointcloud, evalpts=100)

# Run the algorithm
max_iter         = 350   # Amount of iterations the EBDM runs per refinement
atol             = 1e-10 # Absolute tolerance, has the same unit of the datapoints
rtol             = 1e-9  # Relative tolerance, has no unit, is scaled with first displacement error
max_refs         = 2     # state how many refinements can be done on the surfaces
continuity_relax = 1     # Continuity relaxation factor (interpatch continuity)
displace_relax   = 0.1   # Displacement (over)relaxation factor
cont_iters_only  = 50     # Number of iterations that are used prior to reaching max_iter where only continuity is adjusted, no displacement
reval_every      = 30    # Re-evaluate the surfaces (sampling) every nth iteration, low values decrease performance
ref_rtol         = 1e-2  # The relative tolerance (rtol) below which a uniform refinement is applied
refine_wait      = 10
recons_every     = 1     # Reconstrain the 'update' normal constraint every nth iteration. Only works if 
save_iters_every = None  # Save iterations of the fit every nth iteration
constrainCont    = True  # Constrain the dsplacement as a result of the continuity correction (not recommended)
multipatch_nurbs_surface_fit = EBDM.run(max_iter=max_iter, max_refs=max_refs, atol=atol, rtol=rtol, ref_rtol=ref_rtol, refine_wait=refine_wait,
                                        displace_relax=displace_relax, 
                                        continuity_relax=continuity_relax, continuity_iters_only=cont_iters_only, 
                                        reval_every=reval_every, recons_every=recons_every, save_iters_every=save_iters_every,
                                        constrainCont=constrainCont)
error_info = EBDM.error_info() # Return the error info (displacement error, continuity error, when refinement was applied etc.), returns a dict

## 6) Postprocessing
if save_iters_every == None:
    plot = Plotting(multipatch_nurbs_surface_fit, pointcloud, error_info=error_info)
    plot.surface(show_cerror=True)
    plot.error(relative=True, energy=True)
    #plot.error_video(os.path.join(directC, "output", "video", "Error plot.mp4")) # Animated video of the error plot
    #plot.surface_video(gifName=os.path.join(directC,"output","video","Surface fit coarse 0.png"))

    
    ## Convert back to multipatchobject
    multipatch_surface_fitted = update_multipatch_object(multipatch_surface, multipatch_nurbs_surface_fit)
    # multipatch_surface_fitted.save_vtk(os.path.join(directC, "output", "vtk", "Multipatch surface"))
    # multipatch_surface_fitted.save_controlnet(os.path.join(directC, "output", "vtk", "Multipatch surface control net"))

else: # save to results to multiple sub-vtk files to make video
    for i, mp_nurbs_fit in enumerate(multipatch_nurbs_surface_fit):
        multipatch_surface_endo_fitted = update_multipatch_object(multipatch_surface, mp_nurbs_fit)

        # Save surface
        multipatch_surface_endo_fitted.save_vtk(os.path.join(directC, "output", "vtk (video)", f"Multipatch surface iter {str(i).zfill(3)}"))

        # Save control net
        multipatch_surface_endo_fitted.save_controlnet(os.path.join(directC, "output", "vtk (video)", f"Multipatch controlnet iter {str(i).zfill(3)}"))



