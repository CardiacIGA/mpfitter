import numpy as np
from geomdl import BSpline
from vtnurbs.construct import MultipatchSurface, MultipatchSolid
from ebdm.toolbox import init_multipatch_surfaces, update_multipatch_object
import ebdm.EBDM_3D as fit
from ebdm.postprocess import Plotting
from ebdm.LV_utils import get_ellipsoid_features
import os
_=np.newaxis; mm2m=1e-3
directC=os.path.realpath(os.path.dirname(__file__)) # Current directory
directG=os.path.split(os.path.split(directC)[0])[0] # Global directory of ebdm module
# Filenames to be loaded
fileEndoPoints=os.path.join(directG,"pointclouds","atlas","Atlas endocardium.txt")
fileEpiPoints =os.path.join(directG,"pointclouds","atlas","Atlas epicardium.txt")
fileEndoTempl =os.path.join(directG,"template","left ventricle","Endocardium_GEOM_DATA.txt") 
fileEpiTempl  =os.path.join(directG,"template","left ventricle","Epicardium_GEOM_DATA.txt")



## 1) Load the Atlas point cloud
endocard_points = np.loadtxt(fileEndoPoints,skiprows=1,delimiter=',')*mm2m # Endocardium
epicard_points  = np.loadtxt(fileEpiPoints,skiprows=1,delimiter=',')*mm2m  # Epicardium
# TODO add pre-processing, slicing the base such that all max(z) points are within the same plane
# base_plane  = dict(normal=np.array([-0.04228951, -0.04014556,  0.99829852]),
#                    point =np.array([-0.0011997779787097495, -0.0001704055356167005, 0.02682202550820625]))
base_plane  = dict(normal=np.array([0, 0, 1]), # z-base
                   point =np.array([-0.0011997779787097495, -0.0001704055356167005, 0.02682202550820625]))

## 2)a Load the NURBS left ventricle template
EndoTemplate = MultipatchSurface.load_txt(fileEndoTempl, returnDict=True) # Endocardium
EpiTemplate  = MultipatchSurface.load_txt(fileEpiTempl, returnDict=True)  # Epicardium
# set surface boundary names
#multipatch_endo_surface.print_localbound_names()
boundary_names = {"base" : [(0,'bottom'), (1,'bottom'), (2,'bottom'),(3,'bottom')]}
#vertical-interface (same plane)
#patch0-left + patch1-right; patch1-left + patch2-left


multipatch_endo_surface = MultipatchSurface("MultiPatchSurface", "endocardium", EndoTemplate["patch vertices"],
                                            EndoTemplate["patch connectivity"], EndoTemplate["control points"],
                                            EndoTemplate["weights"], EndoTemplate["nelems"], {None : (1,1)}, bnames=boundary_names)

multipatch_epi_surface = MultipatchSurface("MultiPatchSurface", "epicardium",  EpiTemplate["patch vertices"],
                                            EpiTemplate["patch connectivity"], EpiTemplate["control points"],
                                            EpiTemplate["weights"], EpiTemplate["nelems"], {None : (1,1)}, bnames=boundary_names)



## 4) Initialize and convert the MultipatchSurface to usable SpliPy objects
constraints_epi  = {"base"  : ("plane", base_plane)}
constraints_endo = {"base"  : ("plane", base_plane)}
multipatch_nurbs_endo_surface = init_multipatch_surfaces(multipatch_endo_surface, constraints=constraints_endo) # Epicardium
multipatch_nurbs_epi_surface  = init_multipatch_surfaces(multipatch_epi_surface,  constraints=constraints_epi) # Epicardium


## 5) Specify the EBDM input parameters and run the fitting
max_iter         = 300    # Amount of iterations the EBDM runs per refinement
atol             = 1e-10  # Absolute tolerance, has the same unit of the datapoints
rtol             = 1e-9   # Relative tolerance, has no unit, is scaled with first displacement error
max_refs         = 2      # state how many refinements can be done on the surfaces
continuity_relax = 1      # Continuity relaxation factor (interpatch continuity)
displace_relax   = 3      # Displacement (over)relaxation factor
cont_iters_only  = 50     # Number of iterations that are used prior to reaching max_iter where only continuity is adjusted, no displacement
reval_every      = 50     # Re-evaluate the surfaces (sampling) every nth iteration, low values decrease performance
ref_rtol         = 1e-2   # The relative tolerance (rtol) below which a uniform refinement is applied
recons_every     = 1      # Reconstrain the 'update' normal constraint every nth iteration. Only works if 
save_iters_every = None   # Save iterations of the fit every nth iteration
constrainCont    = True   # Constrain the dsplacement as a result of the continuity correction (not recommended)

EBDM_ENDO = fit.EBDM(*multipatch_nurbs_endo_surface, datapoints=endocard_points, evalpts=100)
EBDM_EPI  = fit.EBDM(*multipatch_nurbs_epi_surface, datapoints=epicard_points, evalpts=100)

multipatch_nurbs_endo_surface_fit = EBDM_ENDO.run(max_iter=max_iter, max_refs=max_refs, atol=atol, rtol=rtol, ref_rtol=ref_rtol, 
                                                  displace_relax=displace_relax, 
                                                  continuity_relax=continuity_relax, continuity_iters_only=cont_iters_only, 
                                                  reval_every=reval_every, recons_every=recons_every, save_iters_every=save_iters_every,
                                                  constrainCont=constrainCont)

multipatch_nurbs_epi_surface_fit = EBDM_EPI.run(max_iter=max_iter, max_refs=max_refs, atol=atol, rtol=rtol, ref_rtol=ref_rtol, 
                                                displace_relax=displace_relax, 
                                                continuity_relax=continuity_relax, continuity_iters_only=cont_iters_only, 
                                                reval_every=reval_every, recons_every=recons_every, save_iters_every=save_iters_every,
                                                constrainCont=constrainCont)

error_info_endo = EBDM_ENDO.error_info() # Return the error info (displacement error, continuity error, when refinement was applied etc.), returns a dict
error_info_epi  = EBDM_EPI.error_info() # Return the error info (displacement error, continuity error, when refinement was applied etc.), returns a dict


## 6) Post-processing of the fitted surfaces
plotEndo = Plotting(multipatch_nurbs_endo_surface_fit, endocard_points, error_info=error_info_endo)
# plotEndo.surface(show_cerror=True)
plotEndo.error(relative=True, energy=True)

plotEpi = Plotting(multipatch_nurbs_epi_surface_fit, epicard_points, error_info=error_info_epi)
# plotEpi.surface(show_cerror=True)
plotEpi.error(relative=True, energy=True)


## 7) Convert the fitted surfaces to usable solid objects MultipatchSolid
multipatch_surface_endo_fitted = update_multipatch_object(multipatch_endo_surface, multipatch_nurbs_endo_surface_fit, surface_type="left-inner")
multipatch_surface_epi_fitted  = update_multipatch_object(multipatch_epi_surface, multipatch_nurbs_epi_surface_fit, surface_type="left-outer")
#multipatch_surface_fitted.save_vtk(os.path.join(directC, "output", "vtk", "Epicardium surface"))

# Possible translation and scaling of the shape (scales ventricle such that Vcav~= 44 [ml] and Vwall~=136 [ml])
# mpsurf_endo_copy = multipatch_surface_endo_fitted.copy()
# mpsurf_epi_copy  = multipatch_surface_epi_fitted.copy()
# mpsurf_endo_copy.translate(-base_plane["point"])
# mpsurf_epi_copy.translate(-base_plane["point"])
# mpsurf_endo_copy.scale(0.71, direction='xyz')
# mpsurf_epi_copy.scale(0.91, direction='xyz')
# solid = MultipatchSolid(mpsurf_epi_copy, mpsurf_endo_copy, tnelems=(2**max_refs,))
# solid.get_volume()
# solid.save_vtk(os.path.join(directC, "output", "vtk", "Ventricle fit solid (scaled z-base)"), boundary_only=False, nrsamples=10)#, boundary=True)
# solid.save_txt(os.path.join(directC, "output", "txt", "Ventricle fit solid (scaled z-base)"), boundary=solid.boundary_names())


solid = MultipatchSolid(multipatch_surface_epi_fitted, multipatch_surface_endo_fitted, tnelems=(2**max_refs,))
solid.get_volume()
# solid.save_vtk(os.path.join(directC, "output", "vtk", "Ventricle fit solid"), boundary_only=False, nrsamples=10)#, boundary=True)
# solid.save_txt(os.path.join(directC, "output", "txt", "Ventricle fit solid"), boundary=solid.boundary_names())
