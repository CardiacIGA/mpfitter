from pydicom import dcmread
import os
from ebdm.segmentation import Segmentation, Echo, Generate3Dpointcloud
from vtnurbs.construct import MultipatchSurface, MultipatchSolid
from ebdm.toolbox import init_multipatch_surfaces, update_multipatch_object
from ebdm.LV_utils import position, plane_from_points
import ebdm.EBDM_3D as fit
from ebdm.postprocess import Plotting
import numpy as np

##--------------------------------------------------##
## 1) Read the DICOM files and perform segmentation ##
##--------------------------------------------------##
# Define directories
directC   = os.path.realpath(os.path.dirname(__file__)) # Current directory
directG=os.path.split(os.path.split(directC)[0])[0]     # Global directory of ebdm module
dirdicom  = os.path.join(directC,  "data")              # Dicom file folder
outputDir = os.path.join(directC,  "output")            # General output folder
outputFig = os.path.join(outputDir,"images")            # Image folder
outputDat = os.path.join(outputDir,"datapoints")        # Datapoints folder
outputVTK = os.path.join(outputDir,"vtk")               # VTK results folder
outputVol = os.path.join(outputDir,"volumes")           # Volumes results folder
outputGEOM = os.path.join(outputDir,"geometry")         # Geometry results folder
fileEndoTempl = os.path.join(directG,"template","left ventricle","Endocardium_GEOM_DATA.txt") 
fileEpiTempl  = os.path.join(directG,"template","left ventricle","Epicardium_GEOM_DATA.txt")

# Define files to be loaded
#files     = "Echo_AP2CH_fnr30","Echo_AP4CH_fnr34","Echo_PSAX_fnr42" # DICOM file names
files     = "Echo_AP2CH_fnr58","Echo_AP4CH_fnr54","Echo_PSAX_fnr65" # DICOM file names
loaddata2D  = False # Load existing 2D segmentation data 
resegment2D = False #"PSAX" # Specify the name of the view you want to resegment
loaddata3D  = False

# Some standard orientation of the views (based on clinical protocol)
orientation = {"AP2CH": (  0,  0,    0),
               "AP4CH": (  0,  0,  120),
               "PSAX" : ( 90,  0,    0)} # Orientation along the principal axes: (x,y,z)


# initialize segmentation object and other lists
segment  = Segmentation()
echos    = []
views    = [file.split("_")[1] for file in files]                  # View names of each DICOM file
framenrs = [int(file.split("_")[2][3:]) for file in files]         # Frame numbers to be segmented of each DICOM file


if loaddata2D:
    datapoints_loaded = segment.load_txt("Datapoints 2D", outputDir=outputDat)


for ifile, iview, iframenr in zip(files,views,framenrs):

    # Store the DICOM files incl. relevant data into suitable class object "Echo()"
    echo = Echo()
    echo.dicom    = dcmread(os.path.join(dirdicom,ifile+".dcm"))
    echo.filename = ifile
    echo.framenr  = iframenr
    echo.view     = iview
    echo.orientation = orientation[iview]

    if loaddata2D:
        echo.datapoints2D["endocard"] = datapoints_loaded[f"View {echo.view} of endocard"]
        echo.datapoints2D["epicard"]  = datapoints_loaded[f"View {echo.view} of epicard"]
    echos += [echo]


# Check if we need to perform a manual segmentation or reload exisitng 2D data
if not loaddata2D or resegment2D:
    if resegment2D:
       echoResegment = [echo for echo in echos if echo.view==resegment2D] 
       points  = segment.manual(*echoResegment, returnPoints=True, saveFig=True, outputDir=outputFig)
    else:
       points  = segment.manual(*echos, returnPoints=True, saveFig=True, outputDir=outputFig) # Segment the DICOM files manually
    segment.save_to_txt(*echos, filename="Datapoints 2D", outputDir=outputDat) # save results to txt-file

# Initialize the pointcloud Object
pointcloud = Generate3Dpointcloud()
if not loaddata3D:
    
    # Interpolate the 2D segmented data
    pointcloud.interpolate(*echos)
    #pointcloud.visualize2D(*echos, save=False)

    # Combine the echo slices (optimal fit)
    pointcloud.combine_echos(*echos)
    pointcloud.visualize2D(*echos, save=False)
    pointcloud.visualize3D(*echos, save=False)
    pointcloud3D = pointcloud.get_3Dpoints(*echos, keepSlices=False) # Return the 3D point cloud

    # Save 3D datapoints to txt file
    pointcloud.save_to_txt(*echos, filename="Datapoints 3D", outputDir=outputDat)





##-------------------------------------------------------------##
## 2) Fit the 3D left-ventricle template to the 3D point-cloud ##
##-------------------------------------------------------------##
pointcloud3D, baseplane   = pointcloud.load_txt("Datapoints 3D", inputDir=outputDat, cutBase=True, combineViews=True) # Returns dict
# np.savetxt(os.path.join(outputDat,"Endocard points.txt"), pointcloud3D["endocard"], header='x,y,z', delimiter=',', comments='')
# np.savetxt(os.path.join(outputDat,"Epicard points.txt"), pointcloud3D["epicard"], header='x,y,z', delimiter=',', comments='')

# ## 2)a Load the NURBS left ventricle template
EndoTemplate = MultipatchSurface.load_txt(fileEndoTempl, returnDict=True) # Endocardium
EpiTemplate  = MultipatchSurface.load_txt(fileEpiTempl,  returnDict=True) # Epicardium
# set surface boundary names
boundary_names = {"base" : [(0,'bottom'), (1,'bottom'), (2,'bottom'),(3,'bottom')]}

multipatch_endo_surface = MultipatchSurface("MultiPatchSurface", "endocardium", EndoTemplate["patch vertices"],
                                            EndoTemplate["patch connectivity"], EndoTemplate["control points"],
                                            EndoTemplate["weights"], EndoTemplate["nelems"], {None : (1,1)}, bnames=boundary_names)

multipatch_epi_surface = MultipatchSurface("MultiPatchSurface", "epicardium",  EpiTemplate["patch vertices"],
                                            EpiTemplate["patch connectivity"], EpiTemplate["control points"],
                                            EpiTemplate["weights"], EpiTemplate["nelems"], {None : (1,1)}, bnames=boundary_names)

# Rotate such that datapoints do not line-up with patch interface
multipatch_endo_surface.rotate(10)
multipatch_epi_surface.rotate(10)

# Perform a first rough fit on the datapoints
multipatch_endo_surface = position(pointcloud3D["endocard"], baseplane, multipatch_endo_surface)
multipatch_epi_surface  = position(pointcloud3D["epicard"],  baseplane, multipatch_epi_surface)


# ## 4) Initialize and convert the MultipatchSurface to usable SpliPy objects
pverts_endo = multipatch_endo_surface.patchverts()
pverts_epi  = multipatch_epi_surface.patchverts()

plane_endo_interf1 = plane_from_points(pverts_endo[0], pverts_endo[1], pverts_endo[4])
plane_endo_interf2 = plane_from_points(pverts_endo[2], pverts_endo[3], pverts_endo[6])
plane_epi_interf1  = plane_from_points(pverts_epi[0], pverts_epi[1], pverts_epi[4])
plane_epi_interf2  = plane_from_points(pverts_epi[2], pverts_epi[3], pverts_epi[6])

constraints_epi  = {"base"  : ("plane", baseplane), None : ("normal", "fixed"),}
                    #"patch0-patch3":("plane", plane_epi_interf1),  "patch1-patch2":("plane", plane_epi_interf1),
                    #"patch0-patch1":("plane", plane_epi_interf2),  "patch2-patch3":("plane", plane_epi_interf2)}
constraints_endo = {"base"  : ("plane", baseplane), None : ("normal", "fixed"),}
                    #"patch0-patch3":("plane", plane_endo_interf1), "patch1-patch2":("plane", plane_endo_interf1),
                    #"patch0-patch1":("plane", plane_endo_interf2), "patch2-patch3":("plane", plane_endo_interf2),}

multipatch_nurbs_endo_surface = init_multipatch_surfaces(multipatch_endo_surface, constraints=constraints_endo) # Epicardium
multipatch_nurbs_epi_surface  = init_multipatch_surfaces(multipatch_epi_surface,  constraints=constraints_epi) # Epicardium


# ## 5) Specify the EBDM input parameters and run the fitting
max_iter         = 300    # Amount of iterations the EBDM runs per refinement
atol             = 1e-10  # Absolute tolerance, has the same unit of the datapoints
rtol             = 1e-9   # Relative tolerance, has no unit, is scaled with first displacement error
max_refs         = 2      # state how many refinements can be done on the surfaces
continuity_relax = 1      # Continuity relaxation factor (interpatch continuity)
displace_relax   = 0.1    # Displacement (over)relaxation factor
cont_iters_only  = 50     # Number of iterations that are used prior to reaching max_iter where only continuity is adjusted, no displacement
reval_every      = 10     # Re-evaluate the surfaces (sampling) every nth iteration, low values decrease performance
ref_rtol         = 1e-3   # The relative tolerance (rtol) below which a uniform refinement is applied
recons_every     = 1      # Reconstrain the 'update' normal constraint every nth iteration. Only works if 
save_iters_every = None   # Save iterations of the fit every nth iteration
constrainCont    = True   # Constrain the dsplacement as a result of the continuity correction (not recommended)

EBDM_ENDO = fit.EBDM(*multipatch_nurbs_endo_surface, datapoints=pointcloud3D["endocard"], evalpts=100)
EBDM_EPI  = fit.EBDM(*multipatch_nurbs_epi_surface, datapoints=pointcloud3D["epicard"], evalpts=100)

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
# Do some scaling of the error


## 6) Post-processing of the fitted surfaces
plotEndo = Plotting(multipatch_nurbs_endo_surface_fit, pointcloud3D["endocard"], error_info=error_info_endo)
# # plotEndo.surface(show_cerror=True)
plotEndo.error(relative=True, energy=True)

plotEpi = Plotting(multipatch_nurbs_epi_surface_fit, pointcloud3D["epicard"], error_info=error_info_epi)
# plotEpi.surface(show_cerror=True)
plotEpi.error(relative=True, energy=True)

## 7) Convert the fitted surfaces to usable solid objects MultipatchSolid
multipatch_surface_endo_fitted = update_multipatch_object(multipatch_endo_surface, multipatch_nurbs_endo_surface_fit, surface_type="left-inner")
multipatch_surface_epi_fitted  = update_multipatch_object(multipatch_epi_surface, multipatch_nurbs_epi_surface_fit, surface_type="left-outer")
#multipatch_surface_fitted.save_vtk(os.path.join(directC, "output", "vtk", "Epicardium surface"))

solid = MultipatchSolid(multipatch_surface_epi_fitted, multipatch_surface_endo_fitted, tnelems=(2**max_refs,))
#solid.save_vtk(os.path.join(outputVTK, "Ventricle fit solid"), boundary_only=True, nrsamples=10)
# V_inner, V_wall = solid.get_volume(output = True, wall = True)
#np.savetxt(path + '\\Volumes.txt', Volumes)
##-------------------##
## 3) Postprocessing ##
##-------------------##
#
# V_inner, V_wall  = solid.get_volume(output = True, wall = True)
#     Volumes = np.array([V_inner, V_wall])
#     np.savetxt(path + '\\Volumes.txt', Volumes)