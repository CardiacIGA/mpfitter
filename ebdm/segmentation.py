## This script shows is used to segment echocardiogram data
import numpy as np
import matplotlib.pyplot as plt
from geomdl import fitting 
from scipy.spatial.transform import Rotation as R
import treelog, os, copy
from numba import njit
_=np.newaxis
# Define colorscheme to be used in all plots
colorscheme = {'orange' : '#EE7733',
                'blue'   : '#0077BB',
                'teal'   : '#009988',
                'red'    : '#CC3311',
                'grey'   : '#BBBBBB',
                'black'  : '#000000',
                'cyan'   : '#33BBEE',
                'magenta': '#EE3377'}

class Echo:
    '''
    Echo object that stored all the information of a specific dicom file incl. its segmentation
    
    '''

    def __init__(self,):
        self.name     = "Echocardiogram struct"
        self.view     = None    # str: View
        self.framenr  = 0       # int: Frame number that is of interest (to be segmented)
        self.filename = None    # str. File name of the DICOM file
        self.dicom    = None    # dcmread Object: Loaded/Read DICOM file (with dcmread) associated with the echo
        self.orientation  = 0   # float:          Orientation of the 2D data set in 3D space
        #self.rightventr2D = np.array([0,0])   # Numpy array: Coordinate of the right ventricle relative to the 2D data set
        #self.rightventr3D = np.array([0,0,0]) # Numpy array: Coordinate of the right ventricle relative to the 3D data set
        self.datapoints2D = {}   # dict:           Dict containing segmented data points in 2D for epi or endo (keys)
        self.datapoints3D = {}   # dict:           Dict containing segmented data points in 2D for epi or endo (keys)
        self.Bspline2D    = {}   # dict:           Dict containing the 2D B-spline curve fitted on data points  for epi or endo (keys)
        self.Bspline3D    = {}   # dict:           Dict containing the 3D B-spline curve fitted on data points  for epi or endo (keys)
        return
    
    def copy(self,):
        return copy.deepcopy(self)

    def viewlong(self):
        if self.view == "PSAX":
            view = "Parasternal long axis (PSAX)"
        elif self.view == "AP2CH":
            view = "Apical 2 chamber (AP2CH)"
        elif self.view == "AP4CH":
            view = "Apical 4 chamber (AP3CH)"
        # elif self.view == "AP3CH":
        #     view = "Apical 3 chamber (AP4CH)"
        # elif self.view == "AP5CH":
        #     view = "Apical 5 chamber (AP5CH)"
        elif self.view != None:
            raise ValueError(f"Unsupported specified view {self.view}")
        return view
    
    def __repr__(self) -> str:
        return f"EchoClass: {self.view}"
        
    def __str__(self) -> str:
        return f"Echocardiogram data structure of the {self.viewlong()} view"




class Segmentation():
    
    def __init__(self):
        return
    
    def manual(self, *echos : Echo, returnPoints : bool =False, saveFig : bool = False, outputDir : str = "output") -> dict: # perform manual segmentation
        dp = {}
        for echo in echos:
            ds = echo.dicom
            dx = ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaX/100 # Convert [cm] to [m]
            dy = ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaY/100

            img = ds.pixel_array[echo.framenr,:,:,:]
            img_gray = self.rgb2gray(img)    

            plt.imshow(img_gray, cmap="gray", aspect='auto')
            plt.title(f"Segment the endocardium ({echo.view})")
            coordinates_input_in = np.array(plt.ginput(-1, timeout=0))
            dp[echo.view + '_endocard'] = coordinates_input_in*np.array([dx,dy])
            dp[echo.view + '_endocard'][:,0]*= -1 # Mirror the x-values because the y-values are already mirrored (when viewing images the pixel counts form top-left to bottom-left)
            plt.close()

            plt.imshow(img_gray, cmap="gray")
            plt.title("Segment the epicardium")
            plt.scatter(coordinates_input_in[:,0],coordinates_input_in[:,1], facecolors='none', edgecolors=colorscheme["cyan"])
            coordinates_input_out = np.array(plt.ginput(-1, timeout=0))
            dp[echo.view + '_epicard'] = coordinates_input_out*np.array([dx,dy])
            dp[echo.view + '_epicard'][:,0]*= -1 # Mirror the x-values because the y-values are already mirrored (when viewing images the pixel counts form top-left to bottom-left)
            plt.close()

            plt.imshow(img_gray, cmap="gray")
            plt.scatter(coordinates_input_in[:,0],coordinates_input_in[:,1], facecolors='none', edgecolors=colorscheme["cyan"])
            plt.scatter(coordinates_input_out[:,0],coordinates_input_out[:,1], facecolors='none', edgecolors=colorscheme["magenta"])
            if saveFig:
                plt.savefig(os.path.join(outputDir,f"segmentation_{echo.view}"))
            plt.close()
            echo.datapoints2D = {"epicard": dp[echo.view + '_epicard'], "endocard": dp[echo.view + '_endocard']}

        self.points = dp

        if returnPoints:
            return dp
        
    @staticmethod
    def rgb2gray(rgb):# applies the formula to convert RBG into brightness
        return np.dot(rgb, [0.2989, 0.5870, 0.1140])

    @staticmethod
    def save_to_txt(*echos : Echo, filename : str ="Datapoints 2D", outputDir : str = "output"):
        filepath = os.path.join(outputDir,filename+".txt")
        with open(filepath, 'w') as f:
            f.write("Data points extracted from 2D echocardiogram data"+"\n\n")
            for echo in echos:
                for key, data in echo.datapoints2D.items():
                    view = echo.view
                    surf = key
                    f.write(f"View {view} of {surf}\n")
                    if type(data) == np.ndarray:
                        lines = "\n".join( [ str(row.tolist()) for row in data ] )
                    f.write(lines) 
                    f.write("\n\n")     
        return
    
    def save_to_pickle(self, filename : str ="Datapoints 2D", outputDir : str = "output"):
        ''' Save a new 'filename.pickle' file.  '''        
        import pickle
        with open(outputDir + filename + '.pickle', 'wb') as f: 
            pickle.dump(self.points,f) 
        return
    
    ## TODO
    def save_to_json(self, filename : str ="Datapoints 2D"):
        raise NotImplementedError

    @staticmethod
    def load_txt(filename : str ="Datapoints 2D", outputDir : str = "output"):
        filepath   = os.path.join(outputDir,filename+".txt")
        datapoints = {"epicard": None, "endocard": None} # Initialize
        import ast

        with open(filepath, 'r') as f:
            lines = f.readlines()
        # Strip empty spaces and remove '\n'
        lines = [idat.strip("\n") for idat in lines if len(idat.strip("\n")) != 0]   
        catch_line = "View"
        idx = []   
        for i, line in enumerate(lines):
            if catch_line in line.strip(" "):
                idx += [i]
        idx += [None]
        for i,j in zip(idx[:-1],idx[1:]):
            datapoints[lines[i]] = np.array([ast.literal_eval(iline) for iline in lines[(i+1):j]])

        return datapoints




class Generate3Dpointcloud():
    # Class that interpolates the 2D segmented data points using splines and transforms it to a useable 3D (sparse) point cloud.

    def __init__(self,):
        return    
    
    def interpolate(self, *echos : Echo):
        ''' Function that interpolates the 2D data points by means of splines (fits a spline using EBDM) for each echo view.  '''
        # max_iter    = 15
        # min_error   = 1e-7
        # max_ref     = 10
        # sample_size = 200


        for echo in echos:
            for boundary, datapoints2D in echo.datapoints2D.items():

                if echo.view != "PSAX":

                    # check what the basal points are (make sure the first and last point of the array are located at the base):
                    idx = 1 if np.linalg.norm(np.diff(datapoints2D[[0,2],:],axis=0)) < np.linalg.norm(np.diff(datapoints2D[[0,1],:],axis=0)) else -1
                    if idx == 1:
                        echo.datapoints2D[1:,:] = np.roll(echo.datapoints2D[1:,:], -1, 0)    

                    # Specify fitting settings
                    degree = 3
                    curvefitted = fitting.interpolate_curve(datapoints2D.tolist(), degree)
                    treelog.info(f"{echo.view} view ({boundary}) fitted with B-spline")

                    # Set numpy array variants
                    curvefitted.ctrlptsNP = np.array(curvefitted.ctrlpts)
                    curvefitted.evalptsNP = np.array(curvefitted.evalpts)
                    echo.Bspline2D[boundary] = curvefitted #.evalptsnp

                    # datapoints3D = np.c_[datapoints2D, np.zeros(len(datapoints2D))] # Concatenate third dimension

                    # # Create the spline with beginning and end point based on the data points
                    # curve = BSpline.Curve()
                    # # check what the basal points are:
                    # idx = 1 if np.linalg.norm(np.diff(datapoints3D[[0,2],:],axis=0)) < np.linalg.norm(np.diff(datapoints3D[[0,1],:],axis=0)) else -1
                    # cps   = [list(datapoints3D[0,:]), list((datapoints3D[0,:] + datapoints3D[idx,:]) / 2 + np.array([-0.005, -0.005, 0])), list(datapoints3D[idx,:])]
                    # curve.degree        = 2
                    # curve.ctrlpts       = cps
                    # curve.knotvector    = [0,0,0,1,1,1]
                    # curve.sample_size   = sample_size
                    # curve.evaluate()
                    # curve.evalptsnp = np.array(curve.evalpts)

                    # Run the EBDM with the input parameters
                    #curvefitted = fit.EBDM(curve, datapoints3D, max_iter, min_error, max_ref)
                else:
                    # Specify fitting settings
                    degree  = 3
                    Xpseudo = datapoints2D[-1] + 0.98*(datapoints2D[0] - datapoints2D[-1])# define a pseudo point close to the first (closed curve)
                    datapoints2D_circ = np.r_[datapoints2D, Xpseudo[np.newaxis]]
                    
                    curvefitted = fitting.interpolate_curve(datapoints2D_circ.tolist(), degree)
                    treelog.info(f"{echo.view} view ({boundary}) fitted with B-spline")
                    
                    curvefitted.ctrlpts[-1] = curvefitted.ctrlpts[0]# Fix the end-control point based on the pseudo coordinate to the first control point (closing the curve)
                    curvefitted.evalptsnp   = np.array(curvefitted.evalpts)  
                    curvefitted.ctrlptsNP   = np.array(curvefitted.ctrlpts)

                    echo.Bspline2D[boundary] = curvefitted #.evalptsnp

                    # datapoints3D = np.c_[datapoints2D, np.zeros(len(datapoints2D))]
                    # #datapoints3D[:, [1, 0]] = datapoints3D[:, [0, 1]]                       # Switch x-y columns to match orientation
                    # # Create the spline with beginning and end point based on the data points
                    # curve = BSpline.Curve()
                    # ymaxarg = np.argmax(datapoints3D[:,1])
                    # ymax = np.max(datapoints3D[:,1])
                    # ymin = np.min(datapoints3D[:,1])
                    # xmax = np.max(datapoints3D[:,0])
                    # xmin = np.min(datapoints3D[:,0])
                    
                    # cps = [list(datapoints3D[ymaxarg,:]), [xmax,ymax,0], [xmax,ymin,0], [xmin,ymin,0], [xmin,ymax,0], list(datapoints3D[ymaxarg,:])]
                    # curve.degree        = 2
                    # curve.ctrlpts       = cps
                    # curve.knotvector    = [0,0,0,1,2,3,4,4,4]
                    # curve.sample_size   = sample_size
                    # curve.evaluate()
                    # curve.evalptsnp = np.array(curve.evalpts)
                    
                    # # Run the EBDM with the input parameters
                    # max_iter    = 30
                    # min_error   = 1e-7
                    # max_ref     = 1
                    # 0
                    #curvefitted = fit.EBDM(curve, datapoints3D, max_iter, min_error, max_ref)
                    #conn = np.linspace(0,len(),)
                    #connection  = [[0,1], [1,2], [2,3], [3,0]]
                    #curvefitted = fit2D.EBDM(curve, datapoints3D, max_iter, min_error, max_ref)
                    #dp[file] = dp_view
           
        return

    def combine_echos(self, *echos : Echo, tilt_range : tuple = (-40,40), tilt_steps : int = 101, angles_AP4CH : dict = {}, angles_PSAX : dict = {}):

        # Initialize tilt tangles if not provided:
        if "tilt_x_min" not in angles_PSAX.keys(): angles_PSAX["tilt_x_min"]   = -20
        if "tilt_x_max" not in angles_PSAX.keys(): angles_PSAX["tilt_x_max"]   =   0
        if "tilt_x_nstp" not in angles_PSAX.keys(): angles_PSAX["tilt_x_nstp"] =  15
        if "tilt_y_min" not in angles_PSAX.keys(): angles_PSAX["tilt_y_min"]   = -20
        if "tilt_y_max" not in angles_PSAX.keys(): angles_PSAX["tilt_y_max"]   =  20
        if "tilt_y_nstp" not in angles_PSAX.keys(): angles_PSAX["tilt_y_nstp"] =  15
        if "rot_min"  not in angles_PSAX.keys(): angles_PSAX["rot_min"]     = -20 
        if "rot_max"  not in angles_PSAX.keys(): angles_PSAX["rot_max"]     =  20
        if "rot_nstp" not in angles_PSAX.keys(): angles_PSAX["rot_nstp"]    =  15
        if "trans_z_nstp" not in angles_PSAX.keys(): angles_PSAX["trans_z_nstp"] =  15

        if "tilt_x_min"  not in angles_AP4CH.keys(): angles_AP4CH["tilt_x_min"]  = -40
        if "tilt_x_max"  not in angles_AP4CH.keys(): angles_AP4CH["tilt_x_max"]  =  40
        if "tilt_x_nstp" not in angles_AP4CH.keys(): angles_AP4CH["tilt_x_nstp"] = 101

        # Transform echos to 3D space (initialize)
        for echo in echos:
 
            # 1) + 2) Rotate and translate 2D points so that they coincide with the basal plane/aortic valve
            # Only for AP views not PSAX
            if echo.view != "PSAX": 

                # 1) Rotate the data points slices in 2D space
                baseTangVec2D = ( np.mean(np.c_[echo.datapoints2D["endocard"][0,:], echo.datapoints2D["epicard"][0,:]], axis=1) -
                                  np.mean(np.c_[echo.datapoints2D["endocard"][-1,:], echo.datapoints2D["epicard"][-1,:]], axis=1) ) # Mean tangential vector allong the basal plane
                
                refvector2D = np.array([1,0])
                sign        = np.sign(baseTangVec2D[-1])
                angle       = np.arccos( np.dot( baseTangVec2D, refvector2D )/(np.linalg.norm(baseTangVec2D)*np.linalg.norm(refvector2D)) )
                RotMatrix   = R.from_euler('z', sign*angle, degrees=False).as_matrix()[:2,:2]
                for boundary, datapoints2D in echo.datapoints2D.items():
                    echo.datapoints2D[boundary] = np.dot(datapoints2D, RotMatrix) 
                
                baseZmean   =  np.mean(np.c_[echo.datapoints2D["endocard"][0,:] , echo.datapoints2D["epicard"][0,:]  , 
                                             echo.datapoints2D["endocard"][-1,:], echo.datapoints2D["epicard"][-1,:]], axis=1)[-1]

                # 2) Center the slices at (0,0,0), which is where the mitral valve is
                datapoints2DEndo = echo.datapoints2D["endocard"]
                centerX = datapoints2DEndo[0,0] + 0.5*(datapoints2DEndo[-1,0] - datapoints2DEndo[0,0]) #min(datapoints2DEndo[:,0]) + (max(datapoints2DEndo[:,0])-min(datapoints2DEndo[:,0]))/2
                center  = np.array([centerX,baseZmean])  # X-shift: center; Z-shift : Translate such that Base equals xy-plane

                for bound in echo.datapoints2D:
                    # Translate accordingly according to steps 1) and 2)
                    echo.datapoints2D[bound] -= center 

                    # Rotate and Translate Bspline similarly
                    echo.Bspline2D[bound].ctrlptsNP  = np.dot(echo.Bspline2D[bound].ctrlptsNP, RotMatrix)
                    echo.Bspline2D[bound].ctrlptsNP -= center 

            else: # Else center the PSAX view
                center = np.mean(echo.datapoints2D["endocard"],axis=0)
                for bound in echo.datapoints2D:
                    # Translate accordingly according to steps 1) and 2)
                    echo.datapoints2D[bound] -= center

                    # Translate Bspline similarly
                    echo.Bspline2D[bound].ctrlptsNP -= center 





            # 3) Transform 2D points to 3D and apply correct rotation to it    
            for boundary, datapoints2D in echo.datapoints2D.items():

                orientation = np.array(echo.orientation)    
                if echo.view != "PSAX":
                    orientation += np.array([0,0,180]) # Add 180 degrees because echo views are always flipped
                else:
                    orientation += np.array([180,0,0])

                # Reorient datapoints based on specified angles
                echo.datapoints3D[boundary] = self.transform(orientation, datapoints2D)

                # Build the 3D spline representation identically 
                echo.Bspline3D[boundary] = copy.deepcopy(echo.Bspline2D[boundary])
                echo.Bspline3D[boundary].ctrlptsNP = self.transform(orientation, echo.Bspline2D[boundary].ctrlptsNP)

                # Set the actual ctrlpts again
                echo.Bspline2D[boundary].ctrlpts = echo.Bspline2D[boundary].ctrlptsNP.tolist() 
                echo.Bspline3D[boundary].ctrlpts = echo.Bspline3D[boundary].ctrlptsNP.tolist() 
        


        # 4) Perform tilt and rocking motion for AP4CH view
        #angles    = np.linspace(tilt_range[0],tilt_range[1],tilt_steps)
        angles    = np.linspace(angles_AP4CH["tilt_x_min"], angles_AP4CH["tilt_x_max"], angles_AP4CH["tilt_x_nstp"])
        echoAP4CH = [echo for echo in echos if echo.view=="AP4CH"][0]
        echoAP2CH = [echo for echo in echos if echo.view=="AP2CH"][0]
        echoAP4CH = self.tilt_AP4CH_echo(echoAP4CH, angles, echoAP2CH) # Rotate AP4CH and compare with AP2CH


        # 5) Fit the PSAX by tilting, planar translation, vertical translation, rotation
        tilt_angles_x  = np.linspace(angles_PSAX["tilt_x_min"], angles_PSAX["tilt_x_max"], angles_PSAX["tilt_x_nstp"]) #np.linspace(-20,0,15) # Angle range for tilting the PSAX view
        tilt_angles_y  = np.linspace(angles_PSAX["tilt_y_min"], angles_PSAX["tilt_y_max"], angles_PSAX["tilt_y_nstp"]) #np.linspace(-20,20,15) # Angle range for tilting the PSAX view
        rot_angles     = np.linspace(angles_PSAX["rot_min"], angles_PSAX["rot_max"], angles_PSAX["rot_nstp"]) #np.linspace(-20,20,15) # Angle range for rotating the PSAX view around its normal axis
        transl_steps   = np.linspace(0,np.min(np.array(echoAP2CH.Bspline3D["endocard"].evalpts)),angles_PSAX["trans_z_nstp"]) # vertical displacement/translation steps of the PSAX view

        echoPSAX  = [echo for echo in echos if echo.view=="PSAX"][0]    
        echoPSAX  = self.fit_PSAX_echo(echoPSAX, echoAP2CH, echoAP4CH, tilt_angles_x, tilt_angles_y, rot_angles, transl_steps)

        self.points3D = {}
        for echo in [echoAP2CH, echoAP4CH, echoPSAX]:
            for bound in ["endocard","epicard"]:
                self.points3D[f"{echo.view}_{bound}"] = echo.datapoints3D[bound]
            

        return
    
    def fit_PSAX_echo(self, echoPSAX, echoAP2CH, echoAP4CH, tilt_angles_x, tilt_angles_y, rot_angles, transl_steps):
        PSAXnormali = np.cross(echoPSAX.datapoints3D["endocard"][0], echoPSAX.datapoints3D["endocard"][1]) #np.array([0,0,1])
        PSAXnormali /= np.linalg.norm(PSAXnormali)

        # Resample Bspline to make it more efficient
        nsamples = 40
        echoPSAX,  sampleSizePSAX  = self.resample_Bspline(echoPSAX, nrsample=nsamples)
        echoAP2CH, sampleSizeAP2CH = self.resample_Bspline(echoAP2CH, nrsample=nsamples)
        echoAP4CH, sampleSizeAP4CH = self.resample_Bspline(echoAP4CH, nrsample=nsamples)

        errors = np.zeros((len(tilt_angles_x), len(tilt_angles_y),len(rot_angles),len(transl_steps))) # Boundary, tilt angle, rotation angle, translation step
        dvec   = errors.copy() # Displacement vectors for centering/shifting the view

        # Loop over each degree of freedom: Tilt in x, tile in y, rotate around normal, and translate along z-axis.
        #for b, bound in enumerate(echoPSAX.Bspline3D): 
        # We do not use a for-loop because the centering should be done on both rings simultaneously 
        points3Di_endo = np.array(echoPSAX.Bspline3D["endocard"].evalpts)
        points3Di_epi  = np.array(echoPSAX.Bspline3D["epicard"].evalpts)

        for x, TangleX in enumerate(tilt_angles_x):
            RotM = R.from_euler('xyz', (TangleX,0,0), degrees=True).as_matrix()

            PSAXnormal_x    = np.dot(PSAXnormali, RotM) 
            points3D_Tilt_endo_x = np.dot(points3Di_endo, RotM) # Tile allong the x-axis (direction of view)
            points3D_Tilt_epi_x  = np.dot(points3Di_epi, RotM)

            for y, TangleY in enumerate(tilt_angles_y):
                RotM = R.from_euler('xyz', (0,TangleY,0), degrees=True).as_matrix()

                PSAXnormal    = np.dot(PSAXnormal_x, RotM) 
                points3D_Tilt_endo = np.dot(points3D_Tilt_endo_x, RotM) # Tile allong the x-axis (direction of view)
                points3D_Tilt_epi  = np.dot(points3D_Tilt_epi_x, RotM)

                for j, Rangle in enumerate(rot_angles):

                    points3D_Rot_endo = self.rotate_around_vec(points3D_Tilt_endo, PSAXnormal, Rangle, degree=True) # Rotate around normal axis
                    points3D_Rot_epi  = self.rotate_around_vec(points3D_Tilt_epi, PSAXnormal, Rangle, degree=True) # Rotate around normal axis


                    for k, Tstep in enumerate(transl_steps):

                        points3D_Trans_endo = points3D_Rot_endo + np.array([0,0,Tstep]) # Shift/translate allong vertical axis
                        points3D_Trans_epi  = points3D_Rot_epi  + np.array([0,0,Tstep]) 
                        points3D_PSAX  = {"endocard": points3D_Trans_endo, "epicard": points3D_Trans_epi}
                        points3D_AP2CH = {}
                        points3D_AP4CH = {}
                        for bound in echoPSAX.Bspline3D: 
                             
                            points3D_ap2ch = np.array(echoAP2CH.Bspline3D[bound].evalpts)
                            points3D_ap4ch = np.array(echoAP4CH.Bspline3D[bound].evalpts)
                            idxhalfAP2CH = int(len(points3D_ap2ch)/2)# Index of half the points
                            idxhalfAP4CH = int(len(points3D_ap4ch)/2)

                            points3D_AP2CH[bound] = ( points3D_ap2ch[:idxhalfAP2CH], points3D_ap2ch[idxhalfAP2CH:] )
                            points3D_AP4CH[bound] = ( points3D_ap4ch[:idxhalfAP4CH], points3D_ap4ch[idxhalfAP4CH:] )

                        dv = np.array([0,0,0])
                        for nr in range(5):
                            points3D_PSAX_C = points3D_PSAX.copy()
                            points3D_PSAX_C["endocard"] += dv
                            points3D_PSAX_C["epicard"]  += dv
                            dv = self.get_center_dispvec(points3D_PSAX_C, (points3D_AP2CH,points3D_AP4CH), return_error=True)
                        dvec[x,y,j,k] = dv

                        for bound in echoPSAX.Bspline3D:
                            error_AP2CH_half1, ids = self.compute_error(points3D_PSAX[bound] + dvec[x,y,j,k], points3D_AP2CH[bound][0], efficient=False, returnIDs=True)
                            error_AP2CH_half2, ids = self.compute_error(points3D_PSAX[bound] + dvec[x,y,j,k], points3D_AP2CH[bound][1], efficient=False, returnIDs=True)
                            error_AP4CH_half1, ids = self.compute_error(points3D_PSAX[bound] + dvec[x,y,j,k], points3D_AP4CH[bound][0], efficient=False, returnIDs=True)
                            error_AP4CH_half2, ids = self.compute_error(points3D_PSAX[bound] + dvec[x,y,j,k], points3D_AP4CH[bound][1], efficient=False, returnIDs=True)
  
                            errors[x,y,j,k] += np.linalg.norm([error_AP2CH_half1, error_AP2CH_half2, 
                                                               error_AP4CH_half1, error_AP4CH_half2])


        # Find optimal value/index and print out the results    
        idx = np.unravel_index( np.argmin(np.linalg.norm(errors,axis=0)), (len(tilt_angles_x), len(tilt_angles_y),len(rot_angles),len(transl_steps))) 
        treelog.info(f"PSAX view minimum at:\n\
                        -Tilt angle x: {tilt_angles_x[idx[0]]:.2f}  [degr]\n\
                        -Tilt angle y: {tilt_angles_y[idx[1]]:.2f}  [degr]\n\
                        -Rot angle:    {rot_angles[idx[2]]:.2f}   [degr]\n\
                        -Transl step:  {transl_steps[idx[3]]*1e2:.2f} [cm]")
        
        # Use the optimal set of values to tilt, rotate and translate the PSAX view
        for bound, datapoints3D in echoPSAX.datapoints3D.items():

            # Apply tilt rotation
            Rot_tilt_x   = R.from_euler('xyz', (tilt_angles_x[idx[0]],0,0), degrees=True).as_matrix()
            Rot_tilt_y   = R.from_euler('xyz', (0,tilt_angles_x[idx[1]],0), degrees=True).as_matrix()
            Rot_tilt   = np.dot(Rot_tilt_x,Rot_tilt_y)
            PSAXnormal = np.dot(PSAXnormali, Rot_tilt) 

            echoPSAX.datapoints3D[bound]        = np.dot(datapoints3D, Rot_tilt)
            echoPSAX.Bspline3D[bound].ctrlptsNP = np.dot(echoPSAX.Bspline3D[bound].ctrlptsNP, Rot_tilt)
            
            # Apply rotation about axis rotation
            echoPSAX.datapoints3D[bound]        = self.rotate_around_vec(echoPSAX.datapoints3D[bound],        PSAXnormal, rot_angles[idx[2]], degree=True)
            echoPSAX.Bspline3D[bound].ctrlptsNP = self.rotate_around_vec(echoPSAX.Bspline3D[bound].ctrlptsNP, PSAXnormal, rot_angles[idx[2]], degree=True)
            
            # Apply translation
            echoPSAX.datapoints3D[bound]        += np.array([0,0,transl_steps[idx[3]]]) + dvec[idx]
            echoPSAX.Bspline3D[bound].ctrlptsNP += np.array([0,0,transl_steps[idx[3]]]) + dvec[idx]
            
            # Reset the ctrlpts to the new ones
            echoPSAX.Bspline3D[bound].ctrlpts   = echoPSAX.Bspline3D[bound].ctrlptsNP.tolist()
        
        # Resample back to original sample size
        echoPSAX,  s = self.resample_Bspline(echoPSAX, nrsample=sampleSizePSAX)
        echoAP2CH, s = self.resample_Bspline(echoAP2CH, nrsample=sampleSizeAP2CH)
        echoAP4CH, s = self.resample_Bspline(echoAP4CH, nrsample=sampleSizeAP4CH)
        
        return echoPSAX

    def tilt_AP4CH_echo(self, echo, angles, refecho): # Tilt a specific echo around its basal tangential vector given a set of angles
        baseTangVec3D = ( np.mean(np.c_[echo.datapoints3D["endocard"][0,:] ,  echo.datapoints3D["epicard"][0,:]], axis=1) -
                          np.mean(np.c_[echo.datapoints3D["endocard"][-1,:], echo.datapoints3D["epicard"][-1,:]], axis=1) ) # Mean tangential vector allong the basal plane
        baseTangVec3D /= np.linalg.norm(baseTangVec3D)

        errors = np.empty((2,len(angles)))
        for i, bound in enumerate(echo.datapoints3D):
            points3D    = np.array(echo.Bspline3D[bound].evalpts)
            refpoints3D = np.array(refecho.Bspline3D[bound].evalpts)
            for j, angle in enumerate(angles):
                points3D_tilt = self.rotate_around_vec(points3D, baseTangVec3D, angle, degree=True)
                errors[i,j], IDs = self.compute_error(points3D_tilt, refpoints3D, efficient=False, returnIDs=True)
        
        idx = np.argmin(np.linalg.norm(errors,axis=0))
        treelog.info(f"AP4CH view minimum tilt angle found at {angles[idx]:.2f} [degr]")
        for bound, datapoints3D in echo.datapoints3D.items():
            echo.datapoints3D[bound] = self.rotate_around_vec(datapoints3D, baseTangVec3D, angles[idx], degree=True)
            echo.Bspline3D[bound].ctrlptsNP = self.rotate_around_vec(echo.Bspline3D[bound].ctrlptsNP, baseTangVec3D, angles[idx], degree=True)
            echo.Bspline3D[bound].ctrlpts   = echo.Bspline3D[bound].ctrlptsNP.tolist()
        
        return echo
    
    # Center the given dictionary containing endo and epicard echo datapoints based on a set/tuple of other reference echos
    def get_center_dispvec(self, datapoints : dict, RefDatapoints : dict, return_error : bool = False):
        dvecList  = []
        #errorList = []
        center    = np.mean([datp for b, datp in datapoints.items()])
        normal    = np.cross(datapoints["endocard"][0], datapoints["endocard"][1])
        normal   /= np.linalg.norm(normal)

        for bound, dataP in datapoints.items():
            vec = np.array([0.,0.,0.])
            err = []
            for refData in RefDatapoints:
                #vec_half = np.array([0.,0.,0.])
                for halfdata in refData[bound]:
                    error, IDs = self.compute_error(dataP, halfdata, efficient=False, returnIDs=True)
                    vector  = (halfdata[IDs[1]] - dataP[IDs[0]])
                    vec    -= 0.1*(  vector - np.dot(vector,normal) ) # Only the in plane vector is of importance! 
                    #err.append(np.linalg.norm(error))
                

            dvecList.append(vec)  
            #errorList.append(np.linalg.norm(err))   
        dvec  = np.mean(dvecList)
        #Error = np.linalg.norm(errorList)

        # # Add displacement to average center
        # for bound in datapoints:
        #     datapoints[bound] += dvec

        return dvec    
        # if return_error:
        #     return center + dvec, Error
        # else:
        #     return center + dvec

    # Resample a Bspline based on Echo object input
    def resample_Bspline(self, echo : Echo, nrsample : int = 80):
        for bound, Bspline in echo.Bspline3D.items():
            sample_size_init = Bspline.sample_size # initial sample size
            echo.Bspline3D[bound].sample_size = nrsample # Reset the sample size
        return echo, sample_size_init


    @staticmethod    
    @njit()  
    def compute_error(datapoints1, datapoints2, efficient=True, returnIDs=False): # The error equals the nearest distance. Efficient = True does not loop over entire array
        # if efficient:
        #     idx = [ int(len(datapoints1)/3), 2*int(len(datapoints1)/3) ]
        # else:
        #     idx = [0,-1]

        # if efficient:
        #     idx  = [ 0, int(len(datapoints1)/3), 2*int(len(datapoints1)/3) ]
        #     norm = np.zeros(len(idx))
        #     for i in idx:
        #         norm[i] = np.min(np.linalg.norm(datapoints1[i] - datapoints2 , axis = 1 ))
        #     idx.pop( np.argmax(norm) )    
        # else:
        #     idx = [0,-1]

        idx   = [0,-1]
        error = 1e3
        for i, datap1 in enumerate(datapoints1[idx[0]:idx[1]]):
            #norm  = np.linalg.norm( datapoints2 - datap1 , axis = 1 )
            dp    = datapoints2 - datap1
            norm  = np.sqrt( dp[:,0]**2 + dp[:,1]**2 + dp[:,2]**2 )
            argmin = np.argmin(norm) 
            if norm[argmin] < error:
                error = norm[argmin]  
                IDs   = (idx[0] + i, argmin)

        return error, IDs
        # if returnIDs:
        #     return error, IDs
        # else:  
        #     return error    
    
    def centerValue(self, dicom): # Return the x-center of the dicom view (tip of the triangular arc) in physical coordinates
        dx = dicom.SequenceOfUltrasoundRegions[0].PhysicalDeltaX/100 # Convert [cm] to [m]
        xmin_view  = dicom.SequenceOfUltrasoundRegions[0].RegionLocationMinX0
        xmax_view  = dicom.SequenceOfUltrasoundRegions[0].RegionLocationMaxX1
        centerXpxl = xmin_view + (xmax_view-xmin_view)/2  # X-pixel coordinate at which the center axis of the dicom view arc is located
        return centerXpxl*dx
    
    def transform(self, orientation : tuple, datapoints : np.ndarray, translate : np.ndarray = np.array([0,0,0])):
        RotMatrix    = R.from_euler('xyz', orientation, degrees=True).as_matrix() # Generate rotation matrix
        datapoints3D = np.c_[datapoints, np.zeros(len(datapoints))] # Convert 2D to 3D (assuming z=0) 
        datapoints3D[:, [2, 1]] = datapoints3D[:, [1, 2]] # Convert x-y values to x-z (assuming y=0)
        return np.dot(datapoints3D + translate, RotMatrix)  # Rotate and translate plane accordingly    

    def rotate_around_vec(self, datapoints, vec, angle, degree=False):
        if degree:
            angle *= np.pi/180
        assert datapoints.shape[1] == 3, "Rotating around vector is only supported for 3D point data"
        x, y, z = vec
        C = np.array([ [ 0, -z,  y],
                       [ z,  0, -x],
                       [-y,  x,  0] ])
        #Rot = np.identity(3) + C*np.sin(angle) + np.dot(C,C)*(1-np.cos(angle))
        Rot = np.identity(3) + C*np.sin(angle) + 2*(np.sin(0.5*angle)**2)*(np.cross(vec, vec) - np.identity(3))
        return np.dot(datapoints, Rot)
    
    def get_tranformation_matrix(self, points_in, points_out):
        dim = points_in.shape[1] # dimension
        C = np.zeros((dim,dim))
        if dim == 2:
            # Select 2 'random' points to build the transformation matrix
            Pin  = points_in[[0,-1],:]
            Pout = points_out[[0,-1],:]
            e1, e2 = np.identity(dim) # Define unit vectors
            e_unit = [e1,e2]
        elif dim == 3:
            # Select 3 'random' points to build the transformation matrix
            Pin  = points_in[[0,1,2],:]
            Pout = points_out[[0,1,2],:]
            

            # Check if the 3D points do not share a zero value -> otherwise solve for 2D and convert back to 3D
            dP = np.sum(abs(Pin), axis=0)
            for idx in range(3):
                if abs(dP[idx]) < 1e-12:
                    e1, e2 = np.identity(2) # Define unit vectors
                    e_unit = [e1,e2]
                    C = np.zeros((2,2))
                    Pin  = np.delete(Pin[:2],idx,1)
                    Pout = np.delete(Pout[:2],idx,1)
                    break
                else:
                    e1, e2, e3 = np.identity(dim) # Define unit vectors
                    e_unit = [e1,e2,e3]
    
        # If there is no difference between the points == no transformation    
        if np.linalg.norm(Pin-Pout) < 1e-12: 
            return np.identity(dim)

        #C = np.zeros((dim,dim))
        for i, e in enumerate(e_unit):
            c = np.linalg.solve(Pin.T, e)
            C[:,i] = c 
        T = np.dot(Pout.T,C)  
        Tref = T.copy()

        if dim != Pin.shape[1]: # The points shared a zero value
           Tref = np.identity(dim)
           if idx == 0:
               Tref[1:,1:] = T 
           elif idx == 1:
               Tref[0,0] =T[0,0]; Tref[0,2] =T[0,1]
               Tref[2,0] =T[1,0]; Tref[2,2] =T[1,1] 
           elif idx == 2:
               Tref[:2,:2] = T
        return Tref


    def visualize2D(self, *echos : Echo, save : bool = False):
        import matplotlib.pyplot as plt
        
        for echo in echos:
            Bspline2Dendo = np.array(echo.Bspline2D['endocard'].evalpts)
            Bspline2Depie = np.array(echo.Bspline2D['epicard'].evalpts)
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(121)
            ax.plot(echo.datapoints2D['endocard'][:,0], echo.datapoints2D['endocard'][:,1], marker='o', markersize=10, linestyle='None', color=colorscheme['blue'])
            ax.plot(np.array(echo.Bspline2D['endocard'].ctrlpts)[:,0], np.array(echo.Bspline2D['endocard'].ctrlpts)[:,1], marker='s', markersize=5, linestyle='dashed', color=colorscheme['black'])
            ax.plot(Bspline2Dendo[:,0], Bspline2Dendo[:,1], linestyle='solid', color=colorscheme['red'])
            ax.set_ylabel('$x_2$ coordinate [m]')
            ax.set_xlabel('$x_1$ coordinate [m]')
            ax.set_title('Endocardium segmentation')
            ax.grid()

            ax = fig.add_subplot(122)
            ax.plot(echo.datapoints2D['epicard'][:,0], echo.datapoints2D['epicard'][:,1], marker='o', markersize=10, linestyle='None', color=colorscheme['blue'])
            ax.plot(np.array(echo.Bspline2D['epicard'].ctrlpts)[:,0], np.array(echo.Bspline2D['epicard'].ctrlpts)[:,1], marker='s', markersize=5, linestyle='dashed', color=colorscheme['black'])
            ax.plot(Bspline2Depie[:,0], Bspline2Depie[:,1], linestyle='solid', color=colorscheme['red'])
            ax.set_ylabel('$x_2$ coordinate [m]')
            ax.set_xlabel('$x_1$ coordinate [m]')
            ax.set_title('Epicardium segmentation')
            ax.grid()
            plt.tight_layout()
            plt.show()
            if save:
                fig.savefig('results/Figures/CVrelation', dpi=600, bbox_inches='tight')
        return
    
    def visualize3D(self, *echos : Echo, save : bool = False):
        import matplotlib.pyplot as plt
        
        ax = plt.figure().add_subplot(projection='3d')

        for echo, color in zip(echos,colorscheme): 
            # Convert 2D Bsplines to 3D space
            for boundary, datapoints3D in echo.datapoints3D.items():
                #Bspline2Dpoints = np.array(echo.Bspline2D[boundary].evalpts)
                #centerX = min(Bspline2Dpoints[:,0]) + (max(Bspline2Dpoints[:,0])-min(Bspline2Dpoints[:,0]))/2
                #centerX = self.centerValue(echo.dicom)
                #Bspline3D = self.transform(echo.orientation, Bspline2Dpoints, translate=np.array([-centerX, 0, 0]))
                Bspline3Dpoints = np.array(echo.Bspline3D[boundary].evalpts)
                ax.scatter(datapoints3D[:,0], datapoints3D[:,1], datapoints3D[:,2], label=echo.view, color=colorscheme[color])
                ax.plot(Bspline3Dpoints[:,0], Bspline3Dpoints[:,1], Bspline3Dpoints[:,2], color=colorscheme[color])
                #self.plotImage3D(ax, echo.dicom, echo.framenr)

        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        # set limits
        dd  = 1.
        lim = [ np.min(datapoints3D)*dd, np.max(datapoints3D)*dd ]
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_zlim(lim)
        plt.show()
        return

    def plotImage3D(self, ax, dicom, framenr : int): # Plot a 2D DICOM image in 3D space (as a plane), not yet working perfectly

        dx = dicom.SequenceOfUltrasoundRegions[0].PhysicalDeltaX/100 # Convert [cm] to [m]
        dy = dicom.SequenceOfUltrasoundRegions[0].PhysicalDeltaY/100

        img = dicom.pixel_array[framenr,:,:,:]
        img_gray = np.dot(img, [0.2989, 0.5870, 0.1140]) # Convert to gray scale
        lx, ly = img_gray.shape 
        xx, yy   = np.meshgrid(np.linspace(0,ly*dy,ly), np.linspace(0,lx*dx,lx))
        cset = ax.contourf(xx, yy, img_gray, 100, zdir='z', offset=0.5, cmap="gray")
        return
    
    @staticmethod
    def save_to_txt(*echos : Echo, filename : str ="Datapoints 3D", outputDir : str = "output"):
        filepath = os.path.join(outputDir,filename+".txt")
        with open(filepath, 'w') as f:
            f.write("Data points (3D) extracted from 2D echocardiogram data"+"\n\n")
            for echo in echos:
                for key, data in echo.datapoints3D.items():
                    view = echo.view
                    surf = key
                    f.write(f"View {view} of {surf}\n")
                    if type(data) == np.ndarray:
                        lines = "\n".join( [ str(row.tolist()) for row in data ] )
                    f.write(lines) 
                    f.write("\n\n")     
        return
    
    @staticmethod
    def load_txt(filename : str ="Datapoints 3D", inputDir : str = "input", cutBase=False, combineViews=False):
        filepath   = os.path.join(inputDir,filename+".txt")
        datapoints = {"epicard": [], "endocard": []} # Initialize
        import ast

        with open(filepath, 'r') as f:
            lines = f.readlines()
        # Strip empty spaces and remove '\n'
        lines = [idat.strip("\n") for idat in lines if len(idat.strip("\n")) != 0]   
        idx   = [] #{"epicard": [], "endocard": []}
        catch_line = "View"
    
        for i, line in enumerate(lines):
            if catch_line in line.strip(" "):
                idx += [i]             
        idx += [None]

        datapoints = {bound : {} for bound in ("endocard","epicard")}
        for i,j in zip(idx[:-1],idx[1:]):
            bound = "endocard" if "endocard" in lines[i] else "epicard"
            if "AP2CH" in lines[i]:
                iview = "AP2CH" 
            elif "AP4CH" in lines[i]:
                iview = "AP4CH"
            elif "PSAX" in lines[i]:
                iview = "PSAX"   

            datapoints[bound][iview] = np.array([ast.literal_eval(iline) for iline in lines[(i+1):j]])


        BaseTangs  = []
        BasePoints = []
        for iview in "AP2CH","AP4CH":
            BasePoint1 = np.mean(np.c_[datapoints["endocard"][iview][0,:] , datapoints["epicard"][iview][0,:]], axis=1)
            BasePoint2 = np.mean(np.c_[datapoints["endocard"][iview][-1,:], datapoints["epicard"][iview][-1,:]], axis=1)
            baseTangVec3D = ( BasePoint1 - BasePoint2 ) # Mean tangential vector allong the basal plane
            baseTangVec3D /= np.linalg.norm(baseTangVec3D)
            BaseTangs.append(baseTangVec3D)
            BasePoints.append(0.5*(BasePoint1+BasePoint2))

        baseNormalVec3D  = np.cross(*BaseTangs)
        baseNormalVec3D /= np.linalg.norm(baseNormalVec3D)
        baseNormalVec3D *= 1 if np.dot(baseNormalVec3D,np.array([0,0,1])) > 0 else -1 # make sure it is pointing upwards
        BasePoint = 0.5*sum(BasePoints)
        #BasePoint -= np.array([0,0,0.008])
        Baseplane = dict(normal=baseNormalVec3D, point=BasePoint)

        if cutBase:
            # Cut off the datapoints above this plane (if there are any)
            datp=datapoints.copy()
            for bound, views in datapoints.items():
                for iv, val in views.items():
                    del_indices = np.argwhere( np.sum(val*baseNormalVec3D, axis=1) - np.dot(BasePoint, baseNormalVec3D) > 0 )
                    datp[bound][iv] = np.delete(val, del_indices, axis=0)
            datapoints = datp

        if combineViews:
            datapoints = {bound : np.concatenate([val for iv, val in views.items()],axis=0) for bound, views in datapoints.items()}

        return datapoints, Baseplane

    @staticmethod
    def get_3Dpoints(*echos : Echo, keepSlices=True): # TODO check keepSlices!
        points3D = {}
        for echo in echos:
            for bound, datapoints3D in echo.datapoints3D.items():
                if bound not in points3D.keys():
                    points3D[bound] = datapoints3D
                else:
                    points3D[bound] = np.r_[ points3D[bound], datapoints3D ]
        return points3D


# Example
if __name__ == '__main__':
    from pydicom import dcmread

    directC   = os.path.realpath(os.path.dirname(__file__)) # Current directory
    dirdicom  = "data" # Dicom file folder
    files     = "Echo_AP2CH_fnr30","Echo_AP4CH_fnr34","Echo_PSAX_fnr42"
    views     = [file.split("_")[1] for file in files]
    framenrs  = [int(file.split("_")[2][3:]) for file in files]
    loaddata  = True
    orientation = {"AP2CH": (0,  0,  0),
                   "AP4CH": (0,  0, 61),
                   "PSAX" : (0, 90,  0)} # Orientation along the principal axes: (x,y,z)

    echos   = []
    segment = Segmentation()
    for ifile, iview, iframenr in zip(files,views,framenrs):
        echo = Echo()
        echo.dicom    = dcmread(os.path.join(directC,dirdicom,ifile+".dcm"))
        echo.filename = ifile
        echo.framenr  = iframenr
        echo.view     = iview
        echo.orientation = np.array(orientation[views])

        if loaddata:
            echos.datapoints2D = segment.load
        echos += [echo]
       
    # Segment the DICOM files manually
    points  = segment.manual(*echos, returnPoints=True)
    #segment.save_to_txt("Datapoints 2D", outputDir=os.path.join(directC,"output")) # save results to txt-file

    Generate3Dpointcloud(*echos).interpolate()