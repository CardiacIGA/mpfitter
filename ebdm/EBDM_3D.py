# -*- coding: utf-8 -*-
"""
Created on Tue July 11th 2023

@author: R. Willems

Optimized and adjusted variant of the EBDM algorithm developed by L. Verberne. Merged fitting algorithm toolbox for 3D surfaces on 
data points, based on the case specific data points set from echocardiograms. Based on the GSA and CPBD methods, merged and applied 
to splines which create a surface. Also contains the post processing tools to generate plots.

Optimized performance by creating own surface data structures based on numpy arrays
"""

import math, os, time, treelog
from scipy.spatial import cKDTree
import scipy
from geomdl import BSpline, operations, NURBS
import numpy as np
import splipy as sp
import matplotlib.pyplot as mpl
import ebdm.toolbox as toolbox
import splipy.surface as SplipyS
import warnings
from scipy.spatial import Voronoi

_=np.newaxis

class EBDM():
    '''The EBDM algorithm class'''

    def __init__(self, *surfaces, datapoints=np.zeros(3), connection=[], evalpts=100):
        '''Initializer 
        
        Input:
        _____________
        surfaces   : list or geomdlSurface;  The initial or template NURBS or B-spline surfaces used for fitting.
        datapoints : np.ndarray of size 3;   The datapoints to which the surface should be fitted.
        connection : list;                   Connectivity of the individual surfaces in case multiple surfaces are provided as input.

        '''

        # Store the input
        self.surfaces    = surfaces
        self.surfaces0   = [surface.clone() for surface in surfaces]
        self.datapoints0 = datapoints # The original datapoints array
        self.connection  = connection
        self.evalpts     = evalpts
        self.warned      = {} # Initialize a warning dict

        # Construct the multipatch mapper
        self._multipatch_mapper_init(self.surfaces,  update=True)

        # Initialize fixed normal vecs (if there are any)
        self.fixed_normals = False
        self.compute_fixed_normals(0) # <- Will set fixed_normals = True if there are any

        # Check for plane constraint (also check for other constraints, but these have disp_vector = null, so no change)
        # If the boundary is within the plane, nothing happens, otherwise, we shift the cps into the plane (forcefully).  
        # We also check if the normal is set to 'fixed', in this case we only use the normals in the initial configuration as constraint vectors.
        for surf in self.surfaces:  # This step is also repeated every iteration in the .run(), but then we have disp_vector != null
            nrcps_u, nrcps_v = surf.nrcps()
            surf.disp_vectors = np.zeros((nrcps_u, nrcps_v, 3)) # Initilize null
            surf.disp_vectors = self.constrain_vector(surf, init=True)
            self.update_surface_cps(surf) # cps and w

        # Divide the data points (assigndata points to correct surfaces/nearest neighbour)
        self.assign_datpoints_to_surface()

        # Initialize the data point object
        self.dataset = self.compute_dataset()    
        self.compute_integral_surfaces()
        return


    #TODO: Add excluding boundaries from continuity check
    def run(self, max_iter=300, max_refs=0, atol=1e-5, rtol=1e-3, ref_rtol=5e-4, displace_relax=3., continuity_relax=1., continuity_iters_only=3, cont_iters_every=1, start_cont=0, save_iters_every=None, ordering=[], reval_every=30, recons_every=1, refine_wait=50, constrainCont=False):
        '''
        Executer function of the EBDM algorithm. Every iteration, each surface patch is evaluated in terms of its displacement and continuity error. 
        The control points are then corrected/displaced accordingly. 
        
        Input:
        _____________
        max_iter   : int;         Maximum number of iterations per refinement level.
        max_refs   : int;         Maximum number of refinements.
        atol       : float;       Absolute tolerance [m] of the fitting algorithm. If the error is below he tolerance, the iterations are stopped.
        rtol       : float;       Relative tolerance [-] of the fitting algorithm. If the error is below he tolerance, the iterations are stopped.
        ref_rtol   : float;       Relative refinement tolerance [-]. When the (relative) residual has a difference of < ref_rtol over 2 iterations, uniform refinement is applied.
   
        displace_relax        : float; Displacement relaxation factor. Overrelaxation is typically desired (>1) due to the convoluting nature of the algorithm (no overshoots with <=1). 
        continuity_relax      : float; Continuity relaxation factor. A value = 1 will enforce C1-geometrical continuity, 0 will neglect this.
        continuity_iters_only : int;   Number of iterations that are used at the prior to i=max_iter (max_iter - continuity_only) to only enforce continuity and neglect displacement.
        start_cont            : int;   Start correcting for the continuity only after nth iterations (if iter > start_cont).
        save_iters_every      : int or None;  Save the iteration results every nth iteration, provide None when not saving

        ordering     : list; Specify a list which lists the ordering of the patches from [most_important, .. -> .., least_important]
        reval_every  : int ; Re-evaluate the surface-datapoint projection every nth iteration. Higher values slow down the algorithm and are not needed for increased accuracy.
        recons_every : int ; Reconstrain/apply the constraints to the control points every nth iteration (value of 1 is desired).
        refine_wait  : int ; Wait nth iterations after a uniform refinement before refining again.
        constrainCont: Bool; True means that the continuity correction is constraint as well.

        Output:
        ____________
        surfaces_fitted : list or geomdlSurface;   The computed or fitted surfaces.

        '''
        if not ordering:
            ordering = [i for i in range(len(self.surfaces))]
        assert len(ordering)==len(self.surfaces), "Specified ordering should be equal to the number of surfaces"

        # Check if there is a constraint which uses the 'fixed normals'
        if self.fixed_normals:
            self.compute_fixed_normals(max_refs)

        # Initialize error arrays
        self.disp_error = np.ndarray(max_iter+1, float)
        self.cont_error = np.ndarray(max_iter+1, float)
        self.kineticE   = np.ndarray(max_iter+1, float)
        # set relative quantities (for disp-error)
        self.disp_error0 = self.dataset.total_error()
        #self.disp_error0 = self.dataset.average_diameter()

        self.disp_error[0] = self.dataset.total_error()
        self.cont_error[0] = self.continuity_error(self.surfaces)
        self.kineticE[0]   = 0
        sizeIt  = len(str(max_iter))
        iterstr = f"{0:<{sizeIt}}"
        treelog.info(f"Iteration {iterstr} L\N{SUPERSCRIPT TWO}-Error: {self.disp_error[0]/self.disp_error0:.2E} C-Error: {self.cont_error[0]:.2E}  Kinetic-E: {self.kineticE[0]:.2E}")

        surfaces_iter  = [] # A list in which the results are stored when save_iters_every is provided
        self.ref_idx   = [] # List of indices when a refinement was conducted
        self.reval_idx = [] # List of indices when the surfaces are re-evaluated (datapoint projection)
        evalpts = (self.evalpts, self.evalpts) #Number of evaluation points in u- and v-directions
        for i in range(max_iter):


            # Recheck the division of the datapoints on the surfaces (projection)
            if i in range(0, max_iter, reval_every)[1:]: # Only re-evaluate/divide every (reval_every)nth iteration
                self.reval_idx.append(i)
                self.assign_datpoints_to_surface()    # Divide the data points (assigndata points to correct surfaces/nearest neighbour)
                self.dataset = self.compute_dataset() # Load the assigned data into the appropriate Dataset object       


            # Some of the last iterations we only want to have a continuity correction as the fit is already correct.
            if i >= (max_iter - continuity_iters_only):
                displace_relax = 0

            # Loop over each patch surface
            for s, surface in enumerate(self.surfaces):

                if i in range(0, max_iter, reval_every)[1:]:
                    # Compute the weights (stored in the surfaces)
                    #surface.Nintegral, surface.Nevalbases = self.compute_integral_surface(surface, print_stats=True)
                    surface.Nintegral, surface.Nevalbases, surface.Sintegral = self.compute_integral_surface_v2(surface, print_stats=False)
                    #self.compute_integral_surface_v2(surface)
                #────────────────────────────────────────────────────#
                # Build the complete displacement vector for each cps:
                #────────────────────────────────────────────────────#

                # 1) Compute the displacement of the cps based on the data (also applies constraints)
                surface.disp_vectors = displace_relax*self.compute_displacement(surface, self.dataset(surface.patchID)) # Pure displacement

                # 2) Constrain the displacement
                if i in range(0, max_iter, recons_every):
                    surface.disp_vectors = self.constrain_vector(surface)
   
                # Update the surface and dataset values
                #self.update_surface_cps(surface, disp_vectors) # cps and w
                
                
            ## Perform multipatch/patchinterface action (outside individual surface loop)    
            # 3) Connect the surfaces
            self.surfaces = self.connect_surfaces(self.surfaces)
            self.surfaces = self.connect_extraordinary_nodes(self.surfaces)

            for surf in self.surfaces:  # -> update the SpliPy surface object()
                # Constrain the displacement due to the continuity vector
                if (i in range(0, max_iter, recons_every)) and constrainCont:
                    surf.disp_vectors = self.constrain_vector(surf, skipInterface=True) # Special case when constraining in normal direction -> Neglect interface, this has already been done in the continuity step
                self.update_surface_cps(surf) # cps and w


            # 3) Apply weighting for patch-interface continuity
            if i >= start_cont:
                for cont_iter in range(cont_iters_every): # Perform a specific number of cont iterations every fit iteration
                    #self.continuity(self.surfaces, continuity_relax=continuity_relax)
                    self.continuity_v2(self.surfaces, continuity_relax=continuity_relax)
                    for surf in self.surfaces:  # -> update the SpliPy surface object()
                        # Constrain the displacement due to the continuity vector
                        if (i in range(0, max_iter, recons_every)) and constrainCont:
                            surf.disp_vectors = self.constrain_vector(surf, skipInterface=True) # Special case when constraining in normal direction -> Neglect interface, this has already been done in the continuity step
                        self.update_surface_cps(surf) # cps and w


            for surf in self.surfaces:  # -> update the Dataset object()  
                self.dataset.update_surfacepoints( surf.sample(evalpts=evalpts)[surf.surfacepoints_index], # Evaluated surface points -> Only select indices that have a datapoint 'attached' to it
                                                   surf.datapoints_index) # Indices that correspond to the complete datapoint array     

            # Assign values to error array
            self.disp_error[i+1] = self.dataset.total_error()
            self.cont_error[i+1] = self.continuity_error(self.surfaces)
            self.kineticE[i+1]   = self.total_kinetic_energy(self.surfaces, average=True)

            # Save each surface if specified
            if save_iters_every != None and (i in range(0,max_iter,save_iters_every)):
                surfaces_iter.append([surf_copy.clone() for surf_copy in self.surfaces])

            # Check if we have reached the desired tolerance 
            iterstr = f"{i+1:<{sizeIt}}"
            treelog.info(f"Iteration {iterstr} L\N{SUPERSCRIPT TWO}-Error: {self.disp_error[i+1]/self.disp_error0:.2E} C-Error: {self.cont_error[i+1]:.2E} Kinetic-E: {self.kineticE[i+1]/self.disp_error0:.2E}")   
            if self.disp_error[i+1] < atol or self.disp_error[i+1]/self.disp_error0 < rtol:
                self.disp_error = self.disp_error[:(i+2)] # Slice the error array, the algorithm finished before it was completely filled
                self.cont_error = self.cont_error[:(i+2)]
                self.kineticE   = self.kineticE[:(i+2)]
                break

            # Evaluate the error, if it does not change sginificantly -> refine if allowed 
            if i > 1 and abs(self.disp_error[i-1] - self.disp_error[i+1])/self.disp_error[i-1] < ref_rtol \
                and len(self.ref_idx) < max_refs \
                and i > (self.ref_idx or [0])[-1] + refine_wait: # Refinement should wait atleast 50 iterations compared to the last 
                self.ref_idx.append(i+1)

                treelog.info(f"Refining the surface (ref={len(self.ref_idx)})")
                for surface in self.surfaces:
                    #surface.refine(1,0)
                    surface.refine(1,1)
                    surface.nr_refine = ( len(surface.knots()[0])-2, len(surface.knots()[0])-2 )#(2**(len(self.ref_idx)-1), 2**(len(self.ref_idx)-1))

                self.reval_idx.append(i+1)
                self.assign_datpoints_to_surface()    # Divide the data points (assigndata points to correct surfaces/nearest neighbour)
                self.dataset = self.compute_dataset() # Load the assigned data into the appropriate Dataset object 
                self._multipatch_mapper_init(self.surfaces, update=True) # Reconstruct the multipatch mapper
                for surface in self.surfaces:
                    #surface.Nintegral, surface.Nevalbases = self.compute_integral_surface(surface, print_stats=True)
                    surface.Nintegral, surface.Nevalbases, surface.Sintegral = self.compute_integral_surface_v2(surface, print_stats=False)

                

            
        contE, self.spatialError = self.continuity_error(self.surfaces, return_spatial=True, nsample=100)

        return [surf.clone() for surf in self.surfaces] if not save_iters_every else surfaces_iter


    def error_info(self, filename="Error info.json", save=False):
        import json
        error_info =  {"Displacement error" : self.disp_error,
                       "Displacement norm"  : self.disp_error / self.disp_error0, 
                       "Continuity error"   : self.cont_error,
                       "Kinetic energy"     : self.kineticE,  
                       "Refinement indices" : self.ref_idx,
                       "Re-evaluate indices": self.reval_idx,
                       "Spatial c-error"    : self.spatialError["Spatial continuity error"],
                       "Spatial interface coords" :self.spatialError["Spatial interface coordinates"]}
        if save:
            error_info_lst = {key : value.tolist() if type(value)==np.ndarray else value for key, value in error_info.items()}
            filename_spl = filename.split(".")
            if len(filename_spl) == 2:
                assert filename_spl[-1] == f"json", "Saving the error info only supported for .json files not {filename_spl[-1]}"
            with open(f"{filename_spl[0]}.json", "w") as outfile:
                 json.dump(error_info_lst, outfile)

        return error_info

    def assign_datpoints_to_surface(self,):
        '''Find the list of data points relevant for each surface by finding which surface is closest to the data point'''

        evalpts_tot = np.empty((0,3), float)
        #localevalpts_tot = np.empty((0,2), float)
        surflen = []
        evalpts = (self.evalpts, self.evalpts) # Evaluation points in u- and v-direction
        for surface in self.surfaces:

            coords, local_coords = surface.sample(evalpts=evalpts, returnLocal=True) #self.eval_surface(surface, evalpts=evalpts, returnLocal=True)

            surflen.append(len(coords)) 
            evalpts_tot = np.concatenate([evalpts_tot, coords], axis = 0)
            #localevalpts_tot = np.concatenate([localevalpts_tot, local_coords], axis = 0)

            # Initialize some quantities for next loop
            # surface.datapoints    = np.empty((0,3), float)       # Datapoints appointed to the surface
            # surface.surfacepoints = np.empty((0,3), float)       # Physical points on the surface that are the nearest neighbour to the datapoint
            # surface.datapoints_index    = np.empty((0,), int)    # The index of the datapoint in the complete datapoint array 
            # surface.surfacepoints_index = np.empty((0,), int)    # The index of the surfacepoint in the complete surfacepoint array
            # surface.surfacepoints_local = np.empty((0,2), float) # The local (parametric) coordinate corresponding to the .surfacepoints
            surface.surfacepoints_local_total = local_coords     # The total grid of the local coordinates used to evaluate the complete surfacepoints array

        

        

        # Fast look-up for nearest neighbours method
        voronoi_kdtree = cKDTree(evalpts_tot) 
        point_dist, point_regions = voronoi_kdtree.query( self.datapoints0 )
        # Check if we have local duplicate regions, if so, neglect the duplicate datapoints
        u, ind = np.unique(point_regions, return_index=True)
        self.datapoints = self.datapoints0[ind] # Set the new datapoints array
        point_regions   = point_regions[ind]

        # Check if points are at or close to the interface, if so, duplicate point and assign it to both patches instead of only 1
        point_regions_int = voronoi_kdtree.query_ball_point( self.datapoints, point_dist[ind] + 1e-4/(self.evalpts-1))
        for i, interface_regions in enumerate(point_regions_int):
            # Check which patches are involved per region
            patch_interval = np.concatenate([np.concatenate([np.array([0]), np.cumsum([surflen])])[:-1,_], 
                                             np.concatenate([np.array([0]), np.cumsum([surflen])])[1:,_]  ], axis=1)
            
            mask_ifreg = lambda intreg : (intreg >= patch_interval[:,0][:,_]).any(1) & (intreg < patch_interval[:,1][:,_]).any(1)
            patchID_interf_regions = [np.where(mask_ifreg(intreg) == True)[0][0] for intreg in interface_regions] 
            patchIDs_interf = np.unique(patchID_interf_regions) 
            
            if len(patchIDs_interf) > 1: # We have multiple patches
                for patchID_interf in patchIDs_interf: 
                    point_region_patch = np.array(interface_regions)[np.equal(patchID_interf_regions, patchID_interf) ]

                    if point_regions[i] in point_region_patch: # Already within structure
                        continue
                    else: # Find point closest to current point = self.datapoints[i]
                        points_on_patch = evalpts_tot[point_region_patch]
                        idx_add = np.argmin( np.linalg.norm(evalpts_tot[point_regions[i]] -  evalpts_tot[point_region_patch], axis=1) )
                        self.datapoints = np.concatenate([self.datapoints, self.datapoints[i][_] ])# Duplicate point
                        point_regions   = np.concatenate([point_regions,   np.array([ point_region_patch[idx_add] ])])



        c = np.tile(point_regions[:,_], len(self.surfaces))
        rel_index_arr = c - np.concatenate([np.array([0]), np.cumsum([surflen])])[:-1]

        index = np.zeros(len(self.datapoints), int)
        rel_dpoint_index = np.zeros(len(self.datapoints), int)
        for i, rel_index in enumerate(rel_index_arr):
            rel_dpoint_index[i] = (rel_index[np.where( rel_index >= 0)]).min()
            index[i] = rel_index[np.where( rel_index >= 0)].argmin()

        for k, surface in enumerate(self.surfaces):
            idx = np.where( k == index )
            surface.datapoints = self.datapoints[idx]               # Datapoints appointed to the surface
            surface.surfacepoints = evalpts_tot[point_regions[idx]] # Physical points on the surface that are the nearest neighbour to the datapoint
            surface.surfacepoints_local = surface.surfacepoints_local_total[rel_dpoint_index[idx]] # The local (parametric) coordinate corresponding to the .surfacepoints
            surface.datapoints_index = idx                          # The index of the datapoint in the complete datapoint array 
            #surface.surfacepoints_index = point_regions[idx]        # The index of the surfacepoint in the complete surfacepoint array
            surface.surfacepoints_index = rel_dpoint_index[idx]     # The index of the surfacepoint in the surfacepoint array

            # Store useful length measurements
            surface.nrdatapoints = len(surface.datapoints) # Number of datapoints assigned to the surface
            # surface.nrcps = len(surface) # Number of control points that define the surface
            # surface.nrcps_u, surface.nrcps_v = surface.shape

        # Class specific variables    
        #self.total_surface_points = evalpts_tot # All evaluated surface points, concatenated into 1 array -> Not required so don't bother storing/updating
        self.select_surface_index = index       # The surface (patch) index that corresponds with each datapoint    
        self.select_surface_points_index = point_regions # The indices of the evaluated points on the surfaces that correspond/are linked to the datapoint array
        self.select_surface_points = evalpts_tot[point_regions] # The select number of evaluated datapoints that correspond/are closest to the datapoint array
        

        # arg_min = np.zeros(len(self.datapoints), int)
        # index = np.zeros(len(self.datapoints), int)
        # for i, datpoint in enumerate(self.datapoints):
        #     dist = np.linalg.norm(evalpts_tot - datpoint, axis=1 ) #np.sum((nodes - data_points[i])**2, axis=1)
        #     arg_min[i] = np.argmin(dist)

        #     rel_index_arr = arg_min[i] - np.concatenate([np.array([0]), np.cumsum([surflen])])[:-1]
        #     rel_dpoint_index = rel_index_arr[np.where( rel_index_arr >= 0)].min()
        #     index[i] = rel_index_arr[np.where( rel_index_arr >= 0)].argmin()

        #     self.surfaces[index[i]].datapoints    = np.concatenate([ self.surfaces[index[i]].datapoints, datpoint[_] ], axis = 0) # Physical datapoint coordinate
        #     self.surfaces[index[i]].surfacepoints = np.concatenate([ self.surfaces[index[i]].surfacepoints, evalpts_tot[arg_min[i] ,_] ], axis = 0)
        #     self.surfaces[index[i]].surfacepoints_local = np.concatenate([ self.surfaces[index[i]].surfacepoints_local, 
        #                                                                    self.surfaces[index[i]].surfacepoints_local_total[rel_dpoint_index ,_] ], axis = 0)
            
        #     self.surfaces[index[i]].datapoints_index    = np.concatenate([ self.surfaces[index[i]].datapoints_index, np.array([i]) ]) # Datapoint index corresponding to the initial datapoint array
        #     self.surfaces[index[i]].surfacepoints_index = np.concatenate([ self.surfaces[index[i]].surfacepoints_index, np.array([rel_dpoint_index]) ])


        # self.select_surface_index = index # The surface (patch) index that corresponds with each datapoint    
        # self.select_surface_points_index = arg_min # The indices of the evaluated points on the surfaces that correspond to the datapoint array
        # self.select_surface_points = self.total_surface_points[arg_min] # The select number of evaluated datapoints that correspond/are closest to the datapoint array
        return
    
    # @staticmethod
    # def eval_surface(surface : sp.Surface, evalpts=(100, 100), returnLocal=False): #-> np.ndarray
    #     u_end, v_end = surface.end()
    #     u_start, v_start = surface.start()

    #     u_sample = np.linspace(u_start, u_end, evalpts[0])
    #     v_sample = np.linspace(v_start, v_end, evalpts[1])
        
    #     if returnLocal:
    #         u, v = np.meshgrid(u_sample, v_sample) 
    #         local = np.concatenate([ u[:,:,_], v[:,:,_] ], axis=2).reshape(-1,2)
    #         return surface(u_sample, v_sample).swapaxes(0,1).reshape(-1,3), local
    #     else:
    #         return surface(u_sample, v_sample).swapaxes(0,1).reshape(-1,3) # Swap axis, because SpliPy + reshape operation..   

    def compute_dataset(self,):
        '''Construct an object that contains all the relevant information per datapoint'''
        return DataPointSet(self.datapoints, self.select_surface_points, self.select_surface_points_index, self.select_surface_index )

    @staticmethod
    def update_surface_cps(surface):
        dvec = np.transpose(surface.disp_vectors, axes=(1,0,2))
        surface.dx = np.linalg.norm(dvec, axis=2).sum() # save to dx variable (used for kinetic energy computation)
        surface.disp_vectors = np.zeros(surface.disp_vectors.shape, float)# Reset the displacement vectors
        if surface.rational:
            weights = surface.controlpoints[...,-1] #[:,:,-1]
            surface.controlpoints[...,:-1] += dvec*weights[...,_] #[:,:,:-1] [:,:,_]
        else:
            surface.controlpoints += dvec
        return surface

    def compute_integral_surfaces(self,):
        for patch, surface in enumerate(self.surfaces):
            #surface.Nintegral, surface.Nevalbases = self.compute_integral_surface(surface, print_stats=True)
            surface.Nintegral, surface.Nevalbases, surface.Sintegral = self.compute_integral_surface_v2(surface, print_stats=False)
        return

    def compute_integral_surface_v2(self, surface, print_stats=False):
        '''Compute the area of the projected datapoint on the NURBS surface. The area is based on the area of each Voronoi segment 
        bounded by a box (0,0) - (1,1), lower left, upper right vertices of the box. Total area should be equal to 1.'''

        # nr of datapoints projected on the local surface
        nrsurfpoints = surface.nrdatapoints
        
        # Build the Voronoi object/diagram
        if nrsurfpoints == 0: # Exception, do not build Voronoi diagram
            AreaPoints = np.array([])

        else: # Else we can calculate the area or weights
            
            if nrsurfpoints < 4: # if voronoi gets <4 points, qhull error is raised, so add some points far away
                rndm_points  = np.array([ [1e2, 1e2], [-1e2, 1e2], [1e2, -1e2] ])
                local_points = np.concatenate([surface.surfacepoints_local, rndm_points])
            else:
                local_points = surface.surfacepoints_local
            vor = Voronoi(local_points, qhull_options="QJ") # Create map with nearest neighbour regions (Voronoi diagram)

            # Compute the bounded polygons (segments of the voronoi diagram)
            regions, vertices = self._voronoi_finite_bbox(self, vor, bbox=(1,1))

            # calculate area polygons
            AreaPoints = self._area_polygons(vertices, regions)
        
        # Integrate in u-direction and multiply results with dv (spacing in v-direction)
        ubasis = surface.bases[0]
        vbasis = surface.bases[1]
        #N_intbases  = np.zeros((nrsurfpoints, vbasis.num_functions(), ubasis.num_functions()), float) # For the integral of the bases in the Voronoi diagram
        N_evalbases = np.zeros((nrsurfpoints, vbasis.num_functions(), ubasis.num_functions()), float) # For the evaluation of the bases at the datapoint (local)
        
        for i, surfpoint in enumerate(surface.surfacepoints_local):
            #N_intbases[i]  = surfpoint_area
            N_evalbases[i] = (ubasis.evaluate(surfpoint[0])*vbasis.evaluate(surfpoint[1]).T)
        
        # Normalize ()
        #N_evalbases /= nrsurfpoints if nrsurfpoints != 0 else 1 # -> Division by nr of datapoints ensures unity: sum(Neval)=1
        # Integral of each curve
        Uintegral = np.array(ubasis.integrate(ubasis.start(),ubasis.end()))
        Vintegral = np.array(vbasis.integrate(vbasis.start(),vbasis.end()))
        Sintegral = Uintegral*Vintegral[:,_] # Integral underneath each basis function

        # Do some post-processing and checks
        exactA = sum(ubasis.integrate(ubasis.start(),ubasis.end()))*sum(vbasis.integrate(vbasis.start(),vbasis.end()))
        assert np.isclose(exactA,1), "Parametric surfaces with area larger than 1 are not supported. Consider splipy function self.reparam()"
        
        if print_stats:
            errorA = (exactA-np.sum(AreaPoints))*100 if nrsurfpoints else 0 # Should be 0!
            treelog.info(f"Surface (patch{surface.patchID}) integral accuracy: {errorA:.1f}\%")


        return AreaPoints, N_evalbases, Sintegral # Store the integral values of each Voronoi diagram (rows = Vornoi diagram index; column = basis function index)       
        

    
    @staticmethod
    def _voronoi_finite_bbox(self, vor, bbox=(1,1), comp_points=8):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions with vertices bounded to a specific box (min/max).

        Parameters
        ----------
        vor : Voronoi
            Input diagram
        bbox : tuple, optional
            Bounding box in which vertices exist.

        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. 

        """

        # Set the bounding box
        bbverts = np.array([[0,0],[bbox[0],0],[bbox[0],bbox[1]],[0,bbox[1]]]) # bounding box xy-coordinates, counterclockwise
        bnames  = "bottom","right","top","left" # Name of the boundaries
        bbverts = np.append(bbverts, bbverts[0][np.newaxis],axis=0)
        bbox_lines = {f"line {bname}" : ( bv1, bv2 - bv1 ) for bname, bv1, bv2 in zip(bnames,bbverts[1:],bbverts[:-1])} 

        # Find region where bboxverts are part of
        bbpoint_index = {}
        for bbvert in bbverts[:-1]:
            dist_vert  = np.linalg.norm((vor.points - bbvert), axis=1)
            arg_min    = np.argsort(dist_vert) 
            i          = 2 if np.isclose(*dist_vert[arg_min[:2]]) else 1
            point_indx = arg_min[:i]
            for pidx in point_indx:
                if pidx in bbpoint_index.keys():
                    bbpoint_index[pidx] = np.concatenate([bbpoint_index[pidx], bbvert[_]], axis=0)
                else: 
                    bbpoint_index[pidx]  = bbvert[_]

        # Check vertices if they are within the bounding box, else remove them and keep track of the removed index
        new_regions = []
        
        # Remove vertices oustide bbox:
        del_indices  = ~self._mask_bbox(vor.vertices) #np.argwhere( (vor.vertices > 1).any(1) ).ravel()
        del_indices_int = np.argwhere(del_indices==True).ravel()
        old_vertices = vor.vertices
        new_vertices = np.delete(vor.vertices, del_indices,axis=0)# (vor.vertices[removed_indices]).tolist()

        # Construct a map for deleted vertices, the idea: "new_index = new_indices_map[old_index]"
        old_indices = np.linspace(0,len(vor.vertices)-1,len(vor.vertices)).astype(int)
        new_indices = np.linspace(0,len(new_vertices)-1,len(new_vertices)).astype(int)
        i = 0
        new_indices_map = {}
        for old_i in old_indices:
            if old_i not in del_indices_int:
                new_indices_map[old_i] = new_indices[i]
                i += 1

        new_indices_map.update({k : -1 for k in del_indices_int})
        new_indices_map[-1] = -1 # default 

        # Construct map to obtain point index from voronoi region index, the idea: "point_index = region_indices_map[region_index]"
        # point_2_region_indices_map = {point_i : region_i for point_i, region_i in enumerate(vor.point_region)}    
        # region_2_point_indices_map = {region_i : point_i for point_i, region_i in enumerate(vor.point_region)}

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, new_indices_map[v1], new_indices_map[v2]))
            all_ridges.setdefault(p2, []).append((p1, new_indices_map[v1], new_indices_map[v2]))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertex_indices = np.array([new_indices_map[vert] for vert in vor.regions[region]])

            # We encountered a finite region
            if all(v >= 0 for v in vertex_indices):
                # finite region
                new_regions.append(vertex_indices.astype(int).tolist())
                continue

            # We encountered a nonfinite region
            ridges = all_ridges[p1]

            new_vert = np.ndarray((0,2),float)
            for p2, v1, v2 in ridges:

                # if v1 and v2 are both -1, we need to find 2 intersecting points with the bbox
                t = vor.points[p2] - vor.points[p1] # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint  = vor.points[[p1, p2]].mean(axis=0)
                line      = (midpoint, n) # line of interest
                xpoints   = np.array([self._intersec_lines(line, bline) for bname, bline in bbox_lines.items()]) # array of intersection points, we need to find the correct ones
                xpoints_mask = self._mask_bbox(xpoints) #(-1e-9 <= xpoints[:,0]) & (xpoints[:,0] <= 1+1e-14) & (-1e-9 <= xpoints[:,1]) & (xpoints[:,1] <= 1+1e-14) # filter out points that are outside the bbox
                xpoints   = xpoints[xpoints_mask]

                dreg={}
                for d, x in enumerate(xpoints):
                    dnorm = np.linalg.norm(vor.points-x,axis=1)
                    point_indices = np.argsort(dnorm)
                    # construct list of point indices that share the same distance to the xpoint    
                    point_indices_closest = [l for l in point_indices[:comp_points] if np.isclose(dnorm[point_indices[0]],dnorm[l])]
                    if p1 in point_indices_closest: # ridge intersection point with bbox
                        dreg[d] = p1 #point_indices_closest[:2] # Get the point index we are closest to
                    # else:
                    #     dreg[d] = np.array([point_indices_closest[0]])
                
                # We are left with two possible xpoints, check if both are required (v1=v2=-1) or only one (v1=-1 or v2=-1)
                if v1 == -1 and v2 == -1:
                    for d, drg in dreg.items():
                            if p1 == drg:
                                new_vert = np.concatenate([new_vert, xpoints[d][_]], axis=0)
                
                else: # we have to determine which is the correct one by checking if either of the two exists in a different region
                    for d, drg in dreg.items():
                            if p1 == drg:#The considered point is indeed part of p1 region
                                new_vert = np.concatenate([new_vert, xpoints[d][_]], axis=0)

            # Check if corners are part of the region
            if p1 in bbpoint_index.keys():   
                new_vert = np.concatenate([new_vert, bbpoint_index[p1]], axis=0)

            # Append the results
            vertex_indices  = np.concatenate([vertex_indices,len(new_vertices)+ np.array(range(0,len(new_vert)))])
            vertex_ind_keep = np.where(vertex_indices >= 0)
            vertex_indices  = vertex_indices[vertex_ind_keep].astype(int).tolist()
            new_vertices    = np.concatenate([new_vertices, new_vert], axis=0)
            
            # Sort region counterclockwise and make sure we only do this for regions that exist
            if len(new_vert) != 0:
                vs = np.asarray([new_vertices[v] for v in vertex_indices])
                c = vs.mean(axis=0)
                angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
                vertex_indices = np.array(vertex_indices)[np.argsort(angles)]

                new_regions.append(vertex_indices)

        return new_regions, new_vertices

    @staticmethod
    def _intersec_lines(line1, line2): # input line = ( np.array([]), np.array([]) ), first index is the physica lpoint, second the vector travelling from that point
        point1, vec1 = line1
        point2, vec2 = line2
        if np.isclose( abs(np.dot(vec1/np.linalg.norm(vec1),vec2/np.linalg.norm(vec2))), 1):# check if vectors are parallel
            return np.array([1e3,1e3])
        else:
            M = np.concatenate([vec1[:,_], -vec2[:,_]], axis=1)
            f = (point2 - point1)[:,_]
            sol = np.linalg.solve(M,f)
            return sol[0]*vec1 + point1


    def _area_polygons(self, points, connectivity):
        area_pr = np.zeros(len(connectivity))
        for i, conn in enumerate(connectivity):
            area_pr[i] = self._area_polygon(points[conn])
        return area_pr
    def _area_polygon(self, p):
        return 0.5 * abs(sum(x0*y1 - x1*y0 for ((x0, y0), (x1, y1)) in self._segments(p)))
    @staticmethod
    def _segments(p):
        return zip(p, np.concatenate([p[1:] , p[0][_]],axis=0))
    @staticmethod
    def _mask_bbox(x, tol=1e-14): # x should be (n,2) shape
        return (-tol <= x[:,0]) & (x[:,0] <= 1+tol) & (-tol <= x[:,1]) & (x[:,1] <= 1+tol)

    # def compute_integral_surface(self, surface, print_stats=False):
    #     '''Compute the displacement  associated with the datapoints for each control point (not the same as NURBS weight!!)'''
    #     evalpts = (self.evalpts, self.evalpts)
   
    #     voronoi_kdtree = cKDTree(surface.surfacepoints_local) # Create map with nearest neighbour regions (Voronoi diagram)
    #     nrsurfpoints = surface.nrdatapoints
    #     #surface_evalpts = self.eval_surface(surface, evalpts=evalpts)
    #     point_dist, point_regions = voronoi_kdtree.query( surface.surfacepoints_local_total ) # Check in which diagram the remaining eval points are located
        
    #     # Integrate in u-direction and multiply results with dv (spacing in v-direction)
    #     ubasis = surface.bases[0]
    #     vbasis = surface.bases[1]
    #     N_intbases  = np.zeros((nrsurfpoints, vbasis.num_functions(), ubasis.num_functions()), float) # For the integral of the bases in the Voronoi diagram
    #     N_evalbases = np.zeros((nrsurfpoints, vbasis.num_functions(), ubasis.num_functions()), float) # For the evaluation of the bases at the datapoint (local)
    #     #N_intbases  = np.zeros((nrsurfpoints, ubasis.num_functions(), vbasis.num_functions()), float) # For the integral of the bases in the Voronoi diagram
    #     #N_evalbases = np.zeros((nrsurfpoints, ubasis.num_functions(), vbasis.num_functions()), float) # For the evaluation of the bases at the datapoint (local)
    #     for i, surfpoint in enumerate(surface.surfacepoints_local):
    #         idst, iregion = voronoi_kdtree.query(surfpoint)
    #         point_indices = np.where( iregion == point_regions )
    #         point_diagram = surface.surfacepoints_local_total[point_indices] # All points inside a specific diagram

    #         min_v = point_diagram[:,1].min() # Min value v-direction of diagram
    #         max_v = point_diagram[:,1].max() # Max value v-direction of diagram
    #         dv = (vbasis.end()-vbasis.start())/(evalpts[1]-1) # Spacing in v-direction

    #         v = min_v
    #         for k in range(int( (max_v-min_v)/dv )):
                
    #             points_constv = point_diagram[self.whereclose( v, point_diagram[:,1] )]
    #             u_min = points_constv[:,0].min()
    #             u_max = points_constv[:,0].max()
                    
    #             #N_intbases[i] += (np.array( ubasis.integrate(u_min, u_max) )*vbasis.evaluate(v).T*dv).T # value of the integrant per basis function in both directions
    #             N_intbases[i] += (np.array( ubasis.integrate(u_min, u_max) )*vbasis.evaluate(v).T*dv)
    #             v += dv    
    #         N_evalbases[i] = (ubasis.evaluate(surfpoint[0])*vbasis.evaluate(surfpoint[1]).T)

    #     # Give user an idea how well the integrals are (can be improved by increasing evalpts)
    #     if print_stats:
    #         exactA  = sum(ubasis.integrate(ubasis.start(),ubasis.end()))*sum(vbasis.integrate(vbasis.start(),vbasis.end()))
    #         approxA = abs(exactA-np.sum(N_intbases))/exactA*100 if nrsurfpoints != 0 else 0 # If there is nothing to integrate return 0
    #         treelog.info(f"Surface (patch{surface.patchID}) integral accuracy: {approxA:.1f}\%")    
        
    #     # Normalize ()
    #     N_evalbases /= nrsurfpoints if nrsurfpoints != 0 else 1 # -> Division by nr of datapoints ensures unity: sum(Neval)=1
    #     # Store result
    #     #surface.Nintegral = N_intbases # Store the integral values of each Voronoi diagram (rows = Vornoi diagram index; column = basis function index)       
    #     return N_intbases, N_evalbases # Store the integral values of each Voronoi diagram (rows = Vornoi diagram index; column = basis function index)       
        

    def compute_displacement(self, surface, dataset, relax=1):
        '''Compute the displacement associated with the datapoints for each control point (not the same as NURBS weight!!)
        Input:
        __________
        surface     : SpliPy surface object; Provide the surface 
        dataset     : DataPointSet object;   Contains all necessary datapoint information
        Output:
        __________
        displacement : np.ndarray; A numpy array of size (n,3) where n denotes the number of control points and 3 the dimensions (displacement for each control point).
        '''

        if surface.nrdatapoints == 0:
            nrcps_u, nrcps_v = surface.nrcps()
            return np.zeros((nrcps_u, nrcps_v, 3)) # No displacement
        
        # Perform a convolution type of operation
        error_vector = dataset.error_vector() # Size (n,3)
        Neval  = surface.Nevalbases           # Size (n,3,3) 
        weight = surface.Nintegral #np.sum(surface.Nintegral, axis=(1,2))  # Size sum(n,3,3) -> (n)

        displacement_vector  = np.transpose( np.dot(Neval.T,error_vector*weight[:,_]), axes=(1,0,2)) # Specific transpose ensures the ordering is correct again (due to transpose in dot product)
        #displacement_vector  /= np.sum(np.transpose( np.dot(Neval.T,weight[:,_]), axes=(1,0,2)))
        
        denom = np.transpose( np.dot(Neval.T,weight[:,_]), axes=(1,0,2))
        zero_values = np.isclose(denom,0)
        denom[zero_values]   = 1. # Set 1 1, because we dont want to divide by 0
        displacement_vector /= denom

        #displacement_vector  /= surface.Sintegral[...,_]

        # Relax the displacement at the vertices. The vertex basis functions have a value of 1 here, 
        # so they displace much faster than inner control points. In order to limit the displacement to some extend, we relax them.
        displacement_vector[ 0, 0] *= relax
        displacement_vector[ 0,-1] *= relax
        displacement_vector[-1, 0] *= relax
        displacement_vector[-1,-1] *= relax

        return displacement_vector

    def constrain_vector(self, surface, skipInterface=False, init=False):
        disp_vectors = surface.disp_vectors
        bound_info   = surface.boundary()
        exnode_info  = surface.vertex_dict()
        indices      = surface.grid_indices() # Total indices
        patchID      = surface.patchID

        # Make sure ordering of dict is correct (None keys as last if there is any)
        if None in surface.constraint_vec.keys():
            constraint_vec = {k: v for k, v in surface.constraint_vec.items() if v is not None}
            constraint_vec[None] = surface.constraint_vec[None] 
        else:
            constraint_vec = surface.constraint_vec

        change_all_indices   = np.full(indices.shape[:-1],True)    
        # We have to exclude interface normals if the continuity has already been updated, because they don't necessarily match. 
        # The displacement at the interface is already shifted in the averaged normal direction. No need to further constrain this.
        #if skipInterface:
        interf_indices = np.full(indices.shape[:-1],False) 
        for interface, ivalue in surface.multipatchconn.items(): 
            if ivalue != -1: # we have an interface
                interface_indices, interface_cps = bound_info[interface]
                interf_indices += self._masking_indices(indices, interface_indices)
                    #change_all_indices *= ~interf_indices


        ## Loop over each constraint
        for k, (bound, cons) in enumerate(constraint_vec.items()): # {"umin" : value, "vmin" : value, ...}    
            btype, value   = cons # check type

             
            if bound == None:
                # Let it be allowed to apply multiple constraints,   
                #assert k == len(constraint_vec) - 1, "Something went wrong with ordering the dictionary, this should not have happened!"                                                                                                                                                        
                change_indices = change_all_indices
                #change_indices = np.full(indices.shape[:-1],True) 
            else:

                # check if bound is an extraordinary node representation
                if type(bound) == tuple: # we have an exnode ('umax','vmin') for example
                    bound_indices, bound_cps = exnode_info[bound]
                else:
                    bound_indices, bound_cps = bound_info[bound]

                change_indices= self._masking_indices(indices, bound_indices) # Create mask of total_indices (indices == bound_indices)
                change_all_indices *= ~change_indices # Keep track of the indices that have already been constraint (False) and which have not (True).
            
            if btype == "vector": # value = np.array()
                disp_vectors[change_indices] = np.outer(np.dot(disp_vectors[change_indices], value), value)
            
            elif btype == "plane": # value = dict(normal=np.array(), point=np.array())
                plane_normal = value['normal']
                plane_point  = value['point']
                bound_points = np.transpose(surface.ctrlpts(), axes=(1,0,2))[change_indices]
                # Give warning if points are initially not located in plane, 
                # and we switch to constraining the points to move only parallel to the plane
                bprint = f"patch{surface.patchID}-{(bound or 'None')}"
                if not np.isclose(np.dot(bound_points, plane_normal) - np.dot(plane_normal, plane_point), 0).all() and not (self.warned[bprint] if bprint in self.warned else False):
                    self.warned[bprint] = True
                    #warnings.warn(f"The 'plane' constrain points of patch{surface.patchID}-{bprint} are not (all) located in the specified plane. Switching to parallel constraint.", Warning, stacklevel=2)
                    treelog.user(f"The 'plane' constrain points of {bprint} are not (all) located in the specified plane. Projecting points onto plane first.")
                
                # compute projection vector
                project_vec = np.sum(plane_normal*(plane_point - bound_points), axis=1)[:,_]*plane_normal

                # Remove the plane normal component of the displacement vector to constrain the disp_vector its movement in-plane only.
                disp_vectors[change_indices]  = disp_vectors[change_indices] - np.outer(np.dot(disp_vectors[change_indices], plane_normal), plane_normal)
                disp_vectors[change_indices] += project_vec

            elif btype == "plane-parallel": # value = dict(normal=np.array(), point=np.array())
                plane_normal = value['normal']
                plane_point  = value['point']
                # Give warning if points are initially not located in plane, 
                # and we switch to constraining the points to move only parallel to the plane
                bprint = f"patch{surface.patchID}-{(bound or 'None')}"
                if not np.isclose(np.dot(np.transpose(surface.ctrlpts(), axes=(1,0,2))[change_indices], plane_normal) - np.dot(plane_normal, plane_point), 0).all() and not (self.warned[bprint] if bprint in self.warned else False):
                    self.warned[bprint] = True
                    #warnings.warn(f"The 'plane' constrain points of patch{surface.patchID}-{bprint} are not (all) located in the specified plane. Switching to parallel constraint.", Warning, stacklevel=2)
                    treelog.user(f"The 'plane' constrain points of {bprint} are not (all) located in the specified plane. Only allow parallel displacement.")
                
                # Remove the plane normal component of the displacement vector to constrain the disp_vector its movement in-plane only.
                disp_vectors[change_indices] = disp_vectors[change_indices] - np.outer(np.dot(disp_vectors[change_indices], plane_normal), plane_normal)

            
            elif btype == "normal": # value = "update" or "fixed"
                
                if value == "fixed":
                    #change_indices *= ~interf_indices
                    self.fixed_normals = True # <- useful to know if there is a constraint which uses the 'fixed normals'
                    surface0   = [surf0.clone() for surf0 in self.surfaces0 if surf0.patchID == patchID][0]
                    surface0.refine( *surface.nr_refine ) # refine the appropriate amount
                    #normal_vec = np.transpose( surface0.normals(), axes=(1,0,2))

                    nref = int(np.log2( len(surface0.knots()[0])-1 ))
                    normal_vec = self.fixed_normal_vec[f"ref {nref}"][f"patch{patchID}"]
                    disp_vectors[change_indices] = np.sum(disp_vectors[change_indices]*normal_vec[change_indices],axis=1)[:,_]*normal_vec[change_indices]
                    
                    # if init:
                    #     #change_indices *= ~interf_indices
                    #     normal_vec = np.transpose( surface.normals(), axes=(1,0,2))
                    #     self.disp_normal_fix = normal_vec
                    # else:
                    #     normal_vec = self.disp_normal_fix
                   # raise NotImplementedError("The 'fixed' normal constraint is not implemented")
                elif value == "update":
                    change_indices *= ~interf_indices
                    normal_vec = np.transpose( surface.normals(), axes=(1,0,2))
                    disp_vectors[change_indices] = np.sum(disp_vectors[change_indices]*normal_vec[change_indices],axis=1)[:,_]*normal_vec[change_indices]
                else: 
                    raise NotImplementedError(f"Unknown normal constraint value {value}, use 'fixed' or 'update'")
                
                #disp_vectors[change_indices] = np.sum(disp_vectors[change_indices]*normal_vec[change_indices],axis=1)[:,_]*normal_vec[change_indices]
            else:
                raise ValueError(f"Unknown specified boundary constraint type {btype}")
         

        #normal_cons_vec = self.constraint_vec(surface)
        # #disp_vectors *= np.sum(disp_vectors*normal_cons_vec,axis=2)[:,:,_]*normal_cons_vec # Multiply with constraint vector
        #disp_vectors = np.sum(disp_vectors*normal_cons_vec,axis=2)[:,:,_]*normal_cons_vec # Multiply with constraint vector
        return disp_vectors
    
    def compute_fixed_normals(self, max_refs):
        '''Computes the normal vectors for each patch for several refinements (0=> ... <= max_refs) 
        with respect to the initial geometry.
        '''
        # Crude way is easiest way
        # Make a copy of the surfaces
        surfaces0 = [surf0.clone() for surf0 in self.surfaces0]
        surfaceID = [surf0.patchID for surf0 in self.surfaces0]
        assert (np.argsort(surfaceID) == surfaceID).all(), "The ordering of the surface list is not logical"
        normal_vectors = {} 
        for i, refinements in enumerate(range(max_refs+1)):
            key     = f"ref {i}"
            n_vecs  = {}
            normal_vecs   = np.ndarray((0,3), float)
            normal_coords = np.ndarray((0,3), float)
            idx   = [0]
            shape = []

            normal_count = np.ndarray(0,float) # Array containing the number
            
            # First compute all normal vectors of each individual patch
            for surface0 in surfaces0:
                if i > 0:
                   surface0.refine( 1, 1 ) # Uniform refinement in u- and v-direction
                n_vecs[f'patch{surface0.patchID}'] = surface0.normals() #np.transpose( surface0.normals(), axes=(1,0,2))

            self._multipatch_mapper_init(surfaces0, update=True)

            # Next average the extraordinary nodes first
            exclude  = {f'patch{pid}' : [] for pid in surfaceID }
            for exnode_name, exnode_info in surface0.multipatchexnodes.items():
                nvecs    = np.ndarray((0,3),float)
                for exnode in exnode_info:
                    patchid, node   = exnode # integer and tuple('umax','umin')
                    indices, ctrlpt = surfaces0[patchid].vertex_dict()[node] # Ordering should be important, return
                    exclude[f'patch{patchid}'] += [indices]
                    u,v = indices
                    nvecs = np.concatenate([nvecs, n_vecs[f'patch{patchid}'][u,v][_] ], axis=0)

                nvecs = np.average(nvecs, axis=0)

                # Assign the averaged values
                for exnode in exnode_info:
                    patchid, node   = exnode # integer and tuple('umax','umin')
                    indices, ctrlpt = surfaces0[patchid].vertex_dict()[node] # Ordering should be important, return
                    u,v = indices
                    n_vecs[f'patch{patchid}'][u,v] = nvecs

            # Finally loop over the interfaces (we assume that only 2 patches are attached to an interface)
            for surface0 in surfaces0:
                patch_self = surface0.patchID
                for patch_other, map_info in surface0.multipatch_mapping.items():
                    # Does not matter if we average or evaluate an interface twice, average of an average remains the same
                    indices_self, indices_other = map_info
                    contain_exnode = np.array( sum([ (excl == indices_self).all(1) for excl in exclude[f'patch{patch_self}']]),bool)

                    u_self,  v_self  = indices_self[~contain_exnode].T
                    u_other, v_other = indices_other[~contain_exnode].T
                    nvec  = 0.5*(n_vecs[f'patch{patch_self}'][u_self,v_self] + n_vecs[f'patch{patch_other}'][u_other,v_other])  
                    if len(nvec.shape) == 1:
                        nvec /= np.linalg.norm(nvec, axis=0)
                    else:
                        nvec /= np.linalg.norm(nvec, axis=1)[:,_]

                    n_vecs[f'patch{patch_self}'][u_self,v_self]    = nvec 
                    n_vecs[f'patch{patch_other}'][u_other,v_other] = nvec

                    
            # Transpose the normal vectors
            for pid, normals in n_vecs.items():
                n_vecs[pid] = np.transpose( normals, axes=(1,0,2))

            normal_vectors[key] = n_vecs.copy()


        #     # First compute all normal vectors of each individual patch
        #     for surface0 in surfaces0:
        #         if i > 0:
        #            surface0.refine( 1, 1 ) # Uniform refinement in u- and v-direction
        #         normal_vecs   = np.concatenate( [normal_vecs,   np.transpose( surface0.normals(), axes=(1,0,2)).reshape(-1,3)], axis=0 ) 
        #         normal_coords = np.concatenate( [normal_coords, np.transpose( surface0.ctrlpts(), axes=(1,0,2)).reshape(-1,3)], axis=0 )
        #         idx   += [len(normal_vecs)]   
        #         shape += [surface0.grid_indices().shape[:-1]] 

        #     x_coords, x_index, x_counts = np.unique(normal_coords, return_inverse=True, return_counts=True, axis=0)

        #     # Values in x_counts which are >1 should be averaged with the corresponding indices values
        #     idupl = np.argwhere(x_counts>1)
        #     for dupl in idupl:
        #         i_interf = np.argwhere(dupl == x_index).ravel()
        #         assert len(i_interf)>1, "Something went wrong."
        #         assert np.isclose(np.sum( np.linalg.norm( abs(normal_coords[i_interf] - normal_coords[i_interf][0]), axis=1 )), 0), "This should be zero."
        #         #print(normal_coords[i_interf])
        #         normal_average = np.average(normal_vecs[i_interf], axis=0)
        #         normal_vecs[i_interf] = normal_average/np.linalg.norm(normal_average)# -> Make sure it has unit length again

        #     for surfID, ibound in zip(surfaceID, zip(idx[:-1],idx[1:],shape)):
        #         imin, imax, s = ibound
        #         n_vecs[key][f"patch{surfID}"] = normal_vecs[imin:imax].reshape(*s,3)

        self.fixed_normal_vec = normal_vectors
        return

    @staticmethod
    def _masking_indices(total_indices, subset_indices):
        '''Create a mask of the total indices by checking at which index/argument the subset_indices == total_indices.
        '''
        change_indices = np.full(total_indices.shape[:-1],False)
        if len(subset_indices.shape) != 2:
            subset_indices = subset_indices[_]
        for subseti in subset_indices:     
            change_indices += np.isclose(total_indices - subseti, 0).all(2)
            #change_indices += np.equal(total_indices, subseti).all(2) # Specify which indices/nodes are constraint and should not be changed when applying the None con later on.
        return change_indices
    
    @staticmethod
    def constraint_vec(SPsurface : SplipyS.Surface) -> np.ndarray:
        '''Calculate the constraint vector that specifies in which direction the controlpoint is allowed to displace:
            Normal to the surface is typical, but one can also specify it manually. The manual constraint vectors are 
            applied afterwards (they overwrite the normal option).
        '''
        knotvec_u, knotvec_v = SPsurface.knots()

        CPSu = knotvec_u[:1] + [ i + 0.5*(j-i) for i,j in zip(knotvec_u[:-1],knotvec_u[1:]) ] + knotvec_u[-1:]
        CPSv = knotvec_v[:1] + [ i + 0.5*(j-i) for i,j in zip(knotvec_v[:-1],knotvec_v[1:]) ] + knotvec_v[-1:]

        normals = SPsurface.normal(CPSu,CPSv)
        return np.transpose(normals, axes=(1,0,2)) # Makes it more intuitive: [ [  ]  ]

    @staticmethod
    def whereclose(a, b, rtol=1e-05, atol=1e-08):
        return np.where(np.isclose(a, b, rtol, atol))

    @staticmethod
    def connect_surfaces(surfaces):

        patchIDs = np.array([ surface.patchID for surface in surfaces ]) # Check ordering
        skip = []
        for surface_self in surfaces:
            bound_info_self = surface_self.boundary() # bound_info: dict( 'umin':( indices, cps ), 'umax':( indices, cps ), .. )
            patchID_self = surface_self.patchID

            for bound_self, value_other in surface_self.multipatchconn.items(): # dict( 'umin' : -1, 'umax' : ( patchID, 'umin' ), ..)
                if value_other != -1: # No neighbouring patch at this boundary/bound_self
                    
                    patchID_other, bound_other = value_other
                    patchID_other = np.argwhere( patchID_other == patchIDs ).ravel()[0] # To make sure
                    
                    PatchConn = [patchID_self, patchID_other]
                    if PatchConn in skip or PatchConn[::-1] in skip:
                        continue
                    else:
                        skip.append(PatchConn)

                    surface_other    = surfaces[patchID_other]
                    bound_info_other = surface_other.boundary()

                    bound_indices_self, cps_coords_self   = bound_info_self[bound_self]
                    bound_indices_other, cps_coords_other = bound_info_other[bound_other]
                        
                    

                    Neval_self  = surface_self.Nevalbases        # Size (n,3,3) if nrcps_u=3 and nrcps_v=3
                    weight_self = surface_self.Nintegral #np.sum(surface_self.Nintegral, axis=(1,2))  # Size sum(n,3,3) -> (n)

                    Neval_other  = surface_other.Nevalbases        # Size (n,3,3) 
                    weight_other = surface_other.Nintegral #np.sum(surface_other.Nintegral, axis=(1,2))  # Size sum(n,3,3) -> (n)

                    # Unpack
                    v_self,  u_self  = bound_indices_self.T # Flip u and v, the disp_vectors and Neval arrays are transposed
                    v_other, u_other = bound_indices_other.T

                    # Select only boundary values
                    Neval_self  = Neval_self[:,u_self,v_self] 
                    Neval_other = Neval_other[:,u_other,v_other]
                    disp_self   = surface_self.disp_vectors[u_self,v_self]
                    disp_other  = surface_other.disp_vectors[u_other,v_other]

                    #disp_new  = np.dot(weight_self,Neval_self)[:,_]*disp_self + np.dot(weight_other,Neval_other)[:,_]*disp_other
                    #disp_new /= (np.dot(weight_self,Neval_self)[:,_] + np.dot(weight_other,Neval_other)[:,_])
                    
                    #weight_self_new = np.dot(weight_self,Neval_self)[:,_]/np.sum(np.dot(weight_self,Neval_self)[:,_]) if weight_self.size != 0 and not np.isclose(np.sum(np.dot(weight_self,Neval_self)[:,_]), 0) else np.dot(weight_self,Neval_self)[:,_]
                    #weight_other_new = np.dot(weight_other,Neval_other)[:,_]/np.sum(np.dot(weight_other,Neval_other)[:,_]) if weight_other.size != 0 and not np.isclose(np.sum(np.dot(weight_other,Neval_other)[:,_]), 0) else np.dot(weight_other,Neval_other)[:,_]

                    weight_self_new = np.dot(weight_self,Neval_self)[:,_]
                    weight_other_new = np.dot(weight_other,Neval_other)[:,_]


                    disp_new  = weight_self_new*disp_self + weight_other_new*disp_other
                    sum_weights = weight_self_new + weight_other_new
                    if np.isclose(sum_weights, 0).any(): # If a weight value is equal to 0 make sure we don't divide by it!
                        mask = np.isclose(sum_weights, 0)
                        sum_weights[mask] = 1 # set to 1 (or any other value, does not matter since numerator is 0 anyway)
                    disp_new /= sum_weights
                    #disp_new /=  weight_self_new + weight_other_new if not np.any((np.abs(weight_self_new) + np.abs(weight_other_new)) < 1e-12) else 1
                    

                    surface_self.disp_vectors[u_self,v_self] = disp_new
                    surface_other.disp_vectors[u_other,v_other] = disp_new

        return surfaces

    @staticmethod
    def connect_extraordinary_nodes(surfaces):
        exnodes = surfaces[0].multipatchexnodes # -> Select any surface to get information on extraordinary nodes, it is saved as a multipatch variable (holds/saved for all surfaces = constant)
        for exnode, patch_vertices in exnodes.items(): # exnodes = {'Exnode 0': [(0, ('umax','vmax')), (1, ('umin','vmax')), ..  ] , 'Exnode 1': ...} 
            disp_count = np.zeros((1,3))
            disp_denom = np.zeros((1,))
            # First determine the displacement of the exnode
            for patch_vertex in patch_vertices: 
                patchID, vertex_name = patch_vertex
                surface = surfaces[patchID]
                indices, cps = surface.vertex_dict()[vertex_name] # Returns the indices of that vertex incl. controlpoints (extraordinary node) in tensor format

                v, u = indices.T # Flip u and v, the disp_vectors and Neval arrays are transposed

                #Neval  = surface.Nevalbases[:,u,v]    # Size (n,3,3) 
                #weight = np.sum(surface.Nintegral, axis=(1,2))  # Size sum(n,3,3) -> (n)
                #disp_vec = surface.disp_vectors[u,v]
                #disp_count += np.dot(weight,Neval)*disp_vec
                #disp_denom += np.dot(weight,Neval)

                Neval  = surface.Nevalbases    # Size (n,3,3) 
                weight = surface.Nintegral #np.sum(surface.Nintegral, axis=(1,2))  # Size sum(n,3,3) -> (n)
                weightN = np.transpose( np.dot(Neval.T,weight[:,_]), axes=(1,0,2)) # Specific transpose ensures the ordering is correct again (due to transpose in dot product)
                weightN /= np.sum(weightN) if not np.isclose( np.sum(weightN), 0) else 1
                disp_vec = surface.disp_vectors[u,v]

                disp_count += weightN[u,v]*disp_vec
                disp_denom += weightN[u,v]

            disp_new = disp_count/disp_denom  if not np.isclose(disp_denom,0) else disp_count

            # Next, apply/set the displacement of the exnode for all surfaces
            for patch_vertex in patch_vertices: 
                patchID, vertex_name = patch_vertex
                surface = surfaces[patchID]

                indices, cps = surface.vertex_dict()[vertex_name] # Returns the indices of that vertex incl. controlpoints (extraordinary node) in tensor format
                v, u = indices.T # Flip u and v, the disp_vectors and Neval arrays are transposed

                surface.disp_vectors[u,v] = disp_new
        return surfaces     

    ## TODO: Exclude certain boundaries from G1-continuity
    def continuity(self, surfaces, continuity_relax=1):

        # Calculate the weights of importance (a weight that show the importance of that cps when fitting. 
        # High values show a high importance -> these should be maintained as much as possible, low values are free to move).
        for surface in surfaces:
            Neval  = surface.Nevalbases
            weight = surface.Nintegral #np.sum(surface.Nintegral, axis=(1,2))

            #wN  = np.transpose( np.dot(Neval.T,weight)) # Notice we do not apply a transpose here like in previous functions
            wN  = (np.dot(Neval.T,weight))
            wN /= np.sum(wN) if not np.isclose( np.sum(wN), 0 ) else 1
            surface.wN = wN # Normalized weight value for each cps in the surface that are a measure of how important the control point is w.r.t. the fitting
                            # High values : Important/Influential control point
                            # Low values  : Low importance to fitting      
        
            # set zero valued continuity_vector
            surface.cont_vectors = np.zeros(list(surface.shape)+[3]) # (nrcps_u, nrcps_v, 3)
            surface.cont_counter = np.zeros(list(surface.shape), int) # Counts how many times a cont_vector is stored (used to average at the end)
            surface._cont_counter_prev = np.zeros(list(surface.shape), int) # Different counter for matching/sharing cps between patches. Values should only be assigned once.
        
        # Normalize the weights in each principal direction (= direction that crosses the patch interfaces)
        boundaries = ["umin","umax","vmin","vmax"]
        patchIDs = np.array([ surface.patchID for surface in surfaces ]) # Check ordering
        skip = []
        for surface_self in surfaces:     
            bound_info_self = surface_self.boundary() # bound_info: dict( 'umin':( indices, cps ), 'umax':( indices, cps ), .. )
            patchID_self = surface_self.patchID

            for bound_self, value_other in surface_self.multipatchconn.items(): # dict( 'umin' : -1, 'umax' : ( patchID, 'umin' ), ..)
                if value_other != -1: # No neighbouring patch at this boundary/bound_self
                    
                    patchID_other, bound_other = value_other
                    patchID_other = np.argwhere( patchID_other == patchIDs ).ravel()[0] # To make sure
                    
                    PatchConn = [patchID_self, patchID_other]
                    if PatchConn in skip or PatchConn[::-1] in skip:
                        continue
                    else:
                        skip.append(PatchConn)

                    surface_other    = surfaces[patchID_other]
                    bound_info_other = surface_other.boundary()

                    # Extract adjacent (external) boundaries -> check if these are indeed external
                    extbound_vertx_self = self.external_boundary(surface_self, surface_other, bound_self, bound_other)


                    bound_indices_self, cpsBound_coords_self   = bound_info_self[bound_self]
                    bound_indices_other, cpsBound_coords_other = bound_info_other[bound_other]


                    # Unpack
                    uv_self  = self._extend_indices(bound_indices_self,  bound_self, surfaceType='self') 
                    uv_other = self._extend_indices(bound_indices_other, bound_other, surfaceType='other')

                    # Evaluated normals at the control points
                    normal_self  = surface_self.normals()[bound_indices_self[:,0], bound_indices_self[:,1]]
                    normal_other = surface_other.normals()[bound_indices_other[:,0], bound_indices_other[:,1]]

                    # Ensure normals are pointing in same direction
                    #normal_self  *= -1 if surface_self.flip_normal else 1
                    #normal_other *= -1 if surface_other.flip_normal else 1

                    # Normalize in principal directions
                    for k, (uv_self_principal, uv_other_principal) in enumerate(zip(uv_self, uv_other)):
                        u_self, v_self   = uv_self_principal.T 
                        u_other, v_other = uv_other_principal.T 

                        # High values = Important for fit
                        # Low values = Not important for fit
                        wN_principal = np.array([ surface_self.wN[u_self,v_self][0], 
                                                  np.minimum(2*np.maximum( surface_self.wN[u_self,v_self][1],surface_other.wN[u_other,v_other][0] ), 1), # Twice as important, the boundary cps will moves twice as much
                                                  surface_other.wN[u_other,v_other][1] ])
                        if np.isclose(wN_principal,1).all(): # If the weights are all == 1 (not possible, but just to be sure)
                            continue
                        wPoR = self.get_wPoR(wN_principal)
                        
                        A, B1 = surface_self.ctrlpts()[u_self,v_self]
                        B2, C = surface_other.ctrlpts()[u_other,v_other]
                        assert np.isclose(B1,B2).all(), "Mismatch between patches, this should not have occured!"
                        B = B1 # Or B2

                        # Apply weighting to locate the Point of Rotation (PoR)
                        PoR1 = wPoR[0]*(B-A) + A
                        PoR2 = wPoR[1]*(C-B) + B

                        # A plane is to be constructed through PoR1 and PoR2, however, we still need the normal component
                        # Normal component is based on the out of plane (plane that spans A, B, C) total normal or cont_vec
                        cont_vec = (normal_self[k] + normal_other[k]) 
                        cont_vec /= np.linalg.norm(cont_vec)

                        # Get out of plane part
                        PoRvec = PoR2 - PoR1
                        PoRvec /= np.linalg.norm(PoRvec)

                        normal_plane = cont_vec - np.dot(cont_vec,PoRvec)*PoRvec # Perpendicular to PoRvec
                        normal_plane /= np.linalg.norm(normal_plane) 
                        plane = (PoR1, normal_plane)
                        lineA = (A, cont_vec)
                        lineB = (B,-cont_vec)
                        lineC = (C, cont_vec)

                        newA = self.plane_line_intersection(plane, lineA)
                        newB = self.plane_line_intersection(plane, lineB)
                        newC = self.plane_line_intersection(plane, lineC)

                        #self._multipatch_mapper(patchID_self, uv_self_principal, 'cont_vectors', np.concatenate([ (newA - A)[_], (newB - B)[_] ], axis=0))

                        # cont_vec_self = np.concatenate([ (newA - A)[_], (newB - B)[_] ], axis=0)
                        # cont_vec_other = np.concatenate([ (newB - B)[_], (newC - C)[_] ], axis=0)

                        # Switch ordering slighty, this is 

                        # Check if a) we have an external boundary node or b) they are already aligned, if so no need to further correct them      
                        # if len(extbound_vertx_self) != 0: # First check if we atleast have the possibility of having an external node
                        #    extboudn_there = False
                        # else:
                        #    extboudn_there =  np.equal(bound_indices_self[k], extbound_vertx_self).all(1).any()
                        if np.equal(bound_indices_self[k], extbound_vertx_self).all(1).any() and not np.isclose(np.cross((B-A),(C-B)),0).all(): # Check if we have a node that is also attached to an external boundary      
                            # Apply weighting to locate the Point of Rotation (PoR) for the external boundary continuity
                            PoR1 = wPoR[0]*(newB-newA) + newA
                            PoR2 = wPoR[1]*(newC-newB) + newB

                            # Get 3rd point to construct plane (perpendicular to previous 'plane')
                            XtraP = normal_plane + PoR1 # Can also use + PoR2
                            normal_plane_extb = np.cross(PoR1 - PoR2, PoR1 - XtraP)
                            normal_plane_extb /= np.linalg.norm(normal_plane_extb)
                            plane_extb = (PoR1, normal_plane_extb)

                            # Extract the tangent in the opposite direction as we are working on, if the points A,B are in u-direction, we want the v-idrection tangent and vice versa
                            v_direction_tangent = np.equal(v_self,v_self[0]).all() # True if we want v-direction tangent, false otherwise
                            mask_t = [not v_direction_tangent, v_direction_tangent]
                            tangent_line = np.squeeze(np.array(surface_self.tangents())[mask_t], axis=0)[u_self[-1],v_self[-1]] 
                            tangent_line -= np.dot(tangent_line, normal_plane)*normal_plane # project on plane, we only want to move inside this plane

                            # Project the tangent onto the plane, we only want to move inside the plane otherwise we 'destroy' the main continuity
                            lineA_extb = (newA, tangent_line)
                            lineB_extb = (newB,-tangent_line) # minus is not important, independent of it
                            lineC_extb = (newC, tangent_line)

                            newA_extb = self.plane_line_intersection(plane_extb, lineA_extb) - newA
                            newB_extb = self.plane_line_intersection(plane_extb, lineB_extb) - newB
                            newC_extb = self.plane_line_intersection(plane_extb, lineC_extb) - newC

                        else: # Not external boundary
                            newA_extb = np.zeros(3)
                            newB_extb = np.zeros(3)
                            newC_extb = np.zeros(3)

                        # newA_extb = np.zeros(3)
                        # newB_extb = np.zeros(3)
                        # newC_extb = np.zeros(3)
                        
                        cont_vec_self  = np.concatenate([ (newA - A + newA_extb)[_], (newB - B + newB_extb)[_] ], axis=0)
                        cont_vec_other = np.concatenate([ (newB - B + newB_extb)[_], (newC - C + newC_extb)[_] ], axis=0)

                        surface_self.cont_vectors[u_self, v_self] += cont_vec_self
                        surface_self.cont_counter[u_self, v_self] += 1 # Counts how many times a cont_vector is stored (used to average at the end)

                        surface_other.cont_vectors[u_other, v_other] += cont_vec_other
                        surface_other.cont_counter[u_other, v_other] += 1 # Counts how many times a cont_vector is stored (used to average at the end)

                        # Check for other patches at the boundary ()
                        self._multipatch_mapper_init(surfaces)
                        self._assign_shared_cont_values_multipatch(surfaces, surface_self, indices=uv_self_principal, value=cont_vec_self, exclude=patchID_other) # Assign value of surface_self to shared other patches, exclude surface_other (already done)
                        self._assign_shared_cont_values_multipatch(surfaces, surface_other, indices=uv_other_principal, value=cont_vec_other, exclude=patchID_self) # Assign value of surface_other to shared other patches, exclude surface_self (already done)

                        
                        #assert np.isclose(surface_self.cont_vectors[bound_indices_self[:,0], bound_indices_self[:,1]],
                        #                  surface_other.cont_vectors[bound_indices_other[:,0], bound_indices_other[:,1]]).all(), "The boundary displacements should match!" 


        ## Average the continuity displacement results for each cps
        for surface in surfaces:
            surface.cont_vectors = continuity_relax*surface.cont_vectors / np.fmax(surface.cont_counter[...,_],1)    
            surface.disp_vectors = np.transpose(surface.cont_vectors, axes=(1,0,2))    
        return



    def continuity_v2(self, surfaces, continuity_relax=1):

        # Calculate the weights of importance (a weight that show the importance of that cps when fitting. 
        # High values show a high importance -> these should be maintained as much as possible, low values are free to move).
        for surface in surfaces:
            Neval  = surface.Nevalbases
            weight = surface.Nintegral #np.sum(surface.Nintegral, axis=(1,2))

            #wN  = np.transpose( np.dot(Neval.T,weight)) # Notice we do not apply a transpose here like in previous functions
            wN  = (np.dot(Neval.T,weight))
            wN /= np.sum(wN) if not np.isclose( np.sum(wN), 0 ) else 1
            surface.wN = wN # Normalized weight value for each cps in the surface that are a measure of how important the control point is w.r.t. the fitting
                            # High values : Important/Influential control point
                            # Low values  : Low importance to fitting      
        
            # set zero valued continuity_vector
            surface.cont_vectors = np.zeros(list(surface.shape)+[3]) # (nrcps_u, nrcps_v, 3)
            surface.cont_counter = np.zeros(list(surface.shape), int) # Counts how many times a cont_vector is stored (used to average at the end)
            surface._cont_counter_prev = np.zeros(list(surface.shape), int) # Different counter for matching/sharing cps between patches. Values should only be assigned once.
        
        # Normalize the weights in each principal direction (= direction that crosses the patch interfaces)
        boundaries = ["umin","umax","vmin","vmax"]
        patchIDs = np.array([ surface.patchID for surface in surfaces ]) # Check ordering
        skip = []
        for surface_self in surfaces:     
            bound_info_self = surface_self.boundary() # bound_info: dict( 'umin':( indices, cps ), 'umax':( indices, cps ), .. )
            patchID_self = surface_self.patchID

            for bound_self, value_other in surface_self.multipatchconn.items(): # dict( 'umin' : -1, 'umax' : ( patchID, 'umin' ), ..)
                if value_other != -1: # No neighbouring patch at this boundary/bound_self
                    
                    patchID_other, bound_other = value_other
                    patchID_other = np.argwhere( patchID_other == patchIDs ).ravel()[0] # To make sure
                    
                    PatchConn = [patchID_self, patchID_other]
                    if PatchConn in skip or PatchConn[::-1] in skip:
                        continue
                    else:
                        skip.append(PatchConn)

                    surface_other    = surfaces[patchID_other]
                    bound_info_other = surface_other.boundary()

                    # Extract adjacent (external) boundaries -> check if these are indeed external
                    extbound_vertx_self = self.external_boundary(surface_self, surface_other, bound_self, bound_other)


                    bound_indices_self, cpsBound_coords_self   = bound_info_self[bound_self]
                    bound_indices_other, cpsBound_coords_other = bound_info_other[bound_other]


                    # Unpack
                    uv_self  = self._extend_indices(bound_indices_self,  bound_self, surfaceType='self') 
                    uv_other = self._extend_indices(bound_indices_other, bound_other, surfaceType='other')

                    # Evaluated normals at the control points
                    #normal_self  = surface_self.normals()[bound_indices_self[:,0], bound_indices_self[:,1]]
                    #normal_other = surface_other.normals()[bound_indices_other[:,0], bound_indices_other[:,1]]

                    # Evaluate the cross-boundary tangent at the control points
                    direction_cross_self  = 0 if bound_self[0] == "u" else 1 # u-direction = 0, v-direction = 1
                    direction_cross_other = 0 if bound_other[0] == "u" else 1 # u-direction = 0, v-direction = 1
                    #tang_cross_self  = surface_self.tangents(direction=direction_cross_self)[bound_indices_self[:,0], bound_indices_self[:,1]]
                    #tang_cross_other = surface_other.tangents(direction=direction_cross_other)[bound_indices_other[:,0], bound_indices_other[:,1]]
                    
                    # Make sure the cross-tangents are pointing inwards
                    #tang_cross_self  *= -1 if surface_self.flip_tangent else 1
                    #tang_cross_other *= -1 if surface_other.flip_tangent else 1

                    # Evaluate the parallel-boundary tangent at the control points
                    direction_parall_self  = 1 if bound_self[0] == "u" else 0 # u-direction = 0, v-direction = 1
                    direction_parall_other = 1 if bound_other[0] == "u" else 0 # u-direction = 0, v-direction = 1
                    #tang_parall_self  = surface_self.tangents(direction=direction_parall_self)[bound_indices_self[:,0], bound_indices_self[:,1]]
                    #tang_parall_other = surface_other.tangents(direction=direction_parall_other)[bound_indices_other[:,0], bound_indices_other[:,1]]
                    
                    # Normalize in principal directions
                    for k, (uv_self_principal, uv_other_principal) in enumerate(zip(uv_self, uv_other)):
                        u_self, v_self   = uv_self_principal.T 
                        u_other, v_other = uv_other_principal.T 

                        ## Below, we adopt the following naming convention:
                        
                        # A, B, and c are the control points in 3D space
                        #
                        #
                        #        A               B                C
                        #        o───────────────o────────────────o
                        #   Inner ctrlpt
                        #     Interface       Outer ctrlpt
                        #       self                             other

                        ##-----------------------------------------------

                        ## 1) Get the weighting value of A, B, and C
                        # High values = Important for fit
                        # Low values  = Not important for fit
                        wN_principal = np.array([ surface_self.wN[u_self,v_self][0], 
                                                  np.minimum(2*np.maximum( surface_self.wN[u_self,v_self][1],surface_other.wN[u_other,v_other][0] ), 1), # Twice as important, the boundary cps will moves twice as much
                                                  surface_other.wN[u_other,v_other][1] ])
                        if np.isclose(wN_principal,1).all(): # If the weights are all == 1 (not possible, but just to be sure)
                            continue
                        wPoR = self.get_wPoR(wN_principal)
                        
                        ## 2) Get the projected A, B, and C ctrlpts on their patch surface
                        cps_local_self  = np.transpose(surface_self.cpslocal(tensor=True), axes=(1,0,2))
                        cps_local_other = np.transpose(surface_other.cpslocal(tensor=True), axes=(1,0,2))
                        A,  B1 = surface_self(cps_local_self[u_self,v_self][:,0], cps_local_self[u_self,v_self][:,1], tensor=False)
                        B2, C  = surface_other(cps_local_other[u_other,v_other][:,0], cps_local_other[u_other,v_other][:,1], tensor=False)
                        assert np.isclose(B1,B2).all(), "Mismatch between patches, this should not have occured!"
                        B = B1 # Or B2

                        ## 3) Get the local basis vectors (normal, tangent_u, tangent_v)
                        # Compute surface normals for bound and inner cps
                        normal_self  = surface_self.normals()[u_self,v_self]
                        normal_other = surface_other.normals()[u_other,v_other]

                        # Compute parallel tangents for bound and inner cps
                        tang_parall_self  = surface_self.tangents(direction=direction_parall_self)[u_self,v_self]   
                        tang_parall_other = surface_other.tangents(direction=direction_parall_other)[u_other,v_other]
                    
                        # Compute cross tangent for bound and inner cps
                        tang_cross_self  = surface_self.tangents(direction=direction_cross_self)[u_self,v_self]
                        tang_cross_other = surface_other.tangents(direction=direction_cross_other)[u_other,v_other]
                        # Make sure the cross-tangents are pointing inwards
                        tang_cross_self  *= -1 if surface_self.flip_tangent[bound_self] else 1
                        tang_cross_other *= -1 if surface_other.flip_tangent[bound_other] else 1
    
                        ##----------------------------------------------------------

                        ## 4) Find intersection between (plane[normal_self, tang_parall_self], tang_cross_self) -> P_self (Same for surface_other)
                        normalParall_self  = np.cross( normal_self[0], tang_parall_self[0] )
                        normalParall_self /= np.linalg.norm(normalParall_self)
                        planeParall_self = (A, normalParall_self)
                        lineCross_self   = (B, tang_cross_self[1])

                        normalParall_other  = np.cross( normal_other[1], tang_parall_other[1] )
                        normalParall_other /= np.linalg.norm(normalParall_other)
                        planeParall_other = (C, normalParall_other)
                        lineCross_other   = (B, tang_cross_other[0])
                        
                        P_self  = self.plane_line_intersection(planeParall_self, lineCross_self)
                        P_other = self.plane_line_intersection(planeParall_other, lineCross_other)


                        ## 5) Compute the Points-of-Rotation
                        # Apply weighting to locate the Point of Rotation (PoR)
                        PoR1 = wPoR[0]*(B-P_self)  + P_self
                        PoR2 = wPoR[1]*(P_other-B) + B

                        ## Compute the coplanar plane (i.e. this is the plane in which all three vectors: tang_cross_other[0], tang_cross_self[1], and tang_parall_self should lay)
                        normal_plane  = np.cross( tang_parall_self[1], PoR1 - PoR2 )
                        normal_plane /= np.linalg.norm(normal_plane)
                        # The normal plane should point in same direction as the other normal vectors!
                        normal_plane *= np.sign( np.dot(normal_plane, normal_self[1]) )

                        co_plane = (PoR1, normal_plane)

                        # Get the displacement vectors c1, c2, and c3
                        c1_vec = normal_plane + normal_self[0]
                        c2_vec = normal_plane + normal_other[1]
                        c3_vec = normal_plane

                        line1 = (P_self,  c1_vec) #(A, c1_vec)
                        line2 = (P_other, c2_vec) #(C, c2_vec)
                        line3 = (B, c3_vec)

                        newA = self.plane_line_intersection(co_plane, line1)
                        newB = self.plane_line_intersection(co_plane, line3)
                        newC = self.plane_line_intersection(co_plane, line2)

                        # # This is an older check of v1, just for debugging now
                        # A, B1 = surface_self.ctrlpts()[u_self,v_self]
                        # B2, C = surface_other.ctrlpts()[u_other,v_other]
                        # assert np.isclose(B1,B2).all(), "Mismatch between patches, this should not have occured!"
                        # B = B1 # Or B2

                        # Apply weighting to locate the Point of Rotation (PoR)
                        # PoR1 = wPoR[0]*(B-A) + A
                        # PoR2 = wPoR[1]*(C-B) + B

                        # # A plane is to be constructed through PoR1 and PoR2, however, we still need the normal component
                        # # Normal component is based on the out of plane (plane that spans A, B, C) total normal or cont_vec
                        # cont_vec = (normal_self[k] + normal_other[k]) 
                        # cont_vec /= np.linalg.norm(cont_vec)

                        # # Get out of plane part
                        # PoRvec = PoR2 - PoR1
                        # PoRvec /= np.linalg.norm(PoRvec)

                        # normal_plane = cont_vec - np.dot(cont_vec,PoRvec)*PoRvec # Perpendicular to PoRvec
                        # normal_plane /= np.linalg.norm(normal_plane) 
                        # plane = (PoR1, normal_plane)
                        # lineA = (A, cont_vec)
                        # lineB = (B,-cont_vec)
                        # lineC = (C, cont_vec)

                        # newA = self.plane_line_intersection(plane, lineA)
                        # newB = self.plane_line_intersection(plane, lineB)
                        # newC = self.plane_line_intersection(plane, lineC)

                        #self._multipatch_mapper(patchID_self, uv_self_principal, 'cont_vectors', np.concatenate([ (newA - A)[_], (newB - B)[_] ], axis=0))

                        # cont_vec_self = np.concatenate([ (newA - A)[_], (newB - B)[_] ], axis=0)
                        # cont_vec_other = np.concatenate([ (newB - B)[_], (newC - C)[_] ], axis=0)

                        # Switch ordering slighty, this is 

                        # Check if a) we have an external boundary node or b) they are already aligned, if so no need to further correct them      
                        # if len(extbound_vertx_self) != 0: # First check if we atleast have the possibility of having an external node
                        #    extboudn_there = False
                        # else:
                        #    extboudn_there =  np.equal(bound_indices_self[k], extbound_vertx_self).all(1).any()


                        if np.equal(bound_indices_self[k], extbound_vertx_self).all(1).any() and not np.isclose(np.cross((B-A),(C-B)),0).all(): # Check if we have a node that is also attached to an external boundary      
                            ctrlpt_self  = surface_self.ctrlpts()[u_self,v_self]
                            ctrlpt_other = surface_other.ctrlpts()[u_other,v_other]

                            # Apply weighting to locate the Point of Rotation (PoR) for the external boundary continuity
                            du_A = newA - P_self
                            du_B = newB - B
                            du_C = newC - P_other
                             
                            PoR1 = wPoR[0]*(ctrlpt_self[1] + du_B  - ctrlpt_self[0] - du_A) + ctrlpt_self[0] + du_A
                            PoR2 = wPoR[1]*(ctrlpt_other[1] + du_C - ctrlpt_self[1] - du_B) + ctrlpt_self[1] + du_B

                            # Get 3rd point to construct plane (perpendicular to previous 'plane')
                            XtraP = normal_plane + PoR1 # Can also use + PoR2
                            normal_plane_extb = np.cross(PoR1 - PoR2, PoR1 - XtraP)
                            normal_plane_extb /= np.linalg.norm(normal_plane_extb)
                            plane_extb = (PoR1, normal_plane_extb)

                            # Extract the tangent in the opposite direction as we are working on, if the points A,B are in u-direction, we want the v-idrection tangent and vice versa
                            v_direction_tangent = np.equal(v_self,v_self[0]).all() # True if we want v-direction tangent, false otherwise
                            mask_t = [not v_direction_tangent, v_direction_tangent]
                            tangent_line = np.squeeze(np.array(surface_self.tangents())[mask_t], axis=0)[u_self[-1],v_self[-1]] 
                            tangent_line -= np.dot(tangent_line, normal_plane)*normal_plane # project on plane, we only want to move inside this plane

                            # Project the tangent onto the plane, we only want to move inside the plane otherwise we 'destroy' the main continuity
                            lineA_extb = (newA, tangent_line)
                            lineB_extb = (newB,-tangent_line) # minus is not important, independent of it
                            lineC_extb = (newC, tangent_line)

                            newA_extb = self.plane_line_intersection(plane_extb, lineA_extb) - newA
                            newB_extb = self.plane_line_intersection(plane_extb, lineB_extb) - newB
                            newC_extb = self.plane_line_intersection(plane_extb, lineC_extb) - newC

                        else: # Not external boundary
                            newA_extb = np.zeros(3)
                            newB_extb = np.zeros(3)
                            newC_extb = np.zeros(3)

                        # newA_extb = np.zeros(3)
                        # newB_extb = np.zeros(3)
                        # newC_extb = np.zeros(3)
                        
                        #cont_vec_self  = np.concatenate([ (newA - A + newA_extb)[_], (newB - B + newB_extb)[_] ], axis=0)
                        #cont_vec_other = np.concatenate([ (newB - B + newB_extb)[_], (newC - C + newC_extb)[_] ], axis=0)

                        cont_vec_self  = np.concatenate([ (newA - P_self + newA_extb)[_], (newB - B + newB_extb)[_] ], axis=0)
                        cont_vec_other = np.concatenate([ (newB - B + newB_extb)[_], (newC - P_other + newC_extb)[_] ], axis=0)


                        surface_self.cont_vectors[u_self, v_self] += cont_vec_self
                        surface_self.cont_counter[u_self, v_self] += 1 # Counts how many times a cont_vector is stored (used to average at the end)

                        surface_other.cont_vectors[u_other, v_other] += cont_vec_other
                        surface_other.cont_counter[u_other, v_other] += 1 # Counts how many times a cont_vector is stored (used to average at the end)

                        # Check for other patches at the boundary ()
                        self._multipatch_mapper_init(surfaces)
                        self._assign_shared_cont_values_multipatch(surfaces, surface_self, indices=uv_self_principal, value=cont_vec_self, exclude=patchID_other) # Assign value of surface_self to shared other patches, exclude surface_other (already done)
                        self._assign_shared_cont_values_multipatch(surfaces, surface_other, indices=uv_other_principal, value=cont_vec_other, exclude=patchID_self) # Assign value of surface_other to shared other patches, exclude surface_self (already done)

                        
                        #assert np.isclose(surface_self.cont_vectors[bound_indices_self[:,0], bound_indices_self[:,1]],
                        #                  surface_other.cont_vectors[bound_indices_other[:,0], bound_indices_other[:,1]]).all(), "The boundary displacements should match!" 


        ## Average the continuity displacement results for each cps
        for surface in surfaces:
            surface.cont_vectors = continuity_relax*surface.cont_vectors / np.fmax(surface.cont_counter[...,_],1)    
            surface.disp_vectors = np.transpose(surface.cont_vectors, axes=(1,0,2))    
        return
    

    @staticmethod
    def external_boundary(surface_self, surface_other, bound_self, bound_other):
        # Check if the considered boundary is attached to an external boundary
        # Extract vertex information of the two patches
        extbound_vertx_self=[]
        for vertex_self, vertex_info_self in surface_self.vertex_dict().items():
            idx_self, cps_self = vertex_info_self
            for vertex_other, vertex_info_other in surface_other.vertex_dict().items():
                idx_other, cps_other = vertex_info_other
                if np.isclose(cps_self,cps_other).all(): # Find matching vertex at the bound
                    bound_adj_self = list(vertex_self) # The possible external boundary self
                    bound_adj_self.remove(bound_self)  # Remove the current boundary bound_self
                    bound_adj_other = list(vertex_other) # The possible external boundary other
                    bound_adj_other.remove(bound_other) # Remove the current boundary bound_other
                    bndvself  = surface_self.multipatchconn[bound_adj_self[0]]
                    bndvother = surface_other.multipatchconn[bound_adj_other[0]]
                    if bndvself == -1 and bndvother == -1: # Check if the two are external boundaries
                        extbound_vertx_self.append(idx_self) # Could also save idx_other; save index which is the vertex at an external boundary
       
        extbound_vertx_self = np.array(extbound_vertx_self)
        if len(extbound_vertx_self) == 0:
            return np.array([[-1,-1]]) # If there are no ext bound, return a list of indices that don't exist (returning an empty list raises an error later on)
        else:
            return extbound_vertx_self if len(extbound_vertx_self.shape)==2 else extbound_vertx_self[_] # list of maximal 2 indices that correspond to vertices of the surface_self which are attached to an external boundary and bound_self           

    def continuity_error(self, surfaces, return_spatial=False, nsample=10):
        '''The continuity error is computed as the integral of the angle (α) squared over the interface (Γ_i):

        ∫( α**2 )dΓ_i,

        with

        α = arccos[ a*b / ( |a| |b| ) ].

        '''
        indices = {"umin": (np.array([0,0]),np.array([0,1])), # u and v coordinates/vertices of that boundary (u,v)
                   "umax": (np.array([1,1]),np.array([0,1])),
                   "vmin": (np.array([0,1]),np.array([0,0])),
                   "vmax": (np.array([0,1]),np.array([1,1]))}
        patchIDs = np.array([ surface.patchID for surface in surfaces ]) # Check ordering
        skip      = []
        error_abs = 0
        dL        = 0
        if return_spatial:
            Espatial = np.ndarray((0),float) # spatial error array
            Xspatial = np.ndarray((0,3),float) # spatial coordinates array
        for surface_self in surfaces:     
            bound_info_self = surface_self.boundary() # bound_info: dict( 'umin':( indices, cps ), 'umax':( indices, cps ), .. )
            patchID_self = surface_self.patchID

            for bound_self, value_other in surface_self.multipatchconn.items(): # dict( 'umin' : -1, 'umax' : ( patchID, 'umin' ), ..)
                if value_other != -1: # No neighbouring patch at this boundary/bound_self
                    
                    patchID_other, bound_other = value_other
                    patchID_other = np.argwhere( patchID_other == patchIDs ).ravel()[0] # To make sure
                    
                    PatchConn = [patchID_self, patchID_other]
                    if PatchConn in skip or PatchConn[::-1] in skip:
                        continue
                    else:
                        skip.append(PatchConn)

                    u_self, v_self = indices[bound_self]
                    u_other, v_other = indices[bound_other]

                    usample_self = np.linspace(u_self[0],u_self[1],nsample)
                    vsample_self = np.linspace(v_self[0],v_self[1],nsample)
                    usample_other = np.linspace(u_other[0],u_other[1],nsample)
                    vsample_other = np.linspace(v_other[0],v_other[1],nsample)

                    normal_self  = surfaces[patchID_self].normal(usample_self, vsample_self, tensor=False)
                    normal_other = surfaces[patchID_other].normal(usample_other, vsample_other, tensor=False)
                    normal_self  *= -1 if surfaces[patchID_self].flip_normal else 1
                    normal_other *= -1 if surfaces[patchID_other].flip_normal else 1

                    X_self  = surfaces[patchID_self](usample_self, vsample_self, tensor=False)
                    X_other = surfaces[patchID_other](usample_other, vsample_other, tensor=False)


                    # du_sample_self = np.insert( np.fmin(1,usample_self + 0.5*(usample_self[1]-usample_self[0])), 0, usample_self[0])
                    # dv_sample_self = np.insert( np.fmin(1,vsample_self + 0.5*(vsample_self[1]-vsample_self[0])), 0, vsample_self[0])
                    # du_sample_other = np.insert( np.fmin(1,usample_other + 0.5*(usample_other[1]-usample_other[0])), 0, usample_other[0])
                    # dv_sample_other = np.insert( np.fmin(1,vsample_other + 0.5*(vsample_other[1]-vsample_other[0])), 0, vsample_other[0])
                    # X_self  = surfaces[patchID_self](du_sample_self, dv_sample_self, tensor=False)
                    # X_other = surfaces[patchID_other](du_sample_other, dv_sample_other, tensor=False)

                    if not np.isclose(np.sum( (X_self - X_other)**2 ), 0):
                        assert np.isclose(np.sum( (X_self - X_other[::-1])**2 ), 0), "Coordinates mismatch, this should not have occcured"
                        # FLip because ordering is wrong way
                        X_other = X_other[::-1]
                        normal_other = normal_other[::-1]


                    α  = np.arccos( np.fmin( 1 - 1e-10, np.sum(normal_self*normal_other, axis=1) / ( np.linalg.norm(normal_self, axis=1) * np.linalg.norm(normal_other, axis=1) ) ))# bound by tol for floating point errors   
                    dX = np.linalg.norm(np.diff(X_self,axis=0),axis=1)
                    dα = α[:-1] + 0.5*np.diff(α)
                    #error += np.sqrt( np.dot(np.degrees(dα)**2, dX) )
                    error_abs += np.dot( np.abs( np.degrees(dα) ), dX) 
                    dL += np.sum(dX) # Length of the interface
                    #error += np.sqrt(np.sum( np.degrees(α)**2 )) / len(α)

                     
                    
                    if return_spatial:
                       Espatial = np.concatenate([Espatial, np.abs(np.degrees(α))])
                       Xspatial = np.concatenate([Xspatial, X_self])

        error = error_abs / dL # Averaged absolute continuity error
        #error = ( np.pi/32 ) / np.tan(np.radians( 90 - 0.5*error ))
        #error = 0.0625*( np.pi/32 ) / np.tan(np.radians( 90 - 0.5*error )) # scaled angle numerator = ( pi/8 ), used to scal to diameter quant; 
                                                                    # denomerator, used to scale degree to usable length
        return (error, {"Spatial continuity error": Espatial, "Spatial interface coordinates": Xspatial}) if return_spatial else error

    @staticmethod
    def _multipatch_mapper_init(surfaces, update=False):
        ids = [surf.patchID for surf in surfaces]
        for surface in surfaces:
            self = surface 

            if hasattr(self, '_cont_counter_prev'):
                self._cont_counter_prev = np.minimum(self._cont_counter_prev,0) # Reset previous counter (initialize)
            else:
                np.zeros(list(surface.shape), int) # Initialize the counter

            if update or not hasattr(self, 'multipatch_mapping'): # update or construct/assign value to multipatch_mapping
                self.multipatch_mapping = {}

                # a) First specify the extraordinary node indices:
                for exnode_self, exnodes_other_info in self.exnodes.items(): # { ("umax","vmax") : ( (1, ("umax","vmin") ), (2, ("umin","vmin") )  ) } 
                    for exnode_other_info in exnodes_other_info:
                        patchID_other, exnode_other = exnode_other_info
                        other = surfaces[np.where(patchID_other == ids)[0][0]] 
                        self.multipatch_mapping[patchID_other] = (self.vertex_dict(indexOnly=True)[exnode_self][_], # create newaxis, useful for later on
                                                                  other.vertex_dict(indexOnly=True)[exnode_other][_] ) # Indices of: (self, other)
                # b) Second, specify the boundary indices

                for bound_self, info_other in self.multipatchconn.items(): # dict( "umin"=(0, "vmax"), "umax"=-1, ..)
                    if info_other == -1: # No adjacent patch
                        continue
                    patchID_other, bound_other = info_other
                    indices_other, cps_other = self.boundary()[bound_other] # dict( 'umax' = ( indices, cps ), ... )
                    indices_self, cps_self = self.boundary()[bound_self]        

                    if type(self.multipatch_mapping[patchID_other]) != tuple:
                        self.multipatch_mapping[patchID_other] = (indices_self, indices_other)
                    else:
                        iself, iother = self.multipatch_mapping[patchID_other]
                        # c) Finally, remove duplicates, make sure both in self and other (self, other)_indices
                        # Following is a sanity double check for the unique function of numpy (not most efficient):
                        conc_self  = np.concatenate([iself, indices_self])
                        conc_other = np.concatenate([iother, indices_other])
                        indices_self_new, _idx_self = np.unique( conc_self, axis=0, return_index=True)
                        indices_other_new, _idx_other = np.unique( conc_other, axis=0, return_index=True)
                        assert (len(indices_self_new) == len(indices_other_new)), "Mismatch in unique indices, this should never happen!"
                        self.multipatch_mapping[patchID_other] = (conc_self[_idx_self], conc_other[_idx_other])
        return

    @staticmethod
    def _assign_shared_cont_values_multipatch(surfaces, ref_surface, indices=np.array([]), value=np.array([]), exclude=None):
        patchIDs = [surf.patchID for surf in surfaces] # list of patch ids
        assert len(indices)==len(value), "Mismatch in number of indices and values."

        for patchID_other, indices_info in ref_surface.multipatch_mapping.items(): # dict( patchID_other : ( indices_self, indices_other ), patchID_other : (..) )
            if patchID_other == exclude:
                continue
            indices_self, indices_other = indices_info
            patchID_other = np.argwhere( patchID_other == patchIDs ).ravel()[0] # To be sure, not relying on list ordering

            
            # Check if the given indices are part of the indices_self array, if so store the corresponding indices
            for row_indices, row_value in zip(indices, value):
                row_mask = np.equal( row_indices, indices_self ).all(1)
                if row_mask.any(): # If there is a match
                    u_other, v_other = indices_other[row_mask].T # Get the corresponding indices of the other patch

                    if surfaces[patchID_other]._cont_counter_prev[u_other, v_other] == 1: # Skip if we already specified this node/cps
                        continue

                    surfaces[patchID_other].cont_vectors[u_other, v_other] += row_value
                    surfaces[patchID_other].cont_counter[u_other, v_other] += 1
                    surfaces[patchID_other]._cont_counter_prev[u_other, v_other] += 1
        return

    @staticmethod
    def _extend_indices(indices, bound, tensor=True, surfaceType='self'): # Indices should be in tensor format (nrcps,u,v), 
        # surfaceType = 'self' or 'other':
        # self  : Indices go from inside patch to outside/boundary
        # other : Indices go from boundary/outside to inside of patch
        assert surfaceType in ('self','other'), "unknown surfaceType when extending the indices, use 'self' or 'other'"
        if bound == "umin":
            shift = indices.copy()
            shift[:,0] += 1
            idx = [shift, indices] if surfaceType=='self' else [indices, shift] 
        elif bound == "umax":
            shift = indices.copy()
            shift[:,0] -= 1
            idx = [shift, indices] if surfaceType=='self' else [indices, shift]
        elif bound == "vmin":
            shift = indices.copy()
            shift[:,1] += 1
            idx = [shift, indices] if surfaceType=='self' else [indices, shift]
        elif bound == "vmax":
            shift = indices.copy()
            shift[:,1] -= 1
            idx = [shift, indices] if surfaceType=='self' else [indices, shift]
        else:
            raise ValueError(f"Unknown boundary name '{bound}'")
        


        if tensor:
            idx = [i[:,_] for i in idx]
            return np.concatenate(idx, axis=1)
        else:
            return np.concatenate(idx, axis=0)


    @staticmethod # Determine the points of rotation PoR
    def get_wPoR(weights): # return the weights of the Point of Rotation (PoR)
        if np.isclose(weights,1).all(): #All weights should not be moved
            return np.array([ weights[1]/(weights[1] + weights[0]), weights[2]/(weights[1] + weights[2]) ])
        elif np.isclose(weights[:-1],1).all(): # If the first 2 are equal to 1
            return np.array([0, 0])
        elif np.isclose(weights[-2:],1).all(): # If the last 2 are equal to 1
            return np.array([1, 1])
        elif np.isclose(weights[[0,2]],0).all(): # The first and last values are equal to 0
            return np.array([0.999, 0.001])
        else:
            if np.isclose([weights[1], weights[2]], 0).all():
                term2 = 0.5
            else:
                term2 = weights[2]/(weights[1] + weights[2])
            if np.isclose([weights[1], weights[0]], 0).all():
                term1 = 0.5
            else:
                term1 = weights[1]/(weights[1] + weights[0]) 
            return np.array([ term1, term2 ])
    
    @staticmethod
    def plane_line_intersection(plane, line): # plane = ( point on plane np.ndarray, normal vector np.ndarry )
        point_p, vec_p = plane # vec_p is the normal of the plane
        point_l, vec_l = line
        assert np.dot(vec_l, vec_p), "There is no intersection, this should never happen."
        d = np.dot( point_p - point_l, vec_p ) / np.dot( vec_l, vec_p )
        return point_l + d*vec_l

    @staticmethod
    def total_kinetic_energy(surfaces, average=False):
        kinE = 0
        ncps = 0
        for surface in surfaces:
            ncps += sum(surface.nrcps())
            kinE += 0.5*surface.dx**2
        if average:
            return kinE/ncps
        else:
            return kinE
        
class DataPointSet:

    def __init__(self, datapoints, surface_points, surface_points_index, surface_index=np.ndarray(0)):
        '''A class used to quickly evaluate data point information (from the data point perspective).
        
        Input:
        _____________
        datapoints           : np.ndarray((n,3), floats); Numpy array of all the datapoints.
        surface_points       : np.ndarray((n,3), floats); Numpy array of all the surface points that correspond to the datapoints array (nearest neighbour).
        surface_points_index : np.ndarray((n,), int);     Numpy array containing the indices of the total_surface_points array that corresponds to the surface_points array.
        surface_index        : np.ndarray((n,), int);     Numpy array containing the index of the surface to which the datapoint is closest located to.

        '''

        assert len(datapoints) == len(surface_points), "Number of datapoints should be equal to the number of surface points (one-to-one)"

        # Store the input
        self._datapoints = datapoints
        self._surface_points = surface_points
        self._surface_points_index = surface_points_index
        if len(surface_index) != 0:
            self._surface_index  = surface_index 
        self._error_vector, self._error = compute_difference( surface_points, datapoints, returnMagnitude=True )    
        return

    def __call__(self, patchID):
        #assert patchID <= np.max(self._surface_index), "PatchID should be lower than maximum number of patches"
        idx = np.where( self._surface_index == patchID)
        return DataPointSet( self._datapoints[idx], self._surface_points[idx], self._surface_points_index[idx]  )

    def __len__(self,):
        return len(self._datapoints)
    
    def update_surfacepoints(self, surface_points, indices):
        self._surface_points[indices] = surface_points
        self._error_vector, self._error = compute_difference( self._surface_points, self._datapoints, returnMagnitude=True ) # Points in the direction of the datapoint
        return
        
    def datapoints(self, index=None):
        if index is not None:
            return self._datapoints[index]
        else:
            return self._datapoints
    
    def surface_index(self, index=None):
        if index is not None:
            return self._surface_index[index]
        else:
            return self._surface_index
        
    def error_vector(self, index=None):
        if index is not None:
            return self._error_vector[index]
        else:
            return self._error_vector
        
    def error(self, index=None):
        if index is not None:
            return self._error[index]
        else:
            return self._error

    def total_error(self,): # Compute L2 error
        return np.linalg.norm(self._error)/len(self._error) # Scale with the length   
        
    def average_diameter(self,):
        xcenter = np.average(self._datapoints, axis=0)
        length  = np.average( np.linalg.norm(self._datapoints - xcenter, axis=1))
        return 2*length # typical length = diameter

def compute_difference(x, y, returnMagnitude=False): # Return difference vector pointing from x to y
    dx = y - x
    if returnMagnitude:
        if len(dx.shape) == 1:
            dx = dx[_]
        return dx, np.linalg.norm(dx, axis=1)
    else:
        return dx
    