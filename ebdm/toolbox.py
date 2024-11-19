import numpy as np
import splipy as sp
from vtnurbs.construct import MultipatchSurface
import splipy.surface as SplipyS
from typing import Tuple, List
import treelog
_ = np.newaxis

# Initialization functions for the EBDM
def init_multipatch_surfaces(multipatch_surface : MultipatchSurface, constraints : dict() = {}) -> List[SplipyS.Surface]: 
    '''Convert a vtnurbs MultipatchSurface into a list of SpliPy Surface Objects(). 
        Note: the MultipatchSurface is always rational or NURBS (B-spline is obtained when setting weights=1).

       Input:___________
       multipatch_surface : MultipatchSurface;  The multipatch surface that is to be converted into a list of SpliPy.Surface Objects().
       constraints        : Dict(); The constraint dictionary specifying which control points are allowed to move only in a specific direction
                                    constraints.keys() -> "Boundary name" : Specify which boundary you want to constrain
                                                       -> None            : Specify None if you want to constraint all control points
                                    constraints.item() -> tuple(type, value)
                                                                -> type  = "vector"     (specific 3D vector constraint)
                                                                   value = np.ndarray()    
                                                                -> type  = "plane"      (plane in which the control point is allowed to move)
                                                                   value = dict('normal' : np.ndarray for normal vector of plane,
                                                                                'point'  : np.ndarray for point on plane)   
                                                                -> type  = "normal"     (normal direction with respect to the initial surface)
                                                                   value = "fixed" or "update" (Either fix the normal, equal to the normals in the initial surface,
                                                                                                or set 'update' to make it equal to the fitted/changing surface)                               
       Returns:___________
       surface_NURBS_list:  list( SplipyS.Surface ); Contains a list of SpliPy Surface Objects() corresponding to each patch of the MultipatchSurce.
    '''

    surface_NURBS_list   = []
    # check_constraints(constraints, multipatch_surface._bnames)
    patch_conn, _extrnode, _extrnodeppatch = multipatch_connectivity(multipatch_surface.patchcon()) # Topology dependent = calculate only ones  
    bound_map = nutils_splipy_boundary_mapp() # Links/maps the local nutils naming convention of the boundaries to the splipy naming convention
    # First redefine all surfaces to make it suitable in the NURBS pyton toolbox and therefore the EBDM
    for patchID, patch_surface in enumerate(multipatch_surface): # Loop over the individual patches of the multipatch domain
        
        surface_nurbs = create_nurbs_surface(patch_surface) # SpliPy surface
        surface_nurbs.patchID   = patchID # Store which patchID this is related to
        surface_nurbs.nr_refine = (0,0) # Number of refinements in u- and v-direction. Initialized at 0, updated during the EBDM iterations.
        #surface_nurbs.constraint_vec = constraint_vec(surface_nurbs) # Normal vectors corresponding to the cps
        surface_nurbs.multipatchconn = patch_conn[f"patch{patchID}"] # Multipatchpatch connectivity
        surface_nurbs.multipatchexnodes = _extrnode
        surface_nurbs.exnodes  = _extrnodeppatch[f"patch{patchID}"] # The extraordinary node (refers to a vertex) of that patch and the other surface vertices connected to it
        surface_nurbs.constraint_vec = assign_constraints(patchID, bound_map, constraints, multipatch_surface._bnames, patch_conn[f"patch{patchID}"], _extrnode) # Assign the constraints specified by the user to the local boundaries of the splipy surface

        # Bound new methods to the object
        surface_nurbs.sample       = sample.__get__(surface_nurbs)
        surface_nurbs.vertex       = vertex.__get__(surface_nurbs)
        surface_nurbs.vertex_dict  = vertex_dict.__get__(surface_nurbs)
        surface_nurbs.nrcps        = nrcps.__get__(surface_nurbs) 
        surface_nurbs.grid_indices = grid_indices.__get__(surface_nurbs)
        surface_nurbs.boundary     = boundary.__get__(surface_nurbs)
        surface_nurbs.normals      = normals.__get__(surface_nurbs)
        surface_nurbs.tangents     = tangents.__get__(surface_nurbs)
        surface_nurbs.ctrlpts      = ctrlpts.__get__(surface_nurbs)
        surface_nurbs.weights      = weights.__get__(surface_nurbs)
        surface_nurbs.cpslocal     = cpslocal.__get__(surface_nurbs)
        surface_nurbs._gss         = _gss.__get__(surface_nurbs)
        surface_nurbs.maximum_basis = maximum_basis.__get__(surface_nurbs)
        
        #surface_nurbs.multipatchmapper = multipatchmapper.__get__(surface_nurbs)

        # Expand the list with SpliPy Surface() objects    
        surface_NURBS_list.append(surface_nurbs)

    evaluate_normals(surface_NURBS_list, patch_conn) # Check if all normals are pointing in the correct/same direction (Important!)
    evaluate_tangents(surface_NURBS_list) # Check if the cross-boundary tangents are pointing inwards or not
    return surface_NURBS_list

def update_multipatch_object(multipatch_surface : MultipatchSurface, surfaces : List[SplipyS.Surface], surface_type="surface") -> MultipatchSurface:
    cps = np.ndarray((0,3), float)
    w   = np.ndarray((0), float)

    connectivity   = multipatch_surface.patchcon()
    vertices       = multipatch_surface.patchverts().astype(float)
    boundary_names = multipatch_surface._bnames
    vertex_names = [("umin","vmin"), ("umax","vmin"), ("umin","vmax"), ("umax","vmax")] # ordered according to nutils connectivity ordering

    nelems  = {}
    pnelems = {}
    for surface in surfaces:
        patchID = surface.patchID

        # extract vertices of surface
        vertices[ connectivity[patchID] ] = np.array([surface.vertex_dict()[vertex][1] for vertex in vertex_names])

        # Set the control point values
        cps = np.concatenate([cps, np.transpose(surface.ctrlpts(), axes=(1,0,2)).reshape(-1,3) ])
        w   = np.concatenate([w, np.transpose(surface.weights(), axes=(1,0)).ravel()])

        ncps_u, ncps_v = surface.nrcps()
        nelems_u = ncps_u - 2 # if we use quadratic
        nelems_v = ncps_v - 2 # if we use quadratic
        pnelems[f"patch{patchID}"] = (nelems_u, nelems_v)

        conres = connectivity[patchID].reshape([2,2])
        nelems[tuple(conres[0,:])] = nelems_u # if we use quadratic
        nelems[tuple(conres[1,:])] = nelems_u 
        nelems[tuple(conres[:,0])] = nelems_v 
        nelems[tuple(conres[:,1])] = nelems_v 

    return MultipatchSurface("MultiPatchSurface", surface_type, vertices, connectivity, cps, w, nelems, pnelems)# bnames=boundary_names)
    
# Saves a single SpliPy surface to a single VTK file, multiple surfaces are saved in multiple files
def splipy_save_vtk(surface, filename, control_net=False, nsample=20):
    from nutils import mesh, export
    patchcon = np.array([[0,1,2,3]]) # Connectivity
    cps = surface.ctrlpts()
    idx = np.array([ [0,0], [-1,0], [0,-1], [-1,-1] ])
    patchverts = cps[idx[:,0], idx[:,1]]
    weights = surface.weights()


    
    dict_keys = [(0,1), (2,3), (0,2), (1,3)]
    order    = surface.order()
    degree   = [i - 1 for i in order]
    knotval_wmult  = surface.knots(with_multiplicities=True)

    knotval  = []
    knotmult = []
    for knotvalm in knotval_wmult:
        knotval_, knot_mult_ = np.unique( knotvalm, return_counts=True )
        knotval  += [knotval_[np.argsort(knotval_)].tolist()]
        knotmult += [knot_mult_[np.argsort(knot_mult_)].astype(int).tolist()]
    nelems   = [len(knotv) - ordr + 2 for knotv, ordr in zip(knotval, order)]

    # convert to dicts
    nelems   = { key : nelems[i] for key, i in zip(dict_keys, (0,0,1,1)) }
    knotmult = { key : knotmult[i] for key, i in zip(dict_keys, (0,0,1,1)) }
    knotval  = { key : knotval[i] for key, i in zip(dict_keys, (0,0,1,1)) }

    topo, lingeom = mesh.multipatch(patches=patchcon, patchverts=patchverts, nelems=nelems)

    bsplines = topo.basis('spline', degree=degree, patchcontinuous=False, knotvalues=knotval, knotmultiplicities=knotmult)

    weight   = bsplines.dot(weights.ravel())
    geom     = bsplines.vector(3).dot((cps*weights).ravel())/weight

    bezier = topo.sample('bezier', nsample)
    X = bezier.eval(geom)
    export.vtk(filename, bezier.tri, X)

    if control_net:
        nrcps_u, nrcps_v = cps.shape[:-1]
        id_u, id_v = np.meshgrid( range(nrcps_u), range(nrcps_v) )

        id_cps   = np.linspace(0,nrcps_u*nrcps_v-1,nrcps_u*nrcps_v).astype(int).reshape(nrcps_u, nrcps_v)
        row_conn = np.concatenate([_conc_conn(id_u)[...,_],   _conc_conn(id_v)[...,_]],axis=2)
        col_conn = np.concatenate([_conc_conn(id_u.T)[...,_], _conc_conn(id_v.T)[...,_]],axis=2)

        conn = np.concatenate([row_conn, col_conn], axis=0)
        CONN = np.ndarray((0,2),int)
        for c in conn:
            c = c.T
            CONN = np.concatenate([CONN, id_cps[c[0],c[1]][_]],axis=0)

        export.vtk( filename + " controlnet", CONN, cps.reshape(-1,3))
    return

def _conc_conn(x): # concatenates the columns
    conn = np.ndarray((0,2),int)
    for col in range(x.shape[1]-1):
        conn = np.concatenate([ conn, np.concatenate([x[:,col,_], x[:,col+1,_]],axis=1)], axis=0)
    return conn

#TODO set assertions to check the constraints dict
# def check_constraints(constraints, bnames_nutils):
#     for bound, value in constraints.items():
#         if bound == None:
#             if 
#             check_plane_cons(value)
#     return

def create_bspline_surface(PatchSurface : MultipatchSurface) -> SplipyS.Surface:
    from warnings import warn
    warn("B-splines are not supported, use NURBS with weights=1", DeprecationWarning, stacklevel=2)
    cps = PatchSurface.cps()
    knotvec = PatchSurface.knotvec()
    order = 3 # This is fixed
    basis_u = sp.BSplineBasis(order=order, knots=knotvec[0])
    basis_v = sp.BSplineBasis(order=order, knots=knotvec[1])
    surface = sp.Surface(basis_u, basis_v, cps)
    return surface


def create_nurbs_surface(PatchSurface : MultipatchSurface) -> SplipyS.Surface:
    '''Create a NURBS SpliPy Surface Object() based on a vtnurbs patch surface Object().'''

    cps = PatchSurface.cps()
    w   = PatchSurface.weights()
    knotvec = PatchSurface.knotvec()
    order = 3 # This is fixed

    cpsh = convert_to_weighted_coords(cps, w)
    basis_u = sp.BSplineBasis(order=order, knots=knotvec[0])
    basis_v = sp.BSplineBasis(order=order, knots=knotvec[1])

    surface = sp.Surface(basis_u, basis_v, cpsh, rational=True)

    return surface

def convert_from_weighted_coords(cpsh : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Convert a weighted (4D) controlpoint aray into a non-weighted (3D) cps and weights (1D) array.'''
    assert cpsh.shape[-1] == 4, "Weighted controlpoint array should be 4D"
    weight = cpsh[...,-1]
    cps = cpsh[...,:-1]/weight
    return cps, weight

def convert_to_weighted_coords(cps : np.ndarray, weight : np.ndarray) -> np.ndarray:
    '''Convert a regular controlpoint (3D; shape=(n,3)) and weights (1D; shape=(1,)) array into a weighted (4D) controlpoint array.'''
    assert cps.shape[-1] == 3, "Controlpoint array should be 3D"
    assert len(cps) == len(weight), "Controlpoint and weigts array should have equal lengths"
    cpsh = np.c_[cps*weight[:,_], weight[:,_]]
    return cpsh

# def constraint_vec(SPsurface : SplipyS.Surface) -> np.ndarray:
#     '''Calculate the constraint vector that specifies in which direction the controlpoint is allowed to displace:
#         Normal to the surface is typical, but one can also specify it manually.
#     '''
#     knotvec_u, knotvec_v = SPsurface.knots()

#     CPSu = knotvec_u[:1] + [ i + 0.5*(j-i) for i,j in zip(knotvec_u[:-1],knotvec_u[1:]) ] + knotvec_u[-1:]
#     CPSv = knotvec_v[:1] + [ i + 0.5*(j-i) for i,j in zip(knotvec_v[:-1],knotvec_v[1:]) ] + knotvec_v[-1:]

#     normals = SPsurface.normal(CPSu,CPSv)
#     return normals

def multipatch_connectivity(connectivity : np.ndarray) -> dict():
    '''Specify the multipatch connectivity in terms of a dictionary that has the patches as keys: 'patch0', 'patch1', ..
         and links the individual SpliPy Surface Objects by means of boundary name indication: umin, umax, vmin, vmax.

       Input:
       connectivity: np.ndarray; Connectivity array of the multipatch surface domain, size: (p,4), with p the nr. of patches.

       Output:
       pbound: dict( str = dict( str=(int, str), str=int, ... )); Dictionary containing the connectivity of the multipatch surface domain, 
                                                            it has the following structure: dict( 'patch1' = dict( "umin"=(0, "vmax"), "umax"=-1, ..)).
                                                            The keys of the first dict() denote the current patch, ('patch0','patch1',..), the valued dict() contains 
                                                            the boundary names of the current 'patch1' as it keys: 'umin', 'umax', 'vmin', 'vmax', and the value is 
                                                            either a -1 (which says that there is no adjacent patch) or a tuple( patchID, boundary_name of patchID ). 
                                                            The patchID is a different patch than 'patch1' and is adjacent/connected to it via its boundary_name. 
    '''
    ndims = min( len(connectivity[0]) // 2, 3) # dimension, curve, surface or solid
    assert ndims == 2, "Only supports two-dimensional parametric domains (surfaces in 3D)"
    conn_resh = connectivity.reshape([-1,] + [2]*ndims) # Reshaped connectivity array

    patch_bound = {f'patch{i}' : {} for i in range(len(connectivity))}
    # Initialize the conn dictionary 
    for i, iconn in enumerate(conn_resh):
        patch_bound[f'patch{i}'] = {tuple(iconn[0,:]) : 'vmin', tuple(iconn[1,:]) : 'vmax', tuple(iconn[:,0]) : 'umin', tuple(iconn[:,1]) : 'umax'}

    bounds = ("umin", "umax", "vmin", "vmax") 
    pbound = {f'patch{i}' : { key: -1 for key in bounds} for i in range(len(connectivity))}
    for i, iconn in patch_bound.items(): # Loop over patch
        for bound, boundtype in iconn.items(): # Loop over boundaries of the individual patch
            for j, jconn in patch_bound.items(): # Loop over remaining bounds and check if the boundary is inside it
                if i != j: # same patch, should not be compared
                    if bound in jconn:
                        pbound[i][boundtype] = (int(j[5:]), jconn[bound])

    # Calculate information on extraordinary nodes
    _extrnode = {}
    vertex_names = [("umin","vmin"), ("umax","vmin"), ("umin","vmax"), ("umax","vmax")]
    k = 0
    for i in range(np.max(connectivity)+1):
        nodes = np.where(i == connectivity)
        if len(nodes[0]) > 2: # We have an extraordinary node
            patchids, vertexids = nodes
            _extrnode[f'ExNode {k}'] = [(patchid, vertex_names[vertexid]) for patchid, vertexid in zip(patchids, vertexids) ]
            k += 1

    # Bewlo wwe determine the exnodes for a specific patch and how it is connected to other patches
    # self.exnodes = { ("umax","vmax") : ( (1, ("umax","vmin") ), (2, ("umin","vmin") )  ) } 
    # Keys = vertex of the current surface
    # Values = tuple of different patches sharing the same exnode with (patchID, vertex_other) 
    _extrnodeppatch = {f"patch{i}" : {} for i in range(len(connectivity))}
    #for i in range(len(connectivity)):
    for exn, exnode_info in _extrnode.items():
        for i, (k, exnode) in enumerate(exnode_info): #[(0, ("umax","vmax")), (1, (...)), (2, (...)), (3, (...))]
            exnode_info_copy = exnode_info.copy()
            exnode_info_copy.pop(i)
            _extrnodeppatch[f'patch{k}'][exnode] = exnode_info_copy

    return pbound, _extrnode, _extrnodeppatch

def nutils_splipy_boundary_mapp():
    return {"left" : "vmin", "right" : "vmax", "top" : "umax", "bottom" : "umin"} # per definition

# TODO: When only one vertex is attached to a boundary, it is missed (loop over vertices?)
def assign_constraints(patchID, bound_map, constraints, nutils_boundaries, multipatch_conn, extrnode):
    constraints_splipy={}
    for conBound, constraint in constraints.items():
        if conBound != None:
            conBound_split = conBound.split('-') 

        if conBound in nutils_boundaries.keys():
            pbounds = [pbound.split('-') for pbound in nutils_boundaries[conBound].split(',')] # patch boundary [ ['patch0','left'], ['patch1','bottom'], .. ]
            for pbound in pbounds:
                if pbound[0] == f"patch{patchID}":
                    splipy_bound = convert2Splipy_boundname(pbound[1])
                    # nutils_bound = pbound[1]
                    # splipy_bound = bound_map[nutils_bound]
                    constraints_splipy[ splipy_bound ] = constraint
        elif conBound == None:
            constraints_splipy[ None ] = constraints[None]
        elif conBound_split[0][:5] == "patch" and len(conBound_split) == 1: # Ex. ['patch0'] We are constraining an entire patch
            if None in constraints_splipy:
                treelog.info(f"Patch{patchID} has already been completely constraint with 'None', consider using only 1 of the two.")
            if conBound_split[0] == f"patch{patchID}": # make sure we only constrain the specified patch
                constraints_splipy[ None ] = constraint
            # Assign same constrain to boundaries connected to this patch
            else:

                # First loop over possibly shared boundaries or interfaces
                for self_bound, other_info in multipatch_conn.items():
                    if other_info == -1:
                        continue
                    patchID_other, bound_other = other_info
                    if f"patch{patchID_other}" == conBound_split[0]:
                        constraints_splipy[ self_bound ] = constraint

                # next loop over the extraordinary nodes and check if these are shared with that boundary
                for exnode, exnode_info in extrnode.items():
                    exnode_info_dict = {pid: xnode for pid, xnode in exnode_info}
                    if np.array([pid in exnode_info_dict.keys() for pid in [patchID, int(conBound_split[0][5:])]]).all(): 
                       extr_node = exnode_info_dict[patchID]
                       if (extr_node[0] in constraints_splipy.keys()) or (extr_node[1] in constraints_splipy.keys()): # the boundary connected to the extrnode is already included. No need to add this point as well
                           continue   
                       else:
                           constraints_splipy[ exnode_info_dict[patchID] ] = constraint

        elif conBound_split[0][:5] == "patch" and conBound_split[1][:5] != "patch": # Check if we have a following combination ["patch1","left"] 
            if conBound_split[0] == f"patch{patchID}": # only if this is the correct patchID
                splipy_bound = convert2Splipy_boundname(conBound_split[1])
                constraints_splipy[ splipy_bound ] = constraint
                # check if we have a nutils bound
                # if conBound_split[1] in bound_map.keys():
                #     nutils_bound = conBound_split[1]
                #     splipy_bound = bound_map[nutils_bound]
                # else: # if we have splipy name, do not map 
                #     splipy_bound = conBound_split[1]
                # constraints_splipy[ splipy_bound ] = constraint
            else: # Else we have to find the possible boundary or (extr)node connected to it by a different patch
                

                # First check if we have an exnode 
                splipy_bound = convert2Splipy_boundname(conBound_split[1])
                patchID_parent = int(conBound_split[0][5:])
                for exnode, exnode_info in extrnode.items():
                    exnode_info_dict = {pid: xnode for pid, xnode in exnode_info}
                    if np.array([pid in exnode_info_dict.keys() for pid in [patchID_parent, patchID ]]).all(): 
                        if (splipy_bound in exnode_info_dict[patchID_parent]):
                            constraints_splipy[ exnode_info_dict[patchID] ] = constraint


                # Finally check if this boundary is connected to a different patch (not exnode)
                # Furthermore, check if this boundary is an interface, if so assign same constraint to the connected patch
                order = {"umin":0,"vmin":1,"umax":2,"vmax":3} # custom mapp to define order of vertex
                l = lambda x : "u" if x == "v" else "v" # Get exact opposite of what we have 
                bound_conn = ( f"{l(splipy_bound[0])}min", f"{l(splipy_bound[0])}max" )     # tuple of the boundaries that are connected to the 'splipy_bound'
                for self_bound, other_info in multipatch_conn.items():
                    if other_info == -1:
                        continue
                    other_patchID, other_bound = other_info
                    if patchID_parent == other_patchID and other_bound == splipy_bound: # We have an interface
                        constraints_splipy[ self_bound ] = constraint
                    elif patchID_parent == other_patchID and self_bound in bound_conn: # The point is attached to a different patch and is at the boundary of the multipatch domain
                        #self_bound_conn = bound_conn[ other_bound==bound_conn ]
                        key = sorted([self_bound, f"{l(self_bound[0])}{splipy_bound[1:]}"], key=lambda d: order[d]) # "..min", "..max", the 'min' should always be before the 'max'
                        constraints_splipy[tuple(key)] = constraint




        elif conBound_split[0][:5] == "patch" and conBound_split[1][:5] == "patch": # If we have an interface, if we have a following combination ["patch1","patch2"] 
            patchID_other = int(conBound_split[1][5:]) if int(conBound_split[1][5:]) != patchID else int(conBound_split[0][5:])
            patchID_self  = int(conBound_split[1][5:]) if int(conBound_split[1][5:]) != patchID_other else int(conBound_split[0][5:]) # maybe self, could also be viewed from other patch
            
            # First check if the other patchID is connected via its boundary
            for self_bound, other_info in multipatch_conn.items():
                if other_info == -1:
                    continue
                other_patchID, other_bound = other_info
                if other_patchID == patchID_other and patchID_self == patchID:
                    constraints_splipy[ self_bound ] = constraint

            # Second, check if the other patchID is connected via its extraordinary node
            for exnode, exnode_info in extrnode.items():
                exnode_info_dict = {pid: xnode for pid, xnode in exnode_info}
                if np.array([pid in exnode_info_dict.keys() for pid in [patchID_other, patchID_self, patchID]]).all(): 
                   if patchID not in [patchID_other, patchID_self]:
                      #exnode_str = ",".join(exnode_info_dict[patchID])
                      #constraints_splipy[ exnode_str ] = constraint 
                      constraints_splipy[ exnode_info_dict[patchID] ] = constraint



        else:
            raise ValueError(f"Uknown constraint boundary specified {conBound}")
        
    # for bound, value in nutils_boundaries.items():
    #     if bound in constraints.keys(): # the specific boundary has a constraint
    #         constraint = constraints[bound]            
    #         pbounds = [pbound.split('-') for pbound in value.split(',')] # patch boundary [ ['patch0','left'], ['patch1','bottom'], .. ]
    #         for pbound in pbounds:
    #             if pbound[0] == f"patch{patchID}":
    #                 nutils_bound = pbound[1]
    #                 splipy_bound = bound_map[nutils_bound]
    #                 constraints_splipy[ splipy_bound ] = constraint
    # if None in constraints.keys():
    #    constraints_splipy[ None ] = constraints[None]             
    return constraints_splipy 

def convert2Splipy_boundname(boundary_name):
    bound_map = nutils_splipy_boundary_mapp() 
    if boundary_name in bound_map.keys():
        nutils_bound = boundary_name
        return bound_map[nutils_bound]
    else: # if we have splipy name, do not map 
        return boundary_name
    
# This function evaluates the tangents at a boundary and identifies whether the cross-boundary tangent is 
# pointing outward (False, *-1) or inwards (True, *1) of the patch at the specific boundary.
def evaluate_tangents(surfaces):
    
    # Specify the boundary convention of SpliPy
    bound_order = {"umin":0,"umax":1,"vmin":2,"vmax":3}
    indices = {"umin": (np.array([0,0.5]), np.array([0.01, 0.5 ])), # u and v coordinates of a center coordinate at the boundary & 
               "umax": (np.array([1,0.5]), np.array([0.99, 0.5 ])), # a corresponding coordinate slightly shifted inwards
               "vmin": (np.array([0.5,0]), np.array([0.5 , 0.01])),
               "vmax": (np.array([0.5,1]), np.array([0.5 , 0.99]))} 
    
    for surface in surfaces:
        flip_tangent = {}
        for bname, idx in indices.items():
            idx_center, idx_inward = idx # Get center index of that boundary and the slightly shifted inwards coordinate
            direction = 0 if bname[0] == "u" else 1 # u-direction = 0, v-direction = 1
            t_aprox  = surface(idx_inward[0], idx_inward[1]) - surface(idx_center[0], idx_center[1]) # Approximated tangent vector 
            t_aprox /= np.linalg.norm(t_aprox)
            t_eval   = surface.tangent(idx_center[0], idx_center[1], direction=direction)
            if np.sign(np.dot( t_aprox, t_eval )) == -1:
                flip_tangent[bname] = True
            else:
                flip_tangent[bname] = False
        surface.flip_tangent = flip_tangent.copy() # Assigns dict containing: {'umax' : False, 'vmax' : True, ...} 
    return

# This function assumes that the initial surface is sufficiently smooth with no sharp C0-continuous interfaces
## TODO: Set exception for sharp C0 boundaries
def evaluate_normals(surfaces, patch_conn):
    # Take 1 surface as a reference
    surfaces[0].flip_normal = False
    #assert 'patch{refpatchID}' == patch_conn.keys()[0], "The list of SpliPy surfaces is not correctly ordered"

    bound_order = {"umin":0,"umax":1,"vmin":2,"vmax":3}
    indices = {"umin": (np.array([0,0]),np.array([0,1])), # u and v coordinates/vertices of that boundary (u,v)
               "umax": (np.array([1,1]),np.array([0,1])),
               "vmin": (np.array([0,1]),np.array([0,0])),
               "vmax": (np.array([0,1]),np.array([1,1]))} 
    #patch_conn = dict( 'patch1' = dict( "umin"=(0, "vmax"), "umax"=-1, ..))
    for patch_self, value in patch_conn.items():
        patchID_self = int(patch_self[5:])
        
        for bound_self, info_other in value.items():
            if info_other == -1:
                continue

            patchID_other, bound_other = info_other
            if hasattr(surfaces[patchID_self], "flip_normal") and hasattr(surfaces[patchID_other], "flip_normal"):
                break

            u_self, v_self = indices[bound_self]
            u_other, v_other = indices[bound_other]

            normal_self  = surfaces[patchID_self].normal(u_self, v_self, tensor=False)
            normal_other = surfaces[patchID_other].normal(u_other, v_other, tensor=False)

            dotnormal = np.dot(normal_self[0], normal_other[0]) # Can also pick the second normal

            if not hasattr(surfaces[patchID_self], "flip_normal"):
                surfaces[patchID_self].flip_normal = True if dotnormal < 0 else False
            elif not hasattr(surfaces[patchID_other], "flip_normal"):
                surfaces[patchID_other].flip_normal = True if dotnormal < 0 else False
            else:
                raise Exception("This should not have occured")

    # Check if all surfaces have been evaluated
    evaluated = []
    for surface in surfaces:
        evaluated.append(hasattr(surface, "flip_normal"))


    if len(evaluated) != len(surfaces):
        evaluate_normals(surfaces, patch_conn)    
    return


## Following functions are Bounded to the SpliPy Object (i.e. added as an additional function method of the already existing surface Object instance)

def sample(self, evalpts=(100, 100), returnLocal=False) -> np.ndarray:
    '''Sample the surface by specifying the evaluation points in both u- and v-directions.

       Input:___________
       evalpts:     (int, int); Number of evaluation (sample) points in each parametric direction
       returnLocal: Bool      ; Returns the local coordinates as well (next to the physical)

       Returns:___________
       phys_coords:  np.ndarray; Contains the physical coordinates of the evaluated points on the surface
       local_coords: np.ndarray; Contains the local coordinates of the evaluated points on the surface
    '''
    assert len(evalpts)==2, "Provide surface evaluation resolution for both parametric directions (u- and v-directions); evalpts=(u,v)"
    u_end, v_end = self.end()
    u_start, v_start = self.start()

    u_sample = np.linspace(u_start, u_end, evalpts[0])
    v_sample = np.linspace(v_start, v_end, evalpts[1])
    
    if returnLocal:
        u, v = np.meshgrid(u_sample, v_sample) 
        local = np.concatenate([ u[:,:,_], v[:,:,_] ], axis=2).reshape(-1,2)
        return self(u_sample, v_sample).swapaxes(0,1).reshape(-1,3), local
    else:
        return self(u_sample, v_sample).swapaxes(0,1).reshape(-1,3) # Swap axis, because SpliPy + reshape operation..  


def vertex(self, weightedCPS=False, tensor=True, indexOnly=False) -> Tuple[np.ndarray, np.ndarray]: 
    '''Return the vertices index of the ctrlpts array and vertices coordinates of the surface (4 corners).

       Input:___________
       weightedCPS: Bool; Returns the weighted (4D) controlpoint array
       tensor:      Bool; Returns the indices based on the tensor product (grid structure) if equal to True
       indexOnly:   Bool; Returns only the indices and not the cps coordinates

       Returns:___________
       indices: np.ndarray; Contains the indices of the vertices with respect to the total controlpoints array
       cps:     np.ndarray; Contains the coordinates of the vertices (either in 3D or 4D)
    '''
    nrcps_u, nrcps_v = self.shape
    if tensor:
        indices = np.array([[0,         0], 
                            [nrcps_u-1, 0],
                            [0, nrcps_v-1],
                            [nrcps_u-1, nrcps_v-1] ], int)
        ind_u, ind_v = indices.T
        cpsh = self.controlpoints[ind_u,ind_v]
    else:
        indices = np.array([0, nrcps_u-1, nrcps_u*(nrcps_v-1), nrcps_u*nrcps_v-1], int)
        cpsh = np.array(self)[indices]

    if weightedCPS:
        return (indices, cpsh) if not indexOnly else indices

    return (indices, cpsh[:,:-1]/cpsh[:,_,-1]) if not indexOnly else indices
            

def vertex_dict(self, weightedCPS=False, tensor=True, indexOnly=False) -> dict():
    '''Identical to self.vertex(), but returns a dictionary with keys that indicate relative location:
            ("umin","vmin"), ("umax","vmin"), ("umin","vmax"), ("umax","vmax")
    '''
    vertex_names = [("umin","vmin"), ("umax","vmin"), ("umin","vmax"), ("umax","vmax")] 
    if indexOnly:
        indices = self.vertex(weightedCPS=weightedCPS, tensor=tensor, indexOnly=indexOnly)
        return { key: idx for key, idx in zip(vertex_names, indices) }
    else: 
        indices, controlpoints = self.vertex(weightedCPS=weightedCPS, tensor=tensor, indexOnly=indexOnly)
        return { key: ( idx, cps ) for key, idx, cps in zip(vertex_names, indices, controlpoints) }


def nrcps(self,total=False) -> Tuple[int, int]:
    '''Return the number of controlpoints that define the surface.

       Input:___________
       total: Bool; Returns the total number of controlpoints if equal to True, False equals the number of controlpoints in each u- and v-direction

       Returns:___________
       nrcps: int or (int,int); Value of number of controlpoints or tuple containing the number of controlpoints in u- and v-directions
    '''
    nrcps_u, nrcps_v = self.shape
    if total:
       return nrcps_u*nrcps_v
    return nrcps_u, nrcps_v

def grid_indices(self,):
    nrcps_u, nrcps_v = self.nrcps()
    U,V = np.meshgrid( np.linspace(0, nrcps_u-1, nrcps_u, dtype=int), 
                       np.linspace(0, nrcps_v-1, nrcps_v, dtype=int))
    return np.concatenate( [U[:,:,_], V[:,:,_]], axis=2) # Indices in tensor format

def boundary(self, weightedCPS=False, tensor=True):
    '''Return the boundary indices of the ctrlpts array and the corresponding controlpoint coordinates (4 boundaries).

       Input:___________
       weightedCPS: Bool; Returns the weighted (4D) controlpoint array
       tensor:      Bool; Returns the indices based on the tensor product (grid structure) if equal to True

       Returns:___________
       dict( str = ( np.ndarray, np.ndarray ), ... ) ; Dictionary with keys corresponding to the boundary name (bnames="umin","umax","vmin","vmax")
                                                       and values a tuple where (indices, cps coordinates)
    '''
    bnames = "umin","umax","vmin","vmax" # Boundary names
    indices = self.grid_indices()
    nrcps_u, nrcps_v = self.nrcps()
    # U,V = np.meshgrid( np.linspace(0, nrcps_u-1, nrcps_u, dtype=int), 
    #                    np.linspace(0, nrcps_v-1, nrcps_v, dtype=int))
    # indices = np.concatenate( [U[:,:,_], V[:,:,_]], axis=2) # Indices in tensor format
    bindices_tensor = (indices[:,0], indices[:,-1], indices[0,:], indices[-1,:]) # Indices in tensor format for each boundary (bnames)
    bindices_flat_ref = np.linspace(0,nrcps_u*nrcps_v-1,nrcps_v*nrcps_v).reshape(-1,nrcps_v).T # Indices in flattened format

    if weightedCPS:
        edges = [np.array(edge) for edge in self.edges()]
    else:
        edges = [np.array(edge)[:,:-1]/np.array(edge)[:,_,-1] for edge in self.edges()]

    if tensor:
        bindices = bindices_tensor
    else:    
        bindices = []
        for bindt in bindices_tensor:
            x, y = bindt.T
            bindices.append(bindices_flat_ref[x,y])

    return { bname: (indices, cps) for bname, indices, cps in zip(bnames,bindices,edges) } # Ordering in zip() = "umin","umax","vmin","vmax"

def _gss(self, f, a, b, bfunc, tol=1e-9):
    '''Golden-section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]
    bfunc: indicates the index of the function we want to find the max

    '''
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        if f(c)[0][bfunc] > f(d)[0][bfunc]: # to find the maximum
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2

def maximum_basis(self,):
    '''Calculate the parametric coordinate at which each individual basis has its maximum.
    '''

    maxima = [[0],[0]]
    for i, basis in enumerate(self.bases):
        for Bfunc in range(1,basis.num_functions()-1):
            maxima[i] += [self._gss(basis, 0, 1, bfunc=Bfunc)]
        maxima[i] += [1]
    return maxima

def cpslocal(self, tensor=False):
    '''Calculate the control point its local coordinate.
    '''

    # Store the CPSu and CPSv values
    # if hasattr(self, "CPSu") and hasattr(self, "CPSv"):
    #     if len(self.CPSu) != self.ctrlpts().shape[0] or len(self.CPSv) != self.ctrlpts().shape[1]: # There is a mismatch due to a refinement step
    #         self.CPSu, self.CPSv = self.maximum_basis()
    # else:
    #     self.CPSu, self.CPSv = self.maximum_basis()
    self.CPSu = self.bases[0].greville()
    self.CPSv = self.bases[1].greville()

    # Could have also used 'Greville'
    # knotvec_u, knotvec_v = self.knots()
    # CPSu = knotvec_u[:1] + [ i + 0.5*(j-i) for i,j in zip(knotvec_u[:-1],knotvec_u[1:]) ] + knotvec_u[-1:]
    # CPSv = knotvec_v[:1] + [ i + 0.5*(j-i) for i,j in zip(knotvec_v[:-1],knotvec_v[1:]) ] + knotvec_v[-1:]

    if tensor: # Return tensor representation
       u,v = np.meshgrid(self.CPSu, self.CPSv)
       return np.concatenate([u[...,_],v[...,_]], axis=2)
    else:
       return (self.CPSu.copy(), self.CPSv.copy())

def normals(self,) -> np.ndarray:
    '''Calculate the normal vectors at the control point its local coordinate.
    '''
    # knotvec_u, knotvec_v = self.knots()
    # CPSu = knotvec_u[:1] + [ i + 0.5*(j-i) for i,j in zip(knotvec_u[:-1],knotvec_u[1:]) ] + knotvec_u[-1:]
    # CPSv = knotvec_v[:1] + [ i + 0.5*(j-i) for i,j in zip(knotvec_v[:-1],knotvec_v[1:]) ] + knotvec_v[-1:]

    CPSu, CPSv = self.cpslocal()

    if self.flip_normal:
        return -self.normal(CPSu,CPSv)
    else:
        return  self.normal(CPSu,CPSv)

def tangents(self, direction=None) -> np.ndarray:
    '''Calculate the tangent vectors at the control point its local coordinate.
    '''
    # knotvec_u, knotvec_v = self.knots()
    # CPSu = knotvec_u[:1] + [ i + 0.5*(j-i) for i,j in zip(knotvec_u[:-1],knotvec_u[1:]) ] + knotvec_u[-1:]
    # CPSv = knotvec_v[:1] + [ i + 0.5*(j-i) for i,j in zip(knotvec_v[:-1],knotvec_v[1:]) ] + knotvec_v[-1:]

    CPSu, CPSv = self.cpslocal()

    return self.tangent(CPSu,CPSv, direction=direction)

def ctrlpts(self, weighted=False):
    if weighted:
        return self.controlpoints
    else:
        weight = self.controlpoints[...,-1]
        cps = self.controlpoints[...,:-1]/weight[...,_]
        return cps
    
def weights(self,):
    return self.controlpoints[...,-1]   


# def multipatchmapper(self, update=False):
#     '''Return the extraordinary node information, which contains the patchIDs connected to it and the corresponding
#         indices of the controlpoints of that specific patchID.

#        Input:___________
#        weightedCPS: Bool; Returns the weighted (4D) controlpoint array
#        tensor:      Bool; Returns the indices based on the tensor product (grid structure) if equal to True

#        Returns:___________
#        dict( str = ( np.ndarray, np.ndarray ), ... ) ; Dictionary with keys corresponding to the boundary name (bnames="umin","umax","vmin","vmax")
#                                                        and values a tuple where (indices, cps coordinates)
#     '''
#     if not update and hasattr(self,'multipatch_mapping'):
#         return self.multipatch_mapping
#     else: # else update or construct/assign value to multipatch_mapping
#         ## TODO Add additional statement for vertices!
#         self.multipatch_mapping = {}
#         # First specify the extraordinary node indices:
#         for exnode_self, exnodes_other_info in self.exnodes.items(): # { ("umax","vmax") : ( (1, ("umax","vmin") ), (2, ("umin","vmin") )  ) } 
#             for exnode_other_info in exnodes_other_info:
#                 patchID_other, exnode_other = exnode_other_info
#                 self.multipatch_mapping[patchID_other] = (self.  ) # Indices of: (self, other)


#         for bound_self, info_other in self.multipatchconn.items(): # dict( "umin"=(0, "vmax"), "umax"=-1, ..)
#             for exnode, mpatch_info in self.multipatchexnodes.items():
#                 for patch_info in mpatch_info:
#                     if self.patchID == patch_info:
#                         1
#             if info_other == -1:
#                 self.multipatch_mapping[patchID_other] = -1
#             patchID_other, bound_other = info_other
#             indices_other, cps_other = self.boundary()[bound_other] # dict( 'umax' = ( indices, cps ), ... )
#             indices_self, cps_self = self.boundary()[bound_self]

#             self.multipatch_mapping[patchID_other] = (indices_self, indices_other)
#     return self.multipatch_mapping

def extraordinary_node(self,):
    '''Return the extraordinary node information, which contains the patchIDs connected to it and the corresponding
        indices of the controlpoints of that specific patchID.

       Input:___________
       weightedCPS: Bool; Returns the weighted (4D) controlpoint array
       tensor:      Bool; Returns the indices based on the tensor product (grid structure) if equal to True

       Returns:___________
       dict( str = ( np.ndarray, np.ndarray ), ... ) ; Dictionary with keys corresponding to the boundary name (bnames="umin","umax","vmin","vmax")
                                                       and values a tuple where (indices, cps coordinates)
    '''
    #self.extrnode

    return
















# def create_surf_spline(surface, delta):
#     '''Create a NURBS-Python spline surface from the input geometry'''
#     surf = BSpline.Surface()

#     # degrees both directions
#     surf.degree_u = 2
#     surf.degree_v = 2

#     # set control points
#     surf.ctrlpts_size_u = 3
#     surf.ctrlpts_size_v = 3

#     ctrlpts2d = surface.cps().reshape(surf.ctrlpts_size_v, surf.ctrlpts_size_u, 3).tolist() 


#     # ctrlpts2d = []
#     # ctrlpts = np.array(surface.cps()) 
#     # # rearrange the control points in the surface to create an updated surface
#     # for c_v in range(surf.ctrlpts_size_v):
#     #     ctrlptsx = []
#     #     for c_u in range(surf.ctrlpts_size_u):
#     #         index = c_v*surf.ctrlpts_size_v + c_u
#     #         ctrlpt = ctrlpts[index]
#     #         for i in range(len(ctrlpt)):
#     #             if -1e-10 < ctrlpt[i] < 1e-10:
#     #                 ctrlpt[i] = 0
#     #         ctrlptsx.append(list(ctrlpt))
#     #     ctrlpts2d.append(ctrlptsx)
#     surf.ctrlpts2d = ctrlpts2d
    
#     # set knot vectors
#     surf.knotvector_u = surface.knotvec()[0]
#     surf.knotvector_v = surface.knotvec()[1]

#     # Set evaluation delta for refinement
#     surf.delta = delta
#     # Evaluate surface points
#     surf.evaluate()
#     surf.evalptsnp = np.array(surf.evalpts)
#     return surf



# def create_surf_nurbs(surface, delta):
#     '''Create a NURBS-Python NURBS surface from the input geometry'''
#     surf = NURBS.Surface()

#     # degrees both directions
#     surf.degree_u = 2
#     surf.degree_v = 2

#     # set control points
#     surf.ctrlpts_size_u = 3
#     surf.ctrlpts_size_v = 3

#     # ctrlptsH = np.concatenate([surface.cps(), surface.weights()[:,np.newaxis]], axis=1) # Homogenous coordinates of the cps 4D
#     # ctrlptsH = ctrlptsH.reshape(surf.ctrlpts_size_v, surf.ctrlpts_size_u, 4).tolist()     
#     # surf.ctrlpts2d = ctrlptsH

#     surf.ctrlpts = surface.cps().tolist()
#     surf.weights = surface.weights().tolist()
#     # ctrlpts2d = []
#     # ctrlpts = np.array(surface.cps()) 
#     # # rearrange the control points in the surface to create an updated surface
#     # for c_v in range(surf.ctrlpts_size_v):
#     #     ctrlptsx = []
#     #     for c_u in range(surf.ctrlpts_size_u):
#     #         index = c_v*surf.ctrlpts_size_v + c_u
#     #         ctrlpt = ctrlpts[index]
#     #         for i in range(len(ctrlpt)):
#     #             if -1e-10 < ctrlpt[i] < 1e-10:
#     #                 ctrlpt[i] = 0
#     #         weight = surface.weights()[index]
#     #         ctrlptsx.append([ctrlpt[0]*weight, ctrlpt[1]*weight, ctrlpt[2]*weight, weight])
#     #     ctrlpts2d.append(ctrlptsx)
#     # surf.ctrlpts2d = ctrlpts2d
    
#     # set knot vectors
#     surf.knotvector_u = surface.knotvec()[0]
#     surf.knotvector_v = surface.knotvec()[1]

#     # Set evaluation delta for refinement
#     surf.delta = delta
#     # Evaluate surface points
#     surf.evaluate()
#     surf.evalptsnp = np.array(surf.evalpts)
#     return surf

# def calc_constraint_vec(surface):
#     '''Calculates the constraint vector for each control point (cps) by evaluating the normal vector 
#     at the NURBS-surface closest to the control point'''

#     degree_u, degree_v = surface.degree
#     knotvec_u, knotvec_v = surface.knotvector 
#     # Remove duplicates
#     knotvec_u = knotvec_u[degree_u:-degree_u]
#     knotvec_v = knotvec_v[degree_v:-degree_v]

#     cps_loc_u = knotvec_u[:1] + [ 0.5*(j-i) for i,j in zip(knotvec_u[:-1],knotvec_u[1:]) ] + knotvec_u[-1:]
#     cps_loc_v = knotvec_v[:1] + [ 0.5*(j-i) for i,j in zip(knotvec_v[:-1],knotvec_v[1:]) ] + knotvec_v[-1:]

#     U, V = np.meshgrid(cps_loc_u, cps_loc_v)
#     local_coord = np.c_[V.ravel(), U.ravel()].tolist() # List of local coordinates that correspond to the cps (switch U and V)
#                                                         # We purposely switch the u- and v-coordinates because geomdl does something strange as well when compared to the cps ordering


#     N = operations.normal(surface, local_coord)

#     normal_vec = np.empty((len(local_coord),3),float)
#     xloc_vec = np.empty((len(local_coord),3),float)
#     for i, val in enumerate(N):
#         xloc_vec[i], normal_vec[i] = val

#     # Reshape the array into correct format for EBDM, subject to change!
#     normal_vec_resh =normal_vec.reshape(surface.ctrlpts_size_v,surface.ctrlpts_size_u,3)
    
#     return [ [i for i in normU] for normU in normal_vec_resh] 