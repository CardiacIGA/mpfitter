# This script calculates the area of the Voronoi segments bounded by a square box
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import splipy as sp
from  nutils import mesh, export, function
_=np.newaxis

def intersec_lines(line1, line2): # input line = ( np.array([]), np.array([]) ), first index is the physica lpoint, second the vector travelling from that point
    point1, vec1 = line1
    point2, vec2 = line2
    if np.isclose( abs(np.dot(vec1/np.linalg.norm(vec1),vec2/np.linalg.norm(vec2))), 1):# check if vectors are parallel
        return np.array([1e3,1e3])
    else:
        M = np.concatenate([vec1[:,_], -vec2[:,_]], axis=1)
        f = (point2 - point1)[:,_]
        sol = np.linalg.solve(M,f)
        return sol[0]*vec1 + point1

def area_polygons(points, connectivity):
    area_pr = np.zeros(len(connectivity))
    for i, conn in enumerate(connectivity):
        area_pr[i] = area_polygon(points[conn])
    return area_pr

def area_polygon(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0 for ((x0, y0), (x1, y1)) in segments(p)))

def segments(p):
    return zip(p, np.concatenate([p[1:] , p[0][_]],axis=0))

def mask_bbox(x, bbox=(1,1), tol=1e-14): # x should be (n,2) shape
    return (-tol <= x[:,0]) & (x[:,0] <= bbox[0]+tol) & (-tol <= x[:,1]) & (x[:,1] <= bbox[1]+tol)

def voronoi_finite_bbox(vor, bbox=(1,1), comp_points=8):
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
    #bbpoint_index = {np.argmin(np.sum((vor.points - bbvert)**2, axis=1)) : i for i, bbvert in enumerate(bbverts[:-1])}  

    # Check vertices if they are within the bounding box, else remove them and keep track of the removed index
    new_regions = []
    
    # Remove vertices oustide bbox:
    del_indices  = ~mask_bbox(vor.vertices, bbox=bbox) #np.argwhere( (vor.vertices > 1).any(1) ).ravel()
    del_indices_int = np.argwhere(del_indices==True).ravel()
    old_vertices = vor.vertices
    new_vertices = np.delete(vor.vertices, del_indices,axis=0)# (vor.vertices[removed_indices]).tolist()

    # Construct a map for deleted vertices, the idea: "new_index = new_indices_map[old_index]"
    old_indices = np.linspace(0,len(vor.vertices)-1,len(vor.vertices)).astype(int)
    new_indices = np.linspace(0,len(new_vertices)-1,len(new_vertices)).astype(int)
    #new_indices_map = {old_i : new_indices[i] for i, old_i in enumerate(old_indices) if old_i not in del_indices}
    i = 0
    new_indices_map = {}
    for old_i in old_indices:
        if old_i not in del_indices_int:
            new_indices_map[old_i] = new_indices[i]
            i += 1

    new_indices_map.update({k : -1 for k in del_indices_int})
    new_indices_map[-1] = -1 # default 

    # Construct map to obtain point index from voronoi region index, the idea: "point_index = region_indices_map[region_index]"
    point_2_region_indices_map = {point_i : region_i for point_i, region_i in enumerate(vor.point_region)}    
    region_2_point_indices_map = {region_i : point_i for point_i, region_i in enumerate(vor.point_region)}

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, new_indices_map[v1], new_indices_map[v2]))
        all_ridges.setdefault(p2, []).append((p1, new_indices_map[v1], new_indices_map[v2]))
    
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertex_indices = np.array([new_indices_map[vert] for vert in vor.regions[region]])
        if p1 == 98:
            print(1)
        # We encountered a finite region
        if all(v >= 0 for v in vertex_indices):
            # finite region
            new_regions.append(vertex_indices.astype(int).tolist())
            continue


        # We encountered a nonfinite region
        ridges = all_ridges[p1]
        #new_region = [v for v in vertex_indices if v >= 0]

        new_vert = np.ndarray((0,2),float)
        for p2, v1, v2 in ridges:
            # if v1 and v2 are both -1, we need to find 2 intersecting points with the bbox


            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint  = vor.points[[p1, p2]].mean(axis=0)
            line      = (midpoint, n) # line of interest
            xpoints   = np.array([intersec_lines(line, bline) for bname, bline in bbox_lines.items()]) # array of intersection points, we need to find the correct ones
            xpoints_mask = mask_bbox(xpoints, bbox=bbox) #(-1e-9 <= xpoints[:,0]) & (xpoints[:,0] <= 1+1e-14) & (-1e-9 <= xpoints[:,1]) & (xpoints[:,1] <= 1+1e-14) # filter out points that are outside the bbox
            xpoints   = xpoints[xpoints_mask]

            dreg={}
            for d, x in enumerate(xpoints):
                dnorm = np.linalg.norm(vor.points-x,axis=1)
                point_indices_closest = np.argsort(dnorm)
                # construct list of point indices that share the same distance to the xpoint    
                point_indices = [l for l in point_indices_closest[:comp_points] if np.isclose(dnorm[point_indices_closest[0]],dnorm[l])]
                if p1 in point_indices: # ridge intersection point with bbox
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


        vertex_indices  = np.concatenate([vertex_indices,len(new_vertices)+ np.array(range(0,len(new_vert)))])
        vertex_ind_keep = np.where(vertex_indices >= 0)
        vertex_indices  = vertex_indices[vertex_ind_keep].astype(int).tolist()
        
        new_vertices    = np.concatenate([new_vertices, new_vert], axis=0)
         

        # Append everything
        #new_regions.append() 

        # sort region counterclockwise and make sure we only do this for regions that exist
        if len(new_vert) != 0:
            vs = np.asarray([new_vertices[v] for v in vertex_indices])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            vertex_indices = np.array(vertex_indices)[np.argsort(angles)]

            new_regions.append(vertex_indices)

    return new_regions, new_vertices

from scipy.spatial import Voronoi

# get random points
# npoints = 10
# np.random.seed(1234)
# points = np.random.rand(npoints, 2)
points = np.array([ [0.3030303 , 0.        ],
                    [0.3030303 , 0.1010101 ],
                    [0.3030303 , 0.2020202 ],
                    [0.3030303 , 0.3030303 ],
                    [0.3030303 , 0.4040404 ],
                    [0.3030303 , 0.49494949],
                    [0.15151515, 0.5959596 ],
                    [0.3030303 , 0.5959596 ],
                    #[0.31313131, 0.5959596 ],
                    [0.45454545, 0.5959596 ],
                    [0.58585859, 0.5959596 ],
                    [0.70707071, 0.5959596 ],
                    [0.80808081, 0.5959596 ],
                    [0.88888889, 0.5959596 ],
                    [0.94949495, 0.5959596 ],
                    [0.98989899, 0.5959596 ],
                    [1.        , 0.5959596 ],
                    [0.3030303 , 0.6969697 ],
                    [0.3030303 , 0.7979798 ],
                    [0.3030303 , 0.8989899 ],
                    [0.3030303 , 1.        ],
                    [0.        , 0.5959596 ]])



# points = np.array([[0.07070707, 0.1010101 ],
#        [0.09090909, 0.07070707],
#        [0.1010101 , 0.04040404],
#        [0.16161616, 0.01010101],
#        [0.15151515, 0.05050505],
#        [0.14141414, 0.08080808],
#        [0.11111111, 0.12121212],
#        [0.08080808, 0.15151515],
#        [0.01010101, 0.19191919],
#        [0.02020202, 0.91919192],
#        [0.04040404, 0.94949495],
#        [0.06060606, 0.97979798],
#        [0.11111111, 0.97979798],
#        [0.1010101 , 0.94949495],
#        [0.07070707, 0.90909091],
#        [0.05050505, 0.87878788],
#        [0.01010101, 0.84848485],
#        [0.        , 0.27272727],
#        [0.        , 0.24242424],
#        [0.06060606, 0.21212121],
#        [0.12121212, 0.17171717],
#        [0.16161616, 0.13131313],
#        [0.19191919, 0.09090909],
#        [0.2020202 , 0.06060606],
#        [0.2020202 , 0.02020202],
#        [0.16161616, 0.97979798],
#        [0.15151515, 0.93939394],
#        [0.13131313, 0.8989899 ],
#        [0.12121212, 0.86868687],
#        [0.08080808, 0.82828283],
#        [0.02020202, 0.78787879],
#        [0.        , 0.75757576],
#        [0.        , 0.41414141],
#        [0.        , 0.35353535],
#        [0.        , 0.31313131],
#        [0.        , 0.27272727],
#        [0.16161616, 0.23232323],
#        [0.21212121, 0.19191919],
#        [0.23232323, 0.15151515],
#        [0.24242424, 0.11111111],
#        [0.25252525, 0.07070707],
#        [0.26262626, 0.03030303],
#        [0.31313131, 0.04040404],
#        [0.31313131, 0.08080808],
#        [0.31313131, 0.12121212],
#        [0.31313131, 0.16161616],
#        [0.31313131, 0.21212121],
#        [0.3030303 , 0.26262626],
#        [0.29292929, 0.31313131],
#        [0.        , 0.35353535],
#        [0.        , 0.46464646],
#        [0.        , 0.55555556],
#        [0.14141414, 0.75757576],
#        [0.17171717, 0.80808081],
#        [0.19191919, 0.84848485],
#        [0.2020202 , 0.88888889],
#        [0.21212121, 0.92929293],
#        [0.21212121, 0.96969697],
#        [0.85858586, 0.94949495],
#        [0.80808081, 0.97979798],
#        [0.36363636, 0.        ],
#        [0.37373737, 0.05050505],
#        [0.38383838, 0.09090909],
#        [0.39393939, 0.14141414],
#        [0.41414141, 0.18181818],
#        [0.44444444, 0.24242424],
#        [0.46464646, 0.29292929],
#        [0.51515152, 0.35353535],
#        [0.56565657, 0.4040404 ],
#        [0.58585859, 0.46464646],
#        [0.57575758, 0.51515152],
#        [0.        , 0.57575758],
#        [0.        , 0.63636364],
#        [0.35353535, 0.68686869],
#        [0.31313131, 0.73737374],
#        [0.3030303 , 0.78787879],
#        [0.3030303 , 0.83838384],
#        [0.29292929, 0.87878788],
#        [0.28282828, 0.92929293],
#        [0.27272727, 0.96969697],
#        [1.        , 0.01010101],
#        [1.        , 0.03030303],
#        [1.        , 0.06060606],
#        [1.        , 0.08080808],
#        [1.        , 0.1010101 ],
#        [1.        , 0.13131313],
#        [1.        , 0.15151515],
#        [1.        , 0.17171717],
#        [1.        , 0.19191919],
#        [1.        , 0.21212121],
#        [1.        , 0.23232323],
#        [1.        , 0.25252525],
#        [1.        , 0.27272727],
#        [1.        , 0.3030303 ],
#        [1.        , 0.31313131],
#        [1.        , 0.34343434],
#        [1.        , 0.36363636],
#        [1.        , 0.38383838],
#        [1.        , 0.4040404 ],
#        [1.        , 0.42424242],
#        [1.        , 0.44444444],
#        [1.        , 0.46464646],
#        [1.        , 0.48484848],
#        [1.        , 0.50505051],
#        [1.        , 0.52525253],
#        [1.        , 0.55555556],
#        [1.        , 0.56565657],
#        [1.        , 0.58585859],
#        [1.        , 0.60606061],
#        [1.        , 0.62626263],
#        [1.        , 0.65656566],
#        [1.        , 0.67676768],
#        [1.        , 0.71717172],
#        [1.        , 0.6969697 ],
#        [1.        , 0.73737374],
#        [1.        , 0.71717172],
#        [1.        , 0.76767677],
#        [0.94949495, 0.75757576],
#        [0.95959596, 0.80808081],
#        [0.88888889, 0.80808081],
#        [0.91919192, 0.84848485],
#        [0.87878788, 0.8989899 ],
#        [0.81818182, 0.91919192],
#        [0.83838384, 0.85858586],
#        [0.75757576, 0.95959596],
#        [0.97979798, 0.03030303],
#        [0.97979798, 0.07070707],
#        [0.98989899, 0.12121212],
#        [0.98989899, 0.16161616],
#        [0.98989899, 0.21212121],
#        [0.98989899, 0.25252525],
#        [1.        , 0.29292929],
#        [1.        , 0.33333333],
#        [1.        , 0.38383838],
#        [1.        , 0.42424242],
#        [1.        , 0.46464646],
#        [1.        , 0.50505051],
#        [1.        , 0.54545455],
#        [1.        , 0.58585859],
#        [1.        , 0.62626263],
#        [1.        , 0.67676768],
#        [0.91919192, 0.03030303],
#        [0.92929293, 0.08080808],
#        [0.92929293, 0.13131313],
#        [0.92929293, 0.17171717],
#        [0.92929293, 0.21212121],
#        [0.93939394, 0.26262626],
#        [0.93939394, 0.3030303 ],
#        [0.94949495, 0.35353535],
#        [0.94949495, 0.39393939],
#        [0.94949495, 0.44444444],
#        [0.94949495, 0.48484848],
#        [0.94949495, 0.52525253],
#        [0.94949495, 0.56565657],
#        [0.94949495, 0.61616162],
#        [0.94949495, 0.65656566],
#        [0.94949495, 0.70707071],
#        [0.86868687, 0.04040404],
#        [0.86868687, 0.09090909],
#        [0.86868687, 0.13131313],
#        [0.86868687, 0.18181818],
#        [0.86868687, 0.22222222],
#        [0.87878788, 0.27272727],
#        [0.88888889, 0.32323232],
#        [0.88888889, 0.36363636],
#        [0.88888889, 0.41414141],
#        [0.88888889, 0.45454545],
#        [0.87878788, 0.50505051],
#        [0.87878788, 0.54545455],
#        [0.88888889, 0.5959596 ],
#        [0.8989899 , 0.63636364],
#        [0.88888889, 0.6969697 ],
#        [0.88888889, 0.74747475],
#        [0.41414141, 0.01010101],
#        [0.43434343, 0.06060606],
#        [0.46464646, 0.11111111],
#        [0.49494949, 0.16161616],
#        [0.47474747, 0.02020202],
#        [0.50505051, 0.07070707],
#        [0.52525253, 0.21212121],
#        [0.55555556, 0.26262626],
#        [0.5959596 , 0.32323232],
#        [0.54545455, 0.12121212],
#        [0.5959596 , 0.18181818],
#        [0.63636364, 0.23232323],
#        [0.66666667, 0.28282828],
#        [0.64646465, 0.37373737],
#        [0.6969697 , 0.34343434],
#        [0.54545455, 0.03030303],
#        [0.58585859, 0.09090909],
#        [0.63636364, 0.14141414],
#        [0.70707071, 0.2020202 ],
#        [0.72727273, 0.25252525],
#        [0.74747475, 0.3030303 ],
#        [0.70707071, 0.43434343],
#        [0.74747475, 0.39393939],
#        [0.77777778, 0.35353535],
#        [0.70707071, 0.49494949],
#        [0.6969697 , 0.55555556],
#        [0.62626263, 0.60606061],
#        [0.53535354, 0.66666667],
#        [0.46464646, 0.71717172],
#        [0.41414141, 0.77777778],
#        [0.39393939, 0.82828283],
#        [0.38383838, 0.87878788],
#        [0.35353535, 0.91919192],
#        [0.33333333, 0.95959596],
#        [0.75757576, 0.90909091],
#        [0.6969697 , 0.94949495],
#        [0.51515152, 1.        ],
#        [0.45454545, 0.97979798],
#        [0.39393939, 0.96969697],
#        [0.61616162, 0.97979798],
#        [0.56565657, 0.95959596],
#        [0.50505051, 0.94949495],
#        [0.43434343, 0.92929293],
#        [0.62626263, 0.92929293],
#        [0.6969697 , 0.8989899 ],
#        [0.55555556, 0.90909091],
#        [0.48484848, 0.8989899 ],
#        [0.46464646, 0.84848485],
#        [0.49494949, 0.80808081],
#        [0.52525253, 0.75757576],
#        [0.58585859, 0.70707071],
#        [0.54545455, 0.85858586],
#        [0.62626263, 0.87878788],
#        [0.58585859, 0.7979798 ],
#        [0.62626263, 0.74747475],
#        [0.64646465, 0.82828283],
#        [0.6969697 , 0.85858586],
#        [0.67676768, 0.78787879],
#        [0.76767677, 0.85858586],
#        [0.82828283, 0.80808081],
#        [0.74747475, 0.80808081],
#        [0.81818182, 0.76767677],
#        [0.75757576, 0.75757576],
#        [0.70707071, 0.73737374],
#        [0.68686869, 0.6969697 ],
#        [0.66666667, 0.64646465],
#        [0.83838384, 0.71717172],
#        [0.76767677, 0.70707071],
#        [0.83838384, 0.66666667],
#        [0.82828283, 0.61616162],
#        [0.80808081, 0.57575758],
#        [0.7979798 , 0.52525253],
#        [0.7979798 , 0.47474747],
#        [0.80808081, 0.43434343],
#        [0.82828283, 0.38383838],
#        [0.82828283, 0.33333333],
#        [0.80808081, 0.28282828],
#        [0.7979798 , 0.23232323],
#        [0.78787879, 0.19191919],
#        [0.7979798 , 0.14141414],
#        [0.7979798 , 0.09090909],
#        [0.80808081, 0.04040404],
#        [0.75757576, 0.65656566],
#        [0.73737374, 0.60606061],
#        [0.72727273, 0.15151515],
#        [0.73737374, 0.1010101 ],
#        [0.66666667, 0.1010101 ],
#        [0.74747475, 0.05050505],
#        [0.67676768, 0.05050505],
#        [0.61616162, 0.05050505],
#        [1.        , 0.75757576],
#        [1.        , 0.77777778],
#        [1.        , 0.7979798 ],
#        [1.        , 0.80808081],
#        [0.96969697, 0.84848485],
#        [0.93939394, 0.88888889],
#        [0.91919192, 0.91919192],
#        [0.91919192, 0.95959596],
#        [1.        , 0.96969697],
#        [1.        , 0.97979798],
#        [1.        , 0.92929293],
#        [1.        , 0.93939394],
#        [1.        , 0.87878788],
#        [1.        , 0.8989899 ],
#        [1.        , 0.83838384],
#        [1.        , 0.85858586],
#        [1.        , 0.81818182],
#        [1.        , 0.83838384],
#        [0.98989899, 0.87878788],
#        [0.97979798, 0.91919192],
#        [0.96969697, 0.96969697]])

# compute Voronoi tesselation
if points.shape[0] < 4:
    rndm_points = np.array([ [1e2, 1e2], [-1e2, 1e2], [1e2, -1e2] ])
    points = np.concatenate([points, rndm_points])
vor = Voronoi(points, qhull_options="QJ")#, incremental=True) # QJ # furthest_site=True)#

# Calculate the voronoi distribution of polygons
regions, vertices = voronoi_finite_bbox(vor)

# calculate area polygons
area_polygs = area_polygons(vertices, regions)
print(f"Area error: {(1-np.sum(area_polygs))*100}%") # Should be 0!
# area_polygs -= np.min(area_polygs)
area_polygs_map = area_polygs/np.max(area_polygs)
#area_polygs  = abs(area_polygs-1)


## Also compute the weight-matrix M_ij that represents the importance to the data fit
cps = np.array([[0,0],[0.5,0],[1,0],
                [0,0.5],[0.5,0.5],[1,0.5],
                [0,1],[0.5,1],[1,1]])
patchcon = np.array([[0,1,2,3]])
patchverts = np.array([ cps[0], cps[2], cps[-3], cps[-1] ])
w   = np.ones(len(cps))
knotvec = [0,0,0,1,1,1]
order   = 3 # This is fixed

cpsh  = np.c_[cps*w[:,_], w[:,_]]
basis = sp.BSplineBasis(order=order, knots=knotvec)
surface = sp.Surface(basis, basis, cpsh, rational=True)

# Integrate in u-direction and multiply results with dv (spacing in v-direction)
ubasis = surface.bases[0]
vbasis = surface.bases[1]
#N_intbases  = np.zeros((nrsurfpoints, vbasis.num_functions(), ubasis.num_functions()), float) # For the integral of the bases in the Voronoi diagram
N_evalbases = np.zeros((len(points), vbasis.num_functions(), ubasis.num_functions()), float) # For the evaluation of the bases at the datapoint (local)

for i, surfpoint in enumerate(points):
    #N_intbases[i]  = surfpoint_area
    N_evalbases[i] = (ubasis.evaluate(surfpoint[0])*vbasis.evaluate(surfpoint[1]).T)

M_ij = np.transpose( np.dot(N_evalbases.T,area_polygs[:,_]), axes=(1,0,2))

# convert to nutils topo/geom for post processing
topo, lingeom = mesh.multipatch(patches=patchcon, patchverts=patchverts, nelems=1)
bsplines = topo.basis('spline', degree=2, patchcontinuous=False)


weight   = bsplines.dot(w.ravel())
geom     = bsplines.vector(2).dot((cps*w[:,_]).ravel())/weight

topo = topo.refine(1)

ns = function.Namespace()
ns.bspline = topo.basis('lagrange', degree=1)
ns.Mij = 'bspline_n ?lhs_n' 
ns.x = geom
#intg = topo.integral('Mij d:x' @ns, degree=5).eval(lhs=M_ij.ravel())
mapp = [0,1,3,4,2,5,6,7,8]

bezier = topo.sample('bezier', 50)
X, Mij = bezier.eval([geom, ns.Mij], lhs=M_ij.ravel()[mapp])

ctrlnet_out  = np.array([[0., 0.],[1., 0.],[1., 1.],[0., 1.],[0., 0.]])
ctrlnet_vert = np.array([ [0.5, 0], [0.5, 1] ]) 
ctrlnet_hor  = np.array([ [0, 0.5], [1, 0.5] ])
## plot the result
# Latex preamble
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({"font.size": "15",
                      "xtick.labelsize": "12",
                      "ytick.labelsize": "12",
                      "legend.fontsize": "12"})

#%% Plotting
nticks=5
ticks = np.around(np.linspace(area_polygs.min()+0.005, area_polygs.max()-0.005, nticks), decimals=2)
cmap = mpl.colormaps.get_cmap("RdGy_r")#'turbo')
fig, ax = plt.subplots()
ax.set_aspect('equal')
for area, reg in zip(area_polygs_map, regions):
    rgba = cmap(area)
    verts = np.concatenate([ vertices[reg], vertices[reg[0]][np.newaxis] ],axis=0)
    ax.fill(verts[:,0], verts[:,1], color=rgba)
    ax.plot(verts[:,0], verts[:,1], '-', color='black', lw=1.0)
#ax.plot(vertices[:,0], vertices[:,1], '.', lw=1.0)    
ax.plot(points[:,0], points[:,1], '.',color='black', lw=1.0)
ax.set_xlabel(r"$\xi$")
ax.set_ylabel(r"$\eta$")
#plt.axis('off')
dlim = 8e-3
ax.set_ylim([-dlim,1+dlim])
ax.set_xlim([-dlim,1+dlim])
ax.spines[['left','right', 'top','bottom']].set_visible(False)
#ax.tick_params(width=1, length=10, zorder=0)
#ax[1].tick_params(width=1, length=10)
# ax.spines['bottom'].set_bounds((0, 0.1))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(area_polygs), vmax=max(area_polygs)))
cbar = plt.colorbar(sm)
cbar.set_label('$m$', rotation=0, labelpad=15)
cbar.ax.locator_params(nbins=10)

#cbar.set_ticks(ticks)
#cbar.set_ticklabels(ticks)
plt.annotate('$\mathbf{d}^{\mathrm{local},(3)}_k$', [0.32 , 0.88])
#plt.show()
#fig.savefig('testing/output/Voronoi_patch3.png', dpi=600, bbox_inches='tight')


nticks=5
ticks = np.around(np.linspace(M_ij.min()+0.005, M_ij.max()-0.005, nticks), decimals=2)
fig2, ax2 = plt.subplots()
im = ax2.tripcolor(X[:,0], X[:,1], bezier.tri, Mij, shading='gouraud', cmap="RdGy_r")
ax2.autoscale(enable=True, axis='both', tight=True)
ax2.plot(points[:,0], points[:,1], '.',color='black', lw=1.0)
ax2.plot(cps[:,0], cps[:,1], linestyle='None' , marker='s',color='black', lw=1.0)
ax2.plot(ctrlnet_out[:,0], ctrlnet_out[:,1], marker=None,color='black', lw=1.0)
ax2.plot(ctrlnet_vert[:,0], ctrlnet_vert[:,1], marker=None,color='black', lw=1.0)
ax2.plot(ctrlnet_hor[:,0], ctrlnet_hor[:,1], marker=None,color='black', lw=1.0)
cb = fig2.colorbar(im)
cb.set_label('$M_{i,j}$', rotation=0, labelpad=15)
cb.ax.locator_params(nbins=9)

#cb.set_ticks(ticks)
#cb.set_ticklabels(ticks)
ax2.set_ylim([-dlim,1+dlim])
ax2.set_xlim([-dlim,1+dlim])
ax2.spines[['left','right', 'top','bottom']].set_visible(False)
ax2.set_xlabel(r"$\xi$")
ax2.set_ylabel(r"$\eta$")
plt.annotate('$\mathbf{P}^{(3)}_{i,j}$', [0.52, 0.42])
plt.show()
#fig2.savefig('testing/output/Mij_patch3.png', dpi=600, bbox_inches='tight')