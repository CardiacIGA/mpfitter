import numpy as np
import treelog

# Utility functions for left ventricle fitting

# We set the point cloud as the ground truth and shift/rotate the template accordingly
def get_ellipsoid_features(pointcloud, base_plane=None):
    X, Y, Z = pointcloud.T

    # first get the basal plane settings
    if not base_plane:
        treelog.info("No basal plane provided for the point cloud, estimating based on structured grid and centered around z-axis. \
                     If the point cloud is not structured nor centered around z, please provide the basal plane!")
        
        # convert to φ-coordinates
        # Filter out value sof x and y close to 0 (not of interest)
        y = Y[np.greater(X, 1e-12)]
        x = X[np.greater(X, 1e-12)]
        z = Z[np.greater(X, 1e-12)]

        φ = np.arctan(y/x)
        idx_base = np.ndarray((0), int)
        for φi in np.unique(φ):
            idx = np.where( φi == φ )
            idx_base = np.concatenate([idx_base, idx[np.argmax(z[idx])]])
        x_base = x[idx_base]
        y_base = y[idx_base]
        z_base = z[idx_base]     

    base_plane = 1 # settings of the plane that defines the base
    radii  = 1     # radii in 3 principal directions
    center = 0     # center/origin of the point cloud
    z_base = 1     # Average 
    return base_plane, radii


def position(pointcloud, baseplane, multipatch_surface):
    basenormal_data = baseplane["normal"]
    basepoint_data  = baseplane["point"]
    
    # Rotate template slightly
    #multipatch_surface.rotate(10, axis='z')

    # center the template 
    base_center_templ = multipatch_surface.get_base_center() # Returns base center coordinate
    multipatch_surface.translate(basepoint_data - base_center_templ)

    # stretch the templte vertically
    # apex_point_data   = pointcloud[np.argmin(pointcloud[:,-1])]
    # apex_point_templ  = multipatch_surface.get_apex_point() # Return apex coordinate

    # # get stretch factor in z-direction
    # assert np.sign(apex_point_data[-1]) == np.sign( apex_point_templ[-1] ), "NotImplemented: Mismatching signs, stretching failed"
    # assert not np.isclose(apex_point_templ[-1],0), "Apex is located at z=0, this is problematic!"
    # sfactor = apex_point_data[-1] / apex_point_templ[-1]
    # multipatch_surface.scale(sfactor, direction='z') 
    return multipatch_surface # The rotation matrix and translation vector of the template

def plane_from_points(x1, x2, x3):
    #TODO set assertion that points are not colinear
    #define a plane based on 3 points
    dv1 = x1-x2
    dv2 = x1-x3
    normal = np.cross(dv1,dv2)
    normal /= np.linalg.norm(normal)
    point=x1
    return dict(normal=normal, point=point)