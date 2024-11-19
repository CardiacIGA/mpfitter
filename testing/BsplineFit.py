import ebdm.EBDM_2D as fit
from geomdl import BSpline
import numpy as np
_ = np.newaxis
data_points = np.array([ [-1,0,0], [0,-1,0], [1,0,0], [0,1,0] ]) # Square
phi = np.linspace(0,2*np.pi,50)
r   = np.cos(phi) + 1
data_points = np.array([ r*np.sin(phi),r*np.cos(phi), np.zeros(len(phi)) ]).T

# Initialize the spline
ref_curve = BSpline.Curve()
ref_curve.degree        = 2
ref_curve.ctrlpts       = [[-1,0,0], [0,-1,0], [1,0,0], [0,1,0], [-1,0,0]]
ref_curve.knotvector    = [0,0,0,1,2,3,3,3]
ref_curve.sample_size   = 150
ref_curve.Nonuniform    = True
#ref_curve.evalptsCustom = fit.sample_spline(ref_curve) # Perform non-uniform sampling
#ref_curve.evalptsCustom = ref_curve.evalpts # Perform uniform sampling
ref_curve_list      = list(ref_curve)

# Only sample unfiromly between knots! Not global uniform
# def sample_spline(curve, level):
#     ref   = level + 1
#     uKnot = np.array(curve.knotvector[curve.degree:-curve.degree])
#     x     = np.array([i for i in range(len(uKnot))])
#     xRef  = np.array([i/ref for i in range((len(uKnot)-1)*ref+1)])
#     uRef  = np.interp(xRef,x,uKnot)
#     return uRef.tolist()

# Run the EBDM for multiple splines
max_iter            = 20        # amount of iterations
min_error           = 0.01     # error criterion
max_ctrlpt_addition = [6]      # state how many control points may be added per spline
connection          = [[0,0]]  # state which curves are connected ([first, last])  
rel                 = 0.5      # relaxation factor to determine (0 = no continuity, 1 = full continuity)   

avgerrors_init, sample_datasets, curves_list, errors_list, continuities = fit.EBDM(ref_curve_list, 
                                                                                  data_points, max_iter, 
                                                                                  min_error, max_ctrlpt_addition,
                                                                                  connection, rel)


    
""" Plot the desired graphs """
blue    = '#0077BB'
cyan    = '#33BBEE'
teal    = '#009988'
orange  = '#EE7733'
red     = '#CC3311'
magenta = '#EE3377'
grey    = '#BBBBBB'
colors  = [blue, red, teal, orange, cyan, magenta, grey]

# Show the final curves
fit.plot_splines(curves_list[-7:], data_points)
# fit.plot_splines(curves_list[-1], data_points)
# fit.plot_spline(ref_curve,grey,'',show=True)
fit.plot_splines(curves_list[:5], data_points)