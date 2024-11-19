import numpy as np
import matplotlib.pyplot as plt
_=np.newaxis

def intersec_lines(line1, line2): # input line = ( np.array([]), np.array([]) ), first index is the physica lpoint, second the vector travelling from that point
    point1, vec1 = line1
    point2, vec2 = line2
    M = np.concatenate([vec1[:,_], -vec2[:,_]], axis=1)
    f = (point2 - point1)[:,_]
    sol = np.linalg.solve(M,f)
    return sol[0]*vec1 + point1

# Coordinates of the cps
A = np.array([0, 0])
B = np.array([-0.3, 0.6])
C = np.array([1, 0])
points = np.concatenate([ A,B, C ]).reshape(-1,2)
 
# Weights of importance
#weights = np.array([0.8, 0.5, 0.2])
# weights = np.array([0.11741352, 0.09111662, 0. ])
weights = np.array([0, 0.9, 0. ])
#weights /= np.sum(weights)

# Get normals
def normal(x, y):
    n = np.array([-1, 1])*(y-x)[::-1]
    return n / np.linalg.norm(n)

dline1  = B-A  
dline2  = C-B
normal1 = normal(A,B)
normal2 = normal(B,C)
disp = -(normal1+normal2)/np.linalg.norm(normal1+normal2)

# Determine the points of rotation PoR
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
        #return np.array([ weights[1]/(weights[1] + weights[0]), weights[2]/(weights[1] + weights[2]) ])
#wPoR1 = weights[1]/(weights[1] + weights[0]) if np.isclose(weights[1], weights[0])
# PoR1 = weights[1]/(weights[1] + weights[0])*(B-A) + A
# PoR2 = weights[2]/(weights[1] + weights[2])*(C-B) + B
wPoR = get_wPoR(weights)
PoR1 = wPoR[0]*(B-A) + A
PoR2 = wPoR[1]*(C-B) + B


line_ref = ( PoR1, (PoR1 - PoR2) )
line_A   = ( A,  disp )
line_B   = ( B, -disp )
line_C   = ( C,  disp )



A_new = intersec_lines(line_ref, line_A)
B_new = intersec_lines(line_ref, line_B)
C_new = intersec_lines(line_ref, line_C)
points_new = np.concatenate([A_new, B_new, C_new]).reshape(-1,2)
#Disp = points_new - points


fig = plt.figure(figsize=(10, 10))
normals = np.concatenate([normal1,normal2]).reshape(-1,2)
ax = fig.add_subplot()
ax.plot(points[:,0],points[:,1], color='r', marker='o')
ax.plot(points_new[:,0],points_new[:,1], color='g', marker='o')
ax.plot(PoR1[0],PoR1[1], color='k', marker='x')
ax.plot(PoR2[0],PoR2[1], color='k', marker='x')
#plt.quiver(*points.T, Disp[:,0], Disp[:,1], color=['r','b','g'], scale=21)
plt.quiver(*np.concatenate([B,B]).reshape(-1,2).T, normals[:,0], normals[:,1], color='k')
ax.set_aspect('equal')
plt.show()


