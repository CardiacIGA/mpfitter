import matplotlib as mpl
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import splipy.surface as SplipyS
import numpy as np
import os, treelog
from PIL import Image
import imageio
_=np.newaxis

class Export:

    def __iter__():
        return
    
    def VTK():
        return
    
    def STL():
        return
    

#TODO Add control-net save to vtk function

class Plotting:

    def __init__(self, surfaces, datapoints, error_info={}):
        if type(surfaces) == SplipyS.Surface:
            surfaces = [surfaces]
        self.surfaces   = surfaces
        self.datapoints = datapoints
        self.error_info = error_info
        self.colors = self._init_colors()
        self.bounds = self._set_bounds(datapoints)
        return
    
    @staticmethod
    def _eval_surface(surface : SplipyS.Surface, evalpts=(100, 100)): #-> np.ndarray
        u_end, v_end = surface.end()
        u_start, v_start = surface.start()

        u_sample = np.linspace(u_start, u_end, evalpts[0])
        v_sample = np.linspace(v_start, v_end, evalpts[1])
        
        return surface(u_sample, v_sample).swapaxes(0,1).reshape(-1,3)
    
    @staticmethod
    def _set_bounds(x,margin=0.2,aspect_equal=True):
        bounds = {}
        dimstr = "x","y","z"
        for k, dim in enumerate(dimstr):
            maxV = np.max(x[:,k])
            minV = np.min(x[:,k])
            Marg = margin*(maxV-minV)
            bounds[f"{dim}max"] = maxV + Marg
            bounds[f"{dim}min"] = minV - Marg
        
        # set aspect ratio equal
        if aspect_equal:
            ...
        return bounds

    @staticmethod
    def _init_colors():
        # Define colors and the order to use them in (https://personal.sron.nl/~pault/)
        blue    = '#0077BB'
        cyan    = '#33BBEE'
        teal    = '#009988'
        orange  = '#EE7733'
        red     = '#CC3311'
        magenta = '#EE3377'
        grey    = '#BBBBBB'
        black   = '#000000'
        return [blue, red, teal, orange, cyan, magenta, grey, black]
    
    def surface(self, ctrlpts=True, evalpts=(10,10), show_cerror=False, saveFig=False, figName="Surface plot.png"):
        if type(self.surfaces[0]) == list: # If we have a list of several solutions, only show the last one
            surfaces = self.surfaces[-1]
        else:
            surfaces = self.surfaces
        self._surface(surfaces, self.datapoints, ctrlpts=ctrlpts, evalpts=evalpts, show_cerror=show_cerror, saveFig=saveFig, figName=figName)
        return
    
    def surface_video(self, ctrlpts=True, evalpts=(10,10), show_cerror=False, fps=10, gifName="Surface plot.png"):
        # Check if we indeed have a list of lists (several solutions)
        if type(self.surfaces[0]) != list:
            raise ValueError("Cannot create a video, only a single solution/iteration is given in 'surfaces' class input")
        #savefolder = os.path.split(gifName)[0] 
        images = []
        writer = imageio.get_writer(gifName[:-4] + ".gif", mode="i", duration=0.1)
        treelog.info("Creating gif")
        for k, surfaces in enumerate(self.surfaces):
            self.progress_bar(k+1, len(self.surfaces))
            self._surface(surfaces, self.datapoints, ctrlpts=ctrlpts, evalpts=evalpts, show_cerror=show_cerror, saveFig=True, figName=gifName, dpi=250)
            img = Image.open(gifName)
            #images.append(img)
            writer.append_data(np.array(img))
            # if k != len(self.surfaces)-1: # If we are not at the last iteration
        writer.close()
        #TODO add check to verify last 3 terms are .gif
        #imageio.mimsave(gifName[:-4] + ".gif", images, 'GIF', fps=10)
        os.remove(gifName)
        return
    
    def _surface(self, surfaces, datapoints, ctrlpts=True, evalpts=(10,10), show_cerror=False, saveFig=False, figName="Surface plot.png", dpi=600):
        # Plot the final surfaces with data points and their control points
        linewidth = 1.5
        
        fig = plt.figure(figsize=(10, 10))
        #fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim3d(self.bounds["xmin"],self.bounds["xmax"])
        ax.set_ylim3d(self.bounds["ymin"],self.bounds["ymax"])
        ax.set_zlim3d(self.bounds["zmin"],self.bounds["zmax"])
        #ax.set_axis_off()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev= 20 , azim= -20)
        ax.set_facecolor((1.0, 1.0, 1.0))
        ax.set_aspect('equal')
        
        # plot data points
        ax.scatter(datapoints[:,0], datapoints[:,1], datapoints[:,2], color="black", linewidth = 1, zorder = 5)
        
        for s, surface in enumerate(surfaces):
            surface.reparam() # make sure the local coordinates range from [0 - 1]

            # Sample the surface
            surface_evalpts = self._eval_surface( surface, evalpts=evalpts)

            # Specify the vertices and faces (triangulation)
            verts, faces = self._verts_faces_surf(surface_evalpts, evalpts=evalpts)
            tri = mtri.Triangulation(verts[:,0], verts[:,1], triangles = faces)
            ls = LightSource(90,45)
            ax.plot_trisurf(verts[:,0], verts[:,1] , verts[:,2], triangles=tri.triangles, color = self.colors[s], lightsource = ls, alpha = 0.5)
            
            
            # Create the boundary sections of the surface to be indicated
            edges = surface.edges() # return tuple with edges
            for edge in edges:
                edge_evalpts = edge.evaluate( np.linspace(0,1,evalpts[0]) )
                x, y, z = edge_evalpts.T
                ax.plot(x, y, z, color = 'black', lw = 2, zorder = 10)

            if ctrlpts: # Plot the controlnet
                ctrlpts_grid = surface.controlpoints # Extract control point grid
                if surface.rational:
                    ctrlpts_grid = ctrlpts_grid[...,:-1]/ctrlpts_grid[...,-1][...,_]

                # Slice the different coordinates    
                Xgrid = ctrlpts_grid[:,:,0]
                Ygrid = ctrlpts_grid[:,:,1]
                Zgrid = ctrlpts_grid[:,:,2]

                ax.plot_surface( Xgrid, Ygrid, Zgrid, color="none", edgecolor=self.colors[s], linewidth = 1 ) #linewidth = linewidth
                ctrlpts_xyz = ctrlpts_grid.reshape(-1,3)
                ax.scatter(ctrlpts_xyz[:,0], ctrlpts_xyz[:,1], ctrlpts_xyz[:,2], color=self.colors[s])

        if show_cerror:
            if len(self.error_info) == 0:
                raise ValueError("Empty 'error_info' provided, please provide this as an input of the class") 
            Xinter = self.error_info["Spatial interface coords"]
            C = self.error_info["Spatial c-error"]
            cerror = ax.scatter3D(Xinter[:,0], Xinter[:,1], Xinter[:,2], c=C, alpha=1, cmap=mpl.colormaps["coolwarm"])#, marker='*')
            plt.colorbar(cerror, shrink=0.8, label="Continuity error [degr]")  

        if saveFig:
            if figName[-3:] != "png":
                figName += ".png"
            plt.savefig(figName, dpi=dpi) 
            return plt.close()
        else:   
            return plt.show()


    def error_video(self, filename="Error plot.mp4", relative=False, reval_lines=True):
        import matplotlib.animation as animation 

        if len(self.error_info) == 0:
            raise ValueError("Empty 'error_info' provided, please provide this as an input of the class") 
        color_dispE = self.colors[0] 
        color_contE = self.colors[1]
        color_comb  = self.colors[2]
   
        if relative: 
            dE  = self.error_info["Displacement norm"].copy()
        else:
            dE  = self.error_info["Displacement error"].copy()
        dC = self.error_info["Continuity error"]
        dI = self.error_info["Refinement indices"]
        It = range(len(dE))

        # Initialize error plot
        figa, axa1 = plt.subplots(figsize=(8, 8))
        axa2 = axa1.twinx()
        #linev = axa1.axvline([-1], [-1e4], [1e4], linestyle='--', color="dimgrey", label="Uniform refinement")
        if reval_lines:
            self._plot_lines(self.error_info["Re-evaluate indices"], axa1, color="lightgrey", label="Datapoint-surface projection")
        self._plot_lines(self.error_info["Refinement indices"], axa1, color="dimgrey", label="Uniform refinement")
        lineE, = axa1.plot(dE[0], It[0], color=color_dispE, linestyle='-') # Displacement error
        lineC, = axa2.plot(dC[0], It[0], color=color_contE, linestyle='-') # Continuity error
        dotE, = axa1.plot(dE[0], It[0], color=color_dispE, linestyle=None, marker='o')
        dotC, = axa2.plot(dC[0], It[0], color=color_contE, linestyle=None, marker='o')
        axa1.tick_params(axis="y", labelcolor=color_dispE)
        axa2.tick_params(axis="y", labelcolor=color_contE)
        axa2.set_ylabel('Continuity error [degr]', color=color_contE)
        if relative:
            axa1.set_ylabel('Relative displacement error [-]', color=color_dispE)
        else:
            axa1.set_ylabel('Displacement error [m]', color=color_dispE)
        axa1.set_xlabel('Iterations')
        axa1.legend(loc='upper right')
        axa1.set_ylim([0,np.max(dE)*1.1])
        axa2.set_ylim([0,np.max(dC)*1.1])
        axa1.set_xlim([0,len(dE)])
        self.align_yaxis(axa1, 0, axa2, 0)

        def animfunc(frame):
            # if frame in dI:
            #    dArg = np.where(frame >= np.array(dI))[0].ravel().tolist()
            #  #  linev = axa1.axvline(dArg, linestyle='--', color="dimgrey")
            #    linev.set_xdata(dArg)
            #    linev.set_ydata([-1e4], [1e4])
            lineE.set_data(It[:frame+1], dE[:frame+1])
            lineC.set_data(It[:frame+1], dC[:frame+1])
            dotE.set_data(It[frame], dE[frame])
            dotC.set_data(It[frame], dC[frame])
            return lineE, lineC, dotE, dotC#, linev #, lineE, lineC
    
    	# def animfunc(frame):
        #     a = 1
        #     return
            # if frame in dI:
            #     linev.set_xdata(frame)
            # lineE.set_xdata(dE[:frame])
            # lineC.set_xdata(dC[:frame])
            # return linev, lineE, lineC

        anim = animation.FuncAnimation(figa, func=animfunc, frames=len(dE), interval=2, blit=True)
        writervideo = animation.FFMpegWriter(fps=60) 
        anim.save(filename, writer=writervideo)
        return plt.show()
    
    def error(self, relative=False, energy=False):
        if len(self.error_info) == 0:
            raise ValueError("Empty 'error_info' provided, please provide this as an input of the class") 
        color_dispE = self.colors[0] 
        color_contE = self.colors[1]
        color_comb  = self.colors[2]
   
        if relative: 
            disp_error  = self.error_info["Displacement norm"].copy()
        else:
            disp_error  = self.error_info["Displacement error"].copy()
        #alpha = 90 - 0.5*self.error_info["Continuity error"]
        #disp_error /= disp_error0/np.tan( np.radians(alpha) )
        #disp_error  = self.error_info["Displacement error"].copy() 
        #disp_error /= 1 if not relative else disp_error[0]


        figa, axa1 = plt.subplots(figsize=(8, 8))
        axa2 = axa1.twinx()
        self._plot_lines(self.error_info["Re-evaluate indices"], axa1, color="lightgrey", label="Datapoint-surface projection")
        self._plot_lines(self.error_info["Refinement indices"], axa1, color="dimgrey", label="Uniform refinement")
        axa1.plot(disp_error, color=color_dispE, linestyle='-')
        axa2.plot(self.error_info["Continuity error"], color=color_contE, linestyle='-')
        #axa2.plot(0.5*(self.error_info["Continuity error"] + disp_error), color=color_comb, linestyle='-')
        axa1.tick_params(axis="y", labelcolor=color_dispE)
        axa2.tick_params(axis="y", labelcolor=color_contE)
        axa2.set_ylabel('Continuity error [degr]', color=color_contE)
        if relative:
            axa1.set_ylabel('Relative displacement error [-]', color=color_dispE)
        else:
            axa1.set_ylabel('Displacement error [m]', color=color_dispE)
        axa1.set_xlabel('Iterations')
        axa1.legend(loc='upper right')
        ymax = max( np.max(disp_error), np.max(self.error_info["Continuity error"]))
        # axa1.set_ylim([0,ymax])
        # axa2.set_ylim([0,ymax])
        #axa1.grid()
        self.align_yaxis(axa1, 0, axa2, 0)
        #plt.show()

        if energy:
            kin_energy  = self.error_info["Kinetic energy"]
            kin_energy /= 1 if not relative else self.error_info["Displacement error"][0]
            color_kinE = self.colors[2]
            figb, axb1 = plt.subplots(figsize=(8, 8))
            axb2 = axb1.twinx()
            self._plot_lines(self.error_info["Re-evaluate indices"], axb1, color="lightgrey", label="Datapoint-surface projection")
            self._plot_lines(self.error_info["Refinement indices"], axb1, color="dimgrey", label="Uniform refinement")
            axb1.plot(disp_error, color=color_dispE, linestyle='-')
            axb2.plot(kin_energy, color=color_kinE, linestyle='-')
            axb1.tick_params(axis="y", labelcolor=color_dispE)
            axb2.tick_params(axis="y", labelcolor=color_kinE)
            if relative:
                axb1.set_ylabel('Relative displacement error [-]', color=color_dispE)
                axb2.set_ylabel('Relative kinetic energy [-]', color=color_kinE)
            else:
                axb1.set_ylabel('Displacement error [m]', color=color_dispE)
                axb2.set_ylabel('Kinetic energy [J]', color=color_kinE)
            axb1.set_xlabel('Iterations')
            axb1.legend(loc='upper right')
            #axb1.grid()
            #axb2.set_ylim([0,1.1])
            self.align_yaxis(axb1, 0, axb2, 0)

        return plt.show()


    

    '''Visualisation utility functions'''
    @staticmethod
    def _verts_faces_surf(surface_evalpts, evalpts=(10,10)):

        # verts = np.empty((0,3), float)
        # for i in range(len(surf.evalpts)):
        #     vert = np.array(surf.evalpts[i])
        #     verts = np.concatenate([verts, [vert]], axis = 0)    

        faces = np.empty((0,3), int)
        evalpts_u, evalpts_v = evalpts
        for v in range(evalpts_v - 1):
            for u in range(evalpts_u - 1):
                i_lower = v*evalpts_u + u
                i_upper = (v + 1)*evalpts_u + u
                face =  np.array([[i_lower], [i_upper + 1], [i_lower + 1]]).T
                faces = np.concatenate([faces, face], axis = 0)
                face =  np.array([[i_lower],[i_upper], [i_upper + 1]]).T
                faces = np.concatenate([faces, face], axis = 0)
        
        return surface_evalpts, faces
    
    @staticmethod
    def _plot_lines(i,ax,linestyle="--",color="grey",label=None):
        if type(i) == int:
            ax.axvline(i, linestyle=linestyle, color=color, label=label)
        else:
            for k, ii in enumerate(i):
                if k != 0: # prevent that we have duplicate labels
                    label=None
                ax.axvline(ii, linestyle=linestyle, color=color, label=label)
        return
    
    @staticmethod
    def align_yaxis(ax1, v1, ax2, v2):
        """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
        _, y1 = ax1.transData.transform((0, v1))
        _, y2 = ax2.transData.transform((0, v2))
        inv = ax2.transData.inverted()
        _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
        miny, maxy = ax2.get_ylim()
        ax2.set_ylim(miny+dy, maxy+dy)
        return
    
    @staticmethod
    def progress_bar(progress, total):
        percent = 100 * (progress / float(total))
        bar = "█" * int(percent) + "-"*(100 - int(percent))
        print(f"\r|{bar}| {percent:.2f}%", end="\r")
        return