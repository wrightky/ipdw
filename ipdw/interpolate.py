#!/usr/bin/env python
import numpy as np
import matplotlib
import skfmm

class Gridded():
    """
    Class for handling interpolation workflow
    """
    def __init__(self, cellsize, boundary, holes=None):
        """
        Instantiate a gridded object. This will read the target cellsize,
        domain boundary, and any internal holes in the domain, and create
        a target raster used as a basemap for later interpolation. The
        basemap is stored in the Gridded.raster attribute.
        
        **Inputs**
            cellsize (int or float) : Cellsize of target raster. Smaller
                values will result in a higher-resolution grid and require
                more computational time
            boundary (list or array) : Coordinates of the domain boundary
                in the same length-scale units as "cellsize". Coordinates
                should be specified as [[x1,y1],[x2,y2],...] pairs in either
                a list or array. Array dimensions should be (N,2).
            holes (list of lists or arrays) : Coordinates of internal "holes"
                in the domain, specified as a list of holes. Each hole should
                be specified in the same format as the "boundary" input (i.e.
                either a list or array of x,y coordinates).
        **Outputs**
            After initialization, the Gridded.raster attribute will contain a
            binary basemap for later interpolation, with 1's everywhere inside
            the domain, and 0's elsewhere.
        """
        # Store inputs as attributes for accessibility
        self.cellsize = float(cellsize)
        if type(boundary) == list:
            boundary = np.array(boundary) # Convert to array if necessary
        self.boundary = boundary
        # For convenience, store extent as [xmin, xmax, ymin, ymax]
        self.extent = [min(boundary[:,0]), max(boundary[:,0]),
                       min(boundary[:,1]), max(boundary[:,1])]
        if holes is not None:
            # Convert holes to arrays if necessary
            if type(holes[0]) == list:
                holes = [np.array(hole) for hole in holes]
        self.holes = holes
        # Call _build_raster function to build basemap:
        self._build_raster()
    
    def _build_raster(self):
        """
        Helper function creates a target raster taking into account domain
        topology. Function is called during Gridded object initialization.
        
        In order to build the target raster, this function relies heavily on
        matplotlib.path.Path() to make use of the domain boundary and hole
        information. The speed of this function is therefore primarily limited
        by the computational time needed to build each Path object and query
        each raster point.
        
        **Inputs**
            None, but relies on attributes supplied during instantiation.
        **Outputs**
            After running, the Gridded.raster attribute will contain a binary
            basemap for later interpolation, with 1's everywhere inside the
            domain, and 0's elsewhere.
        """
        # Construct the target x,y grid using the domain extent and cellsize
        nx = int(np.ceil((self.extent[1]-self.extent[0])/self.cellsize)+1)
        xvect = np.linspace(self.extent[0],
                            self.extent[0]+self.cellsize*(nx-1), nx)
        ny = int(np.ceil((self.extent[3]-self.extent[2])/self.cellsize)+1)
        yvect = np.linspace(self.extent[2],
                            self.extent[2]+self.cellsize*(ny-1), ny)
        gridX, gridY = np.meshgrid(xvect, yvect)
        gridXY_array = np.array([np.concatenate(gridX),
                                 np.concatenate(gridY)]).transpose()
        # gridXY_array contains all raster cell coordinates
        
        # Filter out points outside of domain boundary
        path = matplotlib.path.Path(self.boundary) # Build path
        self.raster = path.contains_points(gridXY_array).astype(int)
        
        # Filter out points inside any interal holes
        if self.holes is not None:
            for hole in self.holes:
                path = matplotlib.path.Path(hole)
                self.raster[path.contains_points(gridXY_array)] = 0
        
        # Reshape to actual raster dimensions
        self.raster.shape = (len(yvect), len(xvect))
        self.raster = np.flipud(self.raster)
        return
    
    def interpolate(self, input_locations, input_values, n_nearest=3, offsets=None, buffer=1):
        """
        Method performs interpolation from discete data points onto target raster
        """
        self.n_nearest = n_nearest
        
        if offsets is None:
            offsets = np.zeros_like(input_values)
        
        self.dist_from_each = np.ones((self.raster.shape[0], self.raster.shape[1], len(input_values)))*np.nan
        for n in range(len(input_values)):
            x = input_locations[n,0]
            y = input_locations[n,1]
            
            iy = int(self.raster.shape[0] - round((y - self.extent[2])/self.cellsize))
            ix = int(round((x - self.extent[0])/self.cellsize))
            
            phi = -1*self.raster
            mask = self.raster==0
            phi = np.ma.MaskedArray(phi, mask)
            if buffer > 0:
                phi[iy-buffer:iy+buffer,ix-buffer:ix+buffer] = 1
            else:
                phi[iy,ix] = 1
            try:
                dist = np.abs(skfmm.distance(phi)*self.cellsize) + offsets[n]
            except ValueError:
                raise ValueError("One or more locations are not within the enclosed boundary. Check locations or try increasing the buffer size")
            dist[mask] = np.nan
            self.dist_from_each[:,:,n] = dist

        self.arg_n_min = np.argsort(dist_from_each)[:,:,:n_nearest]

        numerator = np.zeros_like(self.raster, dtype=float)
        denominator = np.zeros_like(self.raster, dtype=float)
        for i in range(self.n_nearest):
            vi = np.array([input_values[n] for n in self.arg_n_min[:,:,i]])
            v1.shape = self.raster.shape
            di = np.take_along_axis(self.dist_from_each, self.arg_n_min, axis=-1)[:,:,i]
            
            numerator += vi/di
            denominator += 1/di

        self.output = numerator/denominator
        self.output[self.raster==0] = np.nan
        return self.output

    def reinterpolate(self, input_values):
        """
        If self.interpolate has already been run, this function allows you to
        interpolate again from the same locations using new values. Reduces the
        computational cost compared to rebuilding the interpolation again from scratch.
        """
        numerator = np.zeros_like(self.raster, dtype=float)
        denominator = np.zeros_like(self.raster, dtype=float)
        for i in range(self.n_nearest):
            vi = np.array([input_values[n] for n in self.arg_n_min[:,:,i]])
            v1.shape = self.raster.shape
            di = np.take_along_axis(self.dist_from_each, self.arg_n_min, axis=-1)[:,:,i]
            
            numerator += vi/di
            denominator += 1/di

        self.output = numerator/denominator
        self.output[self.raster==0] = np.nan
        return self.output
