#!/usr/bin/env python
import numpy as np
import matplotlib
import skfmm

class Gridded():
    """
    Class for handling interpolation workflow
    """
    def __init__(self, extent, cellsize, boundary, holes=None):
        """
        Initialize a raster object
        extent is [xmin, xmax, ymin, ymax]
        """
        self.extent = [float(x) for x in extent]
        self.cellsize = float(cellsize)
        self.boundary = boundary
        self.holes = holes
        self._build_raster()
    
    def _build_raster(self):
        """
        Helper function creates a target raster taking into account domain topology
        """
        # Get some dimensions and make x,y grid
        nx = int(np.ceil((self.extent[1]-self.extent[0])/self.cellsize)+1)
        xvect = np.linspace(self.extent[0], self.extent[0]+self.cellsize*(nx-1), nx)
        ny = int(np.ceil((self.extent[3]-self.extent[2])/self.cellsize)+1)
        yvect = np.linspace(self.extent[2], self.extent[2]+self.cellsize*(ny-1), ny)
        gridX, gridY = np.meshgrid(xvect, yvect)
        gridXY_array = np.array([np.concatenate(gridX),
                                 np.concatenate(gridY)]).transpose()
        
        # Filter out points outside of domain boundary
        path = matplotlib.path.Path(self.boundary)
        self.raster = path.contains_points(gridXY_array).astype(int)
        
        # Filter out points inside any interal holes
        if getattr(self,'holes',None) is not None:
            for hole in self.holes:
                path = matplotlib.path.Path(hole)
                self.raster[path.contains_points(gridXY_array)] = 0
        
        # Reshape
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
