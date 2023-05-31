#!/usr/bin/env python
import numpy as np
import matplotlib
import skfmm

class Gridded():
    """
    Class for handling interpolation workflow
    """
    def __init__(self,
                 cellsize=None,
                 boundary=None,
                 holes=None,
                 raster=None,
                 extent=None
                 ):
        """
        Instantiate a gridded object using domain geometry information.
        Optional input combinations are EITHER (cellsize, boundary, holes)
        OR (raster, extent). ONE of these combinations is required, and
        any inputs in the other combination will be ignored. 
        
        Creates the basemap used for interpolation, which is stored in the
        Gridded.raster attribute.
        
        **Inputs**
            cellsize (float, optional) : Cellsize of target raster. Smaller
                values will result in a higher-resolution grid and require
                more computational time. Must be specified in combination
                with 'boundary'.
            boundary (list or array, optional) : Coordinates of the domain
                boundary in the same length-scale units as "cellsize".
                Coordinates should be specified as [[x1,y1],[x2,y2],...] pairs
                in either a list or array. Array dimensions should be (N,2).
                Must be specified in combination with 'cellsize'.
            holes (list of lists or arrays, optional) : Coordinates of 
                internal "holes" in the domain, specified as a list of holes.
                Each hole should be specified in the same format as the
                'boundary' input (i.e. either a list or array of x,y
                coordinates). Input is optional, but will only be used if
                'cellsize' and 'boundary' are also specified.
            raster (array, optional) : Binary array used as a basemap for
                interpolation, with 1's in cells that should be filled during
                interpolation, and 0's in cells that should be ignored.
                Distances will be computed within continuous clusters of cells
                in which the value =1. Must be specified in combination with
                'extent', and will be ignored if 'cellsize' and 'boundary'
                are both specified.
            extent (list or array, optional) : (x,y) extent of input raster,
                specified in the order [xmin, xmax, ymin, ymax]. Assumed to be
                the footprint of the raster, not the cell centroid coordinates
                of the boundary cells. If using rasterio, can be accessed
                through the 'bounds' attribute of rasterio.io.DatasetReader.
                Must be specified in combination with 'raster', and will be
                ignored if 'cellsize' and 'boundary' are both specified.
        **Outputs**
            After initialization, the Gridded.raster attribute will contain a
            binary basemap for later interpolation, with 1's in cells to be
            filled by interpolation, and 0's elsewhere. Also saves the
            attributes .cellsize, .extent, .bbox (similar to extent, but
            contains boundary cell centroid coordinates).
        """
        if cellsize is not None and boundary is not None:
            # Call _build_raster function to build basemap:
            self._build_raster(cellsize, boundary, holes)
        elif raster is not None and extent is not None:
            # Call _from_raster to compute model geometry:
            self._from_raster(raster, extent)
        else:
            raise ValueError("Incorrect inputs specified.\nRequires combination "+\
                "of either cellsize AND boundary or raster AND extent")
        
    
    def _build_raster(self, cellsize, boundary, holes):
        """
        Helper function creates a target raster taking into account domain
        topology. Function is called during Gridded object initialization
        if 'cellsize' and 'boundary' are both specified.
        
        In order to build the target raster, this function relies heavily on
        matplotlib.path.Path() to make use of the domain boundary and hole
        information. The speed of this function is therefore primarily limited
        by the computational time needed to build each Path object and query
        each raster point.
        
        **Inputs**
            cellsize (float) : Cellsize of target raster. Smaller
                values will result in a higher-resolution grid and require
                more computational time.
            boundary (list or array) : Coordinates of the domain
                boundary in the same length-scale units as "cellsize".
                Coordinates should be specified as [[x1,y1],[x2,y2],...] pairs
                in either a list or array. Array dimensions should be (N,2).
            holes (list of lists or arrays) : Coordinates of 
                internal "holes" in the domain, specified as a list of holes.
                Each hole should be specified in the same format as the
                'boundary' input (i.e. either a list or array of x,y
                coordinates).
        **Outputs**
            After running, the Gridded.raster attribute will contain a
            binary basemap for later interpolation, with 1's in cells to be
            filled by interpolation, and 0's elsewhere.
        """
        # Store inputs as attributes for accessibility
        self.cellsize = float(cellsize)
        if type(boundary) == list:
            boundary = np.array(boundary) # Convert to array if necessary
        self.boundary = boundary
        # For convenience, store bbox as [xmin, xmax, ymin, ymax]
        self.bbox = [min(boundary[:,0]), max(boundary[:,0]),
                     min(boundary[:,1]), max(boundary[:,1])]
        if holes is not None:
            # Convert holes to arrays if necessary
            if type(holes[0]) == list:
                holes = [np.array(hole) for hole in holes]
        self.holes = holes
        
        # Construct the target x,y grid using the domain bbox and cellsize
        nx = int(np.ceil((self.bbox[1]-self.bbox[0])/self.cellsize)+1)
        xvect = np.linspace(self.bbox[0],
                            self.bbox[0]+self.cellsize*(nx-1), nx)
        ny = int(np.ceil((self.bbox[3]-self.bbox[2])/self.cellsize)+1)
        yvect = np.linspace(self.bbox[2],
                            self.bbox[2]+self.cellsize*(ny-1), ny)
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
        
        # Also save actual raster extent, slightly larger than bbox
        self.extent = [
            xvect[0]-0.5*self.cellsize,
            xvect[-1]+0.5*self.cellsize,
            yvect[0]-0.5*self.cellsize,
            yvect[-1]+0.5*self.cellsize,
        ]
        return

    def _from_raster(self, raster, extent):
        """
        Helper function determines domain geometry from an input raster.
        Function is called during Gridded object initialization if 'raster'
        and 'extent' are both specified.
        
        **Inputs**
            raster (array) : Binary array used as a basemap for
                interpolation, with 1's in cells that should be filled during
                interpolation, and 0's in cells that should be ignored.
                Distances will be computed within continuous clusters of cells
                in which the value =1.
            extent (list or array) : (x,y) extent of input raster,
                specified in the order [xmin, xmax, ymin, ymax]. Assumed to be
                the footprint of the raster, not the cell centroid coordinates
                of the boundary cells. If using rasterio, can be accessed
                through the 'bounds' attribute of rasterio.io.DatasetReader.
        **Outputs**
            After running, the Gridded.raster attribute will contain a
            binary basemap for later interpolation, with 1's in cells to be
            filled by interpolation, and 0's elsewhere.
        """
        # Save into attributes for access
        self.raster = raster.astype(int)
        self.extent = extent
        
        # Check cellsize
        cellsize_y = (extent[3] - extent[2])/raster.shape[0]
        cellsize_x = (extent[1] - extent[0])/raster.shape[1]
        # These need to be equal (within a certain tolerance)
        if (cellsize_y - cellsize_x) > 1e-2:
            raise ValueError(
                "Raster cellsize must be equal in x and y dimensions.\n"+\
                "Current sizes %s and %s are not equal." % (cellsize_x, 
                                                            cellsize_y))
        # If they are within tolerance, use mean cellsize
        self.cellsize = (cellsize_x + cellsize_y)/2
        
        # Determine bbox from extent/cellsize
        self.bbox = [
            extent[0]+0.5*self.cellsize,
            extent[1]-0.5*self.cellsize,
            extent[2]+0.5*self.cellsize,
            extent[3]-0.5*self.cellsize,
        ]
        return

    def interpolate(self, input_locations, input_values, n_nearest=3,
                    power=1, offsets=None, buffer=0, regularization=1e-8):
        """
        Method performs inverse-path-distance-weighted interpolation from
        input data locations onto target raster, using the number of neighbor
        points specified in the function call.
        
        This function relies heavily on scikit-fmm to compute geodesic
        distances within the domain topology. The speed of this function is
        therefore primarily limited by the computational time needed to compute
        each distance raster. Also, as an intermediate step of this process,
        each distance raster will be stored in an array of shape
        (y_dimension_length, x_dimension_length, len(input_locations)), so be
        wary of memory limits if cellsize is small or if a large number of
        input points are provided.
        
        **Inputs**
            input_locations (list or array) : Coordinates of the input data
                in the same length-scale units used to define domain boundary.
                Coordinates should be specified as [[x1,y1],[x2,y2],...] pairs
                in either a list or array. Array dimensions should be (N,2).
            input_values (list or array) : Data values at each input location.
            n_nearest (int, default=3) : Number of nearest input locations to
                use in the interpolation weighting. An input of "1" will return
                nearest-neighbor interpolation; larger values will return 
                "smoother" outputs.
            power (float, default=1) : Exponent on the distance parameter used
                when computing weights, i.e. each weight = (1/d^power)
            offsets (list or array, optional) : Optional constant values by
                which to increase the "zero" distance at each input
                location. Using values >0 at some location will decrease the
                weight of that location during interpolation. Can be used
                to reduce the weight of less certain inputs, or allow data
                locations from outside the enclosed boundary to be moved
                inside the boundary while maintaining the true distance.
            buffer (int, default=0) : Optional number of neighbor cells around
                each input location which is considered "zero" distance.
                Function requires all input locations to be "inside" the
                enclosed domain (or it will return a ValueError), so
                increasing the buffer by a few cells can be helpful if any
                inputs are very close to a boundary/hole. Applied uniformly to
                all locations, so does not affect weighting.
            regularization (float, default=1e-8) : Very small regularization
                parameter used to avoid dividing by zero when computing
                the inverse of distance.
        **Outputs**
            output (array) : Raster of interpolated values.
        """
        # Do some type checks on input values
        self.n_nearest = int(n_nearest)
        self.power = float(power)
        self.regularization = regularization
        if type(input_locations) == list:
            input_locations = np.array(input_locations) # Convert to array
        if type(input_values) == list:
            input_values = np.array(input_values) # Convert to array
        if offsets is None:
            offsets = np.zeros_like(input_values)
        else:
            if type(offsets) == list:
                offsets = np.array(offsets)
        
        # Initialize distance raster
        self.dist_from_each = np.ones((self.raster.shape[0],
                                       self.raster.shape[1],
                                       len(input_values)))*np.nan
        # Loop through input locations and compute distance raster for each
        for n in range(len(input_values)):
            x = input_locations[n,0]
            y = input_locations[n,1]
            
            # Convert from real x,y to raster indices ix,iy
            iy = int(self.raster.shape[0]-round((y-self.bbox[2])/self.cellsize))
            ix = int(round((x - self.bbox[0])/self.cellsize))
            
            # Perform fast marching
            phi = -1*self.raster
            mask = self.raster==0
            phi = np.ma.MaskedArray(phi, mask)
            if buffer > 0:
                phi[iy-buffer:iy+buffer,ix-buffer:ix+buffer] = 1
            else:
                phi[iy,ix] = 1
            try:
                # Actually compute distances
                dist = np.abs(skfmm.distance(phi)*self.cellsize) + offsets[n]
            except ValueError:
                raise ValueError("One or more locations are not within the "+\
                                 "enclosed boundary. Check locations or try "+\
                                 "increasing the buffer size")
            dist[mask] = np.nan # Mask out locations outside domain
            self.dist_from_each[:,:,n] = dist # Store result

        # For each cell, find indices of N nearest input locations
        self.arg_n_min = np.argsort(self.dist_from_each)[:,:,:n_nearest]

        # Perform inverse-path-distance-weighted interpolation
        # IDW formula is sum_i(v_i/d_i^P)/sum_i(1/d_i^P) for i=[1,N]
        numerator = np.zeros_like(self.raster, dtype=float) # init
        denominator = np.zeros_like(self.raster, dtype=float) # init
        # Loop through nearest locations
        for i in range(self.n_nearest):
            # Grab values at i'th nearest location
            vi = np.array([input_values[n] for n in self.arg_n_min[:,:,i]])
            vi.shape = self.raster.shape
            # Grab distance at i'th nearest location
            di = np.take_along_axis(self.dist_from_each,
                                    self.arg_n_min, axis=-1)[:,:,i]
            
            # Add to running tallies
            if self.power == 1:
                numerator += vi/(di + regularization)
                denominator += 1/(di + regularization)
            else:
                numerator += vi/(di**self.power + regularization)
                denominator += 1/(di**self.power + regularization)

        # Divide to finish interpolation
        self.output = numerator/denominator
        self.output[self.raster==0] = np.nan # Mask
        return self.output

    def reinterpolate(self, input_values):
        """
        If self.interpolate has already been run, this function allows you to
        interpolate again from the same locations using new values. Reduces
        the computational cost of rebuilding the interpolation from scratch.
        Aside from input_values, all other inputs must be the same.

        **Inputs**
            input_values (list or array) : Data values at each input location.
        **Outputs**
            output (array) : Raster of interpolated values.
        """
        if type(input_values) == list:
            input_values = np.array(input_values) # Convert to array

        # Perform inverse-path-distance-weighted interpolation
        # IDW formula is sum_i(v_i/d_i^P)/sum_i(1/d_i^P) for i=[1,N]
        numerator = np.zeros_like(self.raster, dtype=float) # init
        denominator = np.zeros_like(self.raster, dtype=float) # init
        # Loop through nearest locations
        for i in range(self.n_nearest):
            # Grab values at i'th nearest location
            vi = np.array([input_values[n] for n in self.arg_n_min[:,:,i]])
            vi.shape = self.raster.shape
            # Grab distance at i'th nearest location
            di = np.take_along_axis(self.dist_from_each,
                                    self.arg_n_min, axis=-1)[:,:,i]
            
            # Add to running tallies
            if self.power == 1:
                numerator += vi/(di + self.regularization)
                denominator += 1/(di + self.regularization)
            else:
                numerator += vi/(di**self.power + self.regularization)
                denominator += 1/(di**self.power + self.regularization)

        # Divide to finish interpolation
        self.output = numerator/denominator
        self.output[self.raster==0] = np.nan # Mask
        return self.output
