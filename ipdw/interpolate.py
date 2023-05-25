#!/usr/bin/env python
import numpy as np
import matplotlib
import skfmm

# Maybe this should be a class called Raster with interpolate as a method
# used to create an ipdw.raster object
def build_raster(extent, cellsize, boundary, holes=None):
    """
    Function creates a target raster taking into account domain topology
    """
    return

def interpolate(raster, input_locations, input_values, extent, n_nearest=3):
    """
    Function performs interpolation from discete data points onto target raster
    """
    return
