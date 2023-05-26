import pytest
import numpy as np
import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(os.path.dirname(__file__))))
import ipdw

# Construct toy problem
cellsize = 5
boundary = [[30,0],[0,40],[50,100],[100,80],[90,20]]
holes = [[[40,30],[50,60],[70,30]]]

grid = ipdw.Gridded(cellsize, boundary, holes)

def test_shape():
    assert grid.raster.shape == (21,21)

def test_outside():
    assert grid.raster[3,3] == 0

def test_inside():
    assert grid.raster[12,5] == 1

def test_hole():
    assert grid.raster[12,10] == 0

def test_area():
    area_fraction_inside = np.sum(grid.raster)/grid.raster.size
    assert area_fraction_inside == pytest.approx(0.528344, rel=0.01)
