import pytest
import numpy as np
import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(os.path.dirname(__file__))))
import ipdw

# Test domain A
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

# Test domain B
cellsize = 1
boundary = [[0,0],[100,0],[100,100],[0,100],[0,52],[70,80],[70,20],[0,48]]
grid = ipdw.Gridded(cellsize, boundary)
input_locations = [[2,45],[80,80]]
input_values = [7,3]

def test_nearest():
    output = grid.interpolate(input_locations, input_values, n_nearest=1)
    assert output[2,55] == 3

def test_region():
    output = grid.interpolate(input_locations, input_values, n_nearest=1)
    assert (np.round(output[80:100,40:50]) == 7).all()

def test_output_equals_input():
    output = grid.interpolate(input_locations, input_values, n_nearest=2)
    assert output[55,2] == pytest.approx(7, rel=0.01)

def test_output_elsewhere():
    output = grid.interpolate(input_locations, input_values, n_nearest=2)
    assert output[80,80] == pytest.approx(4.664106, rel=0.01)

def test_global_average():
    output = grid.interpolate(input_locations, input_values, n_nearest=2)
    assert np.nanmean(output) == pytest.approx(4.5039402, rel=0.01)

def test_reinterpolate():
    output_1 = grid.interpolate(input_locations, [7,3], n_nearest=2)
    output_2 = grid.reinterpolate([6,2])
    diff = output_1 - output_2
    assert np.nanmean(diff) == pytest.approx(1, rel=0.01)