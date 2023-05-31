import pytest
import numpy as np
import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(os.path.dirname(__file__))))
import ipdw

# Test domain 1
cellsize1 = 5
boundary1 = [[30,0],[0,40],[50,100],[100,80],[90,20]]
holes1 = [[[40,30],[50,60],[70,30]]]
grid1 = ipdw.Gridded(cellsize1, boundary1, holes1)

def test_shape():
    assert grid1.raster.shape == (21,21)

def test_outside():
    assert grid1.raster[3,3] == 0

def test_inside():
    assert grid1.raster[12,5] == 1

def test_hole():
    assert grid1.raster[12,10] == 0

def test_area():
    area_fraction_inside = np.sum(grid1.raster)/grid1.raster.size
    assert area_fraction_inside == pytest.approx(0.528344, rel=0.01)

# Test domain 2
cellsize2 = 1
boundary2 = [[0,0],[100,0],[100,100],[0,100],[0,52],[70,80],[70,20],[0,48]]
grid2 = ipdw.Gridded(cellsize2, boundary2)
input_locations2 = [[2,45],[80,80]]
input_values2 = [7,3]

def test_nearest():
    output = grid2.interpolate(input_locations2, input_values2, n_nearest=1)
    assert output[2,55] - 3 < 1e-8

def test_region():
    output = grid2.interpolate(input_locations2, input_values2, n_nearest=1)
    assert (np.round(output[80:100,40:50]) == 7).all()

def test_output_equals_input():
    output = grid2.interpolate(input_locations2, input_values2, n_nearest=2)
    assert output[55,2] == pytest.approx(7, rel=0.01)

def test_output_elsewhere():
    output = grid2.interpolate(input_locations2, input_values2, n_nearest=2)
    assert output[80,80] == pytest.approx(4.664106, rel=0.01)

def test_global_average():
    output = grid2.interpolate(input_locations2, input_values2, n_nearest=2)
    assert np.nanmean(output) == pytest.approx(4.5039402, rel=0.01)

def test_reinterpolate():
    output_1 = grid2.interpolate(input_locations2, [7,3], n_nearest=2)
    output_2 = grid2.reinterpolate([6,2])
    diff = output_1 - output_2
    assert np.nanmean(diff) == pytest.approx(1, rel=0.01)

# Test domain 3
extent = [0, 150, 0, 75]
raster = np.array([[0,0,0,0,0,0],
                   [0,1,1,0,1,0],
                   [0,1,1,1,1,0]]).astype(float)
grid3 = ipdw.Gridded(raster=raster, extent=extent)

def test_from_raster():
    assert (grid3.raster == raster).all()

def test_cellsize():
    assert grid3.cellsize == pytest.approx(25, rel=1e-6)

def test_bbox():
    bbox = np.round(np.array(grid3.bbox),1)
    assert (bbox == np.array([12.5, 137.5, 12.5, 62.5])).all()