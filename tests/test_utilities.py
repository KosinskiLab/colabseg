#
# ColabSeg - Interactive segmentation GUI
#
# Marc Siggel, December 2021

import pytest
import numpy as np
import scipy
from colabseg.utilities import *


def test_plane_fit():
    """Test the plane fit"""
    xy_plane_points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [2, 2, 0],
            [3, 3, 0],
            [2, 0, 0],
            [0, 2, 0],
            [3, 0, 0],
            [0, 3, 0],
        ]
    )

    X, Y, Z = plane_fit(xy_plane_points)
    normal_vector = np.cross(
        [X[0][0] - X[0][1], Y[0][0] - Y[0][1], Z[0][0] - Z[0][1]],
        [X[0][0] - X[1][1], Y[0][0] - Y[1][1], Z[0][0] - Z[1][1]],
        axis=0,
    )

    np.testing.assert_array_equal(normal_vector, np.array([0, 0, 1]))


def test_make_plot_array():
    """Test plot array"""
    meshgrid = (
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        ),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0, 4.0],
            ]
        ),
    )
    result = make_plot_array(0, 5, 0, 5, interp=1)

    np.testing.assert_array_equal(meshgrid, result)


def test_R_2vect():
    """Test rotation matrix"""
    comparison = np.array(
        [
            [0.00000000e00, -1.00000000e00, 0.00000000e00],
            [1.00000000e00, 0.00000000e00, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )

    R_test = R_2vect(np.array([1, 0, 0]), np.array([0, 1, 0]))

    np.testing.assert_array_almost_equal(comparison, R_2vect([1, 0, 0], [0, 1, 0]))
    np.testing.assert_array_equal(comparison.astype(int), R_test.astype(int))


def test_create_sphere_points():
    """Test creation of sphere points. Compare with a sample file"""
    sphere_points = np.loadtxt("./test_data/sphere_test_position_111.txt")

    r = 1
    x0 = 1
    y0 = 1
    z0 = 1
    generated_sphere_points = create_sphere_points(r, x0, y0, z0)

    np.testing.assert_array_almost_equal(sphere_points, generated_sphere_points)


@pytest.mark.parametrize(
    "sphere_file, r0_result, center_result",
    [
        ("missing_wedge_sphere.txt", 1, [0, 0, 0]),
        ("sphere_test_position_000.txt", 1, [0, 0, 0]),
        ("sphere_test_position_111.txt", 1, [1, 1, 1]),
    ],
)
def test_lstsq_sphere_fitting(sphere_file, r0_result, center_result):
    """Test the least squared fitting of a sphere
    test based on a unit sphere.
    """
    sphere_points = np.loadtxt("./test_data/{}".format(sphere_file))

    r0, x0, y0, z0 = lstsq_sphere_fitting(sphere_points)

    np.testing.assert_almost_equal(r0, r0_result)
    np.testing.assert_almost_equal(x0, center_result[0])
    np.testing.assert_almost_equal(y0, center_result[1])
    np.testing.assert_almost_equal(z0, center_result[2])
