#
# ColabSeg - Interactive segmentation GUI
#
# Marc Siggel, December 2021

import pytest
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from colabseg.utilities import plane_fit, make_plot_array, R_2vect
from colabseg.new_gui_functions import *
import filecmp
import os


@pytest.mark.parametrize("filename", ["test_slices_zipped.mrc.gz", "test_slices.mrc"])
def test_load_tomogram(filename):
    """Test loading the load_tomogram"""
    data_structure = ColabSegData()
    data_structure.load_tomogram("./test_data/{}".format(filename))
    data_structure.convert_tomo()
    assert len(data_structure.position_list) == 104400 * 3
    assert len(data_structure.cluster_list_tv[0]) == 104400
    assert len(data_structure.cluster_list_tv) == 3


@pytest.fixture
def data_structure_fixture():
    data_structure = ColabSegData()
    data_structure.load_tomogram("./test_data/test_slices.mrc")
    data_structure.convert_tomo()
    # there is something weird going on with the H5 saving. It only works
    # when this rotation part is loaded...
    data_structure.get_lamina_rotation_matrix()
    data_structure.interpolate_membrane_rbf(cluster_index=0)
    data_structure.interpolate_membrane_rbf(cluster_index=1)
    data_structure.interpolate_membrane_rbf(cluster_index=2)
    return data_structure


def test_convert_tomogram(data_structure_fixture):
    """Check number of clusters"""
    assert len(data_structure_fixture.cluster_list_tv) == 3
    assert len(data_structure_fixture.cluster_list_fits) == 3


def test_get_lamina_rotation_matrix(data_structure_fixture):
    """get rotation matrix from xyz  points"""
    data_structure_fixture.get_lamina_rotation_matrix()
    reference_R = np.array([[1.0, 0.0, -0.0], [0.0, 1.0, 0.0], [0.0, -0.0, 1.0]])
    np.testing.assert_array_equal(data_structure_fixture.lamina_R, reference_R)


def test_trim_cluster_egdes_cluster(data_structure_fixture):
    """test trimming of points based on test file"""
    # calculate exact number of points which are subtracted to test quality
    assert len(data_structure_fixture.cluster_list_tv[0]) == 104400
    data_structure_fixture.trim_cluster_egdes_cluster(
        cluster_indices=[0], trim_min=10, trim_max=10
    )
    assert len(data_structure_fixture.cluster_list_tv[0]) == 102544


def test_reload_original_values(data_structure_fixture):
    """reload_original_values"""
    assert len(data_structure_fixture.cluster_list_tv) == 3
    data_structure_fixture.delete_cluster(cluster_index=0)
    data_structure_fixture.delete_cluster(cluster_index=0)
    assert len(data_structure_fixture.cluster_list_tv) == 1
    data_structure_fixture.reload_original_values()
    assert len(data_structure_fixture.cluster_list_tv) == 3
    assert len(data_structure_fixture.cluster_list_tv[0]) == 104400
    assert len(data_structure_fixture.position_list) == 104400 * 3


def test_delete_fit(data_structure_fixture):
    """Test delting a fit"""
    assert len(data_structure_fixture.cluster_list_fits) == 3
    data_structure_fixture.delete_fit(0)
    assert len(data_structure_fixture.cluster_list_fits) == 2


def test_interpolate_membrane_rbf(data_structure_fixture):
    """Test fitting of the rbfs"""
    assert len(data_structure_fixture.cluster_list_fits) == 3
    assert len(data_structure_fixture.cluster_list_fits[0]) == 103712


def test_delete_multiple_clusters(data_structure_fixture):
    """DOCSTRING"""
    assert len(data_structure_fixture.cluster_list_tv) == 3
    data_structure_fixture.delete_multiple_clusters(cluster_indices=[0, 1])
    assert len(data_structure_fixture.cluster_list_tv) == 1


def test_delete_cluster(data_structure_fixture):
    """DOCSTRING"""
    assert len(data_structure_fixture.cluster_list_tv) == 3
    data_structure_fixture.delete_cluster(0)
    assert len(data_structure_fixture.cluster_list_tv) == 2


def test_merge_clusters(data_structure_fixture):
    """DOCSTRING"""
    assert len(data_structure_fixture.cluster_list_tv) == 3
    data_structure_fixture.merge_clusters([1, 2])
    assert len(data_structure_fixture.cluster_list_tv) == 2
    assert len(data_structure_fixture.cluster_list_tv[1]) == 104400 * 2


def test_dbscan_clustering(data_structure_fixture):
    """Test DBSCAN clustering with open3d"""
    data_structure_fixture.merge_clusters([0, 1])
    data_structure_fixture.dbscan_clustering(cluster_index=1)
    assert len(data_structure_fixture.cluster_list_tv) == 3


def test_write_output_mrc(data_structure_fixture, tmp_path):
    """Test writing a new mrc file out"""
    positions = []
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test_output_file.mrc"

    positions = np.asarray(data_structure_fixture.position_list)
    assert len(positions) == 104400 * 3
    data_structure_fixture.write_output_mrc(positions=positions, output_filename=p)
    new_mrc = ColabSegData()
    # new_mrc.load_tomogram(p)
    # new_mrc.convert_tomo()
    # len(new_mrc.position_list) == len(data_structure_fixture.position_list)


# STATIC Methods
def test_write_txt(data_structure_fixture, tmp_path):
    """Test writing a txt file to disc"""
    positions = []
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test_output.txt"

    positions = np.asarray(data_structure_fixture.position_list)
    data_structure_fixture.write_txt(positions, p)
    len(np.loadtxt(p)) == 104400 * 3


def test_write_xyz(data_structure_fixture, tmp_path):
    """Test writing an xyz file to disc"""
    positions = []
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test_output.txt"

    positions = np.asarray(data_structure_fixture.position_list)
    data_structure_fixture.write_xyz(positions, p)

    with open(p) as file:
        lines = file.readlines()
    assert len(lines) == 104400 * 3 + 2
    # +2 because there are the header lines above the file


def test_save_hdf(data_structure_fixture, tmp_path):
    """Test saving the state as hdf5 file"""
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.h5"
    data_structure_fixture.save_hdf(p)

    data_structure_loaded = ColabSegData()
    data_structure_loaded.load_hdf(p)

    assert len(data_structure_loaded.position_list) == len(
        data_structure_fixture.position_list
    )
    assert len(data_structure_loaded.cluster_list_fits) == len(
        data_structure_fixture.cluster_list_fits
    )
    assert len(data_structure_loaded.cluster_list_tv) == len(
        data_structure_fixture.cluster_list_tv
    )
    assert len(data_structure_loaded.position_list) == 104400 * 3
    assert len(data_structure_loaded.cluster_list_tv[0]) == 104400


def test_load_hdf(data_structure_fixture):
    """Test loading a state as hdf5 file"""
    data_structure_loaded = ColabSegData()
    data_structure_loaded.load_hdf("./test_data/test.h5")
    assert len(data_structure_loaded.position_list) == len(
        data_structure_fixture.position_list
    )
    assert len(data_structure_loaded.cluster_list_fits) == len(
        data_structure_fixture.cluster_list_fits
    )
    assert len(data_structure_loaded.cluster_list_tv) == len(
        data_structure_fixture.cluster_list_tv
    )
    assert len(data_structure_loaded.position_list) == 104400 * 3
    assert len(data_structure_loaded.cluster_list_tv[0]) == 104400


def test_backup_step_to_previous(data_structure_fixture):
    """Test writing the backup lists"""
    assert len(data_structure_fixture.cluster_list_tv_previous) == 0
    data_structure_fixture.backup_step_to_previous()
    assert len(data_structure_fixture.cluster_list_tv_previous) == 3


def test_reload_previous_step(data_structure_fixture):
    """Test reloading a previous step from the GUI"""
    data_structure_fixture.backup_step_to_previous()
    data_structure_fixture.delete_cluster(0)
    assert len(data_structure_fixture.cluster_list_tv) == 2
    data_structure_fixture.reload_previous_step()
    assert len(data_structure_fixture.cluster_list_tv) == 3


def test_eigenvalue_outlier_removal(data_structure_fixture):
    """Test eigenvalue outlier removal."""
    data_structure = ColabSegData()
    data_structure.load_tomogram("./test_data/cross_half.mrc")
    data_structure.convert_tomo()
    data_structure.get_lamina_rotation_matrix()
    print(len(data_structure.cluster_list_tv[0]))
    data_structure.eigenvalue_outlier_removal(0, k_n=300, thresh=0.05)
    print(len(data_structure.cluster_list_tv[0]))
    data_structure_removed = ColabSegData()
    data_structure_removed.load_tomogram("./test_data/removed.mrc")
    data_structure_removed.convert_tomo()
    data_structure_removed.get_lamina_rotation_matrix()
    # data_structure.write_output_mrc(positions=np.asarray(data_structure.cluster_list_tv[0]), output_filename="./removed.mrc")
    np.testing.assert_array_equal(
        data_structure.cluster_list_tv[0], data_structure_removed.cluster_list_tv[0]
    )


def test_calculate_normals(data_structure_fixture):
    """DOCSTRING"""
    pass


def test_plain_fit_and_rotate_lamina():
    """DOCSTRING"""
    pass


def test_trim_cluster_egdes_fit(data_structure_fixture):
    """test trimming of points based on test file"""
    # calculate exact number of points which are subtracted to test quality
    pass


def test_statistical_outlier_removal(data_structure_fixture):
    """Test statistical outlier removal at the edge of the protien"""
    # take sample and add noise manually
    # planes + noisy points?
    pass


def test_crop_fit_around_membrane(data_structure_fixture):
    """DOCSTRING"""
    pass
