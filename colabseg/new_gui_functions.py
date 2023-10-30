#
# ColabSeg - Interactive segmentation GUI
#
# Marc Siggel, December 2021

import os
import subprocess

import h5py
import numpy as np
import open3d as o3d
import scipy.spatial as spatial
from scipy import interpolate
from scipy.spatial.distance import cdist
from pyntcloud import PyntCloud
from tqdm.notebook import tqdm

from .image_io import ImageIO
from .parametrization import PARAMETRIZATION_TYPE
from .utilities import (
    plane_fit,
    make_plot_array,
    R_2vect,
)

class ColabSegData(object):
    """Docstring for ColabSegData data structure framework."""

    def __init__(self):
        super(ColabSegData, self).__init__()
        # filename
        self.mrc_filename = None
        # TODO add load from metadata file function
        # mrc metadata

        self.shape = None
        self.pixel_size = None
        self.boxlength = None
        self.image_array = None
        self.lamina_R = None

        self.position_list = []
        self.intensity_list = []

        # these are the point cloud data original data
        # TODO write reset function for this
        self.cluster_list_tv = []
        self.cluster_list_fits = []

        # do not need this rotated list really?
        # self.cluster_rotated_positions = []
        # this list is used for lated stuff
        self.cluster_list_tv_previous = []
        self.cluster_list_fits_previous = []

        self.raw_tomogram_slice = []
        self.protein_positions_list = []
        self.analysis_properties = {}
        self.analysis_properties["minimal_distances"] = []
        self.analysis_properties["radii"] = []
        self.analysis_properties["normal_selection"] = []
        self.analysis_properties["surface_normals"] = []

    def load_tomogram(self, filename):
        """load_tomogram with pyto library"""
        self.mrc_filename = filename

        if self.mrc_filename.endswith(".mrc.gz") == True:
            split_name = os.path.splitext(self.mrc_filename)[0]
            # os.subprocess(["gunzip", self.mrc_filename])
            subprocess.run(["gunzip", self.mrc_filename])
            image = ImageIO()
            image.readMRC(split_name, memmap=False)
            # os.subprocess(["gzip", split_name])
            subprocess.run(["gzip", split_name])

        else:
            image = ImageIO()
            image.readMRC(self.mrc_filename, memmap=False)

        self.shape = image.shape
        # print(self.shape)
        self.pixel_size = list(np.asarray(image.pixel) * 10)
        self.boxlength = image.length
        self.image_array = image.data
        return

    def convert_tomo(self, step_size=1):
        """optimized version of conversion code"""
        unique = np.unique(self.image_array)
        for u in tqdm(unique[unique != 0]):
            where = np.where(self.image_array == u)
            for i in range(0, len(where[0])):
                self.position_list.append(
                    np.array(
                        [
                            int(where[0][i]) * self.pixel_size[0],
                            int(where[1][i]) * self.pixel_size[1],
                            int(where[2][i]) * self.pixel_size[2],
                        ]
                    )
                )
                self.intensity_list.append(u)

        for index in np.unique(self.intensity_list):
            subset = np.take(
                self.position_list, np.where(self.intensity_list == index)[0], axis=0
            )
            self.cluster_list_tv.append(subset)

        return

    def load_point_cloud(self, filename):
        """Load a plain point cloud from a txt file.
        In case a user has some preprocessed input on hand
        """
        input_data = np.loadtxt(filename)
        if len(input_data[0]) < 3:
            raise ValueError("Too few columns. Need 3 colums in data at least")
        elif len(input_data[0]) == 3:
            self.cluster_list_tv.append(input_data)
            self.position_list = input_data.tolist()
        elif len(input_data[0]) == 4:
            cluster_indices = input_data[:, 3]
            for cluster_index in cluster_indices:
                subset = input_data[input_data[:, 3] == cluster_index]
                self.cluster_list_tv.append(subset)
            self.position_list = input_data.tolist()
        else:
            raise ValueError("Too many columns. Need max 4 columns")
        self.shape = None
        # TODO currently we don't allow for a box if the data in not from a tomogram
        self.boxlength = [0, 0, 0]
        self.pixel_size = 13.4
        print("Loaded data from point cloud")
        return

    def load_stl_file(self, filename):
        """Load an stl file. This is a common output from IMOD as a mesh.
        For sake of consistency only the positions are loaded.
        """
        file_extension = filename.split(".")[-1]
        if file_extension != "stl":
            raise ValueError("Only STL files are supported at this point")
        mesh = o3d.io.read_triangle_mesh(filename)
        points = np.asarray(mesh.vertices)
        self.cluster_list_tv.append(points)
        self.position_list = points
        self.boxlength = [0, 0, 0]
        self.pixel_size = 13.4
        return

    def get_lamina_rotation_matrix(self, alignment_axis="z"):
        com = np.mean(self.position_list, axis=0)
        X, Y, Z = plane_fit(self.position_list - com, order=1)
        normal_vector = np.cross(
            [X[0][0] - X[0][1], Y[0][0] - Y[0][1], Z[0][0] - Z[0][1]],
            [X[0][0] - X[1][1], Y[0][0] - Y[1][1], Z[0][0] - Z[1][1]],
            axis=0,
        )
        if alignment_axis == "x":
            align_vector = np.array([1, 0, 0])
        elif alignment_axis == "y":
            align_vector = np.array([0, 1, 0])
        elif alignment_axis == "z":
            align_vector = np.array([0, 0, 1])
        else:
            raise Exception("Value for alignment axis must be x, y, or z")
        self.lamina_R = R_2vect(normal_vector, align_vector)
        # reverse rotation is transpose or inverse self.lamina_R.T
        return

    def plain_fit_and_rotate_lamina(self, backward=False):
        """Fits plane through extracted point cloud and rotates the positions"""
        com = np.mean(self.position_list, axis=0)

        for i in range(0, len(self.cluster_list_tv), 1):
            self.cluster_list_tv[i] = self.cluster_list_tv[i] - com
            if backward is False:
                self.cluster_list_tv[i] = np.dot(self.cluster_list_tv[i], self.lamina_R)
            if backward is True:
                self.cluster_list_tv[i] = np.dot(
                    self.cluster_list_tv[i], self.lamina_R.T
                )
            self.cluster_list_tv[i] = self.cluster_list_tv[i] + com

        for i in range(0, len(self.cluster_list_fits), 1):
            self.cluster_list_fits[i] = self.cluster_list_fits[i] - com
            if backward is False:
                self.cluster_list_fits[i] = np.dot(
                    self.cluster_list_fits[i], self.lamina_R
                )
            if backward is True:
                self.cluster_list_fits[i] = np.dot(
                    self.cluster_list_fits[i], self.lamina_R.T
                )
            self.cluster_list_fits[i] = self.cluster_list_fits[i] + com
        return

    def trim_cluster_egdes_cluster(
        self, cluster_indices, trim_min=0, trim_max=0, trim_axis="z"
    ):
        """Trim lamina at top and bottom"""
        if trim_axis == "x":
            trim_column = 0
        elif trim_axis == "y":
            trim_column = 1
        elif trim_axis == "z":
            trim_column = 2
        else:
            raise ValueError("Value for trim axis must be x, y, or z")
        for cluster_index in cluster_indices:
            trim_min_val = (
                np.min(self.cluster_list_tv[cluster_index][:, trim_column]) + trim_min
            )
            trim_max_val = (
                np.max(self.cluster_list_tv[cluster_index][:, trim_column]) - trim_max
            )

            trimmed = self.cluster_list_tv[cluster_index]
            trimmed = trimmed[trimmed[:, trim_column] > trim_min_val]
            trimmed = trimmed[trimmed[:, trim_column] < trim_max_val]
            self.cluster_list_tv[cluster_index] = trimmed
        return

    def trim_cluster_egdes_fit(
        self, fit_indices, trim_min=0, trim_max=0, trim_axis="z"
    ):
        """Trim lamina at top and bottom"""
        if trim_axis == "x":
            trim_column = 0
        elif trim_axis == "y":
            trim_column = 1
        elif trim_axis == "z":
            trim_column = 2
        else:
            raise ValueError("Value for trim axis must be x, y, or z")
        for cluster_index in fit_indices:
            trim_min = (
                np.min(self.cluster_list_fits[cluster_index][:, trim_column]) + trim_min
            )
            trim_max = (
                np.max(self.cluster_list_fits[cluster_index][:, trim_column]) - trim_max
            )

            trimmed = self.cluster_list_fits[cluster_index]
            trimmed = trimmed[trimmed[:, trim_column] > trim_min]
            trimmed = trimmed[trimmed[:, trim_column] < trim_max]
            self.cluster_list_fits[cluster_index] = trimmed
        return

    def statistical_outlier_removal(
        self, cluster_index=0, nb_neighbors=100, std_ratio=0.2
    ):
        """Remove the statistical outliers of the membrane"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.cluster_list_tv[cluster_index])
        # TODO add outlier removal to function parameters
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        # cl, ind = cl.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.1)

        point_array = np.asarray(cl.points)
        self.cluster_list_tv[cluster_index] = point_array
        return

    def dbscan_clustering(
        self, cluster_index=0, minimal_dbscsan_size=1000, eps=40, min_points=20
    ):
        """DBSCAN subclustering of the individual clusters after processing
        If the clusters are all numbered the same
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.cluster_list_tv[cluster_index])
        dbscan_labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        print("label count: {}".format(len(np.unique(dbscan_labels))))
        del self.cluster_list_tv[cluster_index]
        # self.cluster_list_tv = self.cluster_list_tv.pop(cluster_index)
        for label in np.unique(dbscan_labels):
            positions_to_write = np.asarray(pcd.points)[dbscan_labels == label]
            if label == -1:
                continue
            if len(positions_to_write) < minimal_dbscsan_size:
                continue
            # print("running on label {}".format(label))
            self.cluster_list_tv.append(positions_to_write)
        return

    def interpolate_membrane_rbf(
        self,
        cluster_index=0,
        skip_to_downsample="auto",
        functiontype="linear",
        smooth=5,
        directionality="xz",
    ):
        """Interpolate extended membrane using scipy RBF fit."""
        points = np.asarray(self.cluster_list_tv)[cluster_index]

        if skip_to_downsample == "auto":
            downsample = int(len(points) / 700)
        else:
            downsample = skip_to_downsample

        mod = points[::downsample]

        if directionality == "xz":
            X, Y, Z = mod[:, 2], mod[:, 1], mod[:, 0]  # n by 3 array
        elif directionality == "yz":
            X, Y, Z = mod[:, 0], mod[:, 2], mod[:, 1]
        elif directionality == "xy":
            X, Y, Z = mod[:, 0], mod[:, 1], mod[:, 2]

        if functiontype == "linear":
            spline = interpolate.Rbf(X, Y, Z, function="linear", smooth=smooth)
        elif functiontype == "multiquadric":
            spline = interpolate.Rbf(X, Y, Z, function="multiquadric", smooth=smooth)
        else:
            raise ValueError(
                "Functiontype argument may only be linear or multiquadric!"
            )
        # generate meshgrid
        xx, yy = make_plot_array(np.min(X), np.max(X), np.min(Y), np.max(Y))
        zz = spline(xx, yy)
        xx, yy, zz = np.ravel(xx), np.ravel(yy), np.ravel(zz)
        interpxyz = np.vstack((xx, yy, zz)).T
        # reorient the files
        if directionality == "xz":
            interpxyz[:, [0, 2]] = interpxyz[:, [2, 0]]
        elif directionality == "yz":
            interpxyz[:, [1, 2]] = interpxyz[:, [2, 1]]

        self.cluster_list_fits.append(interpxyz)

        return

    def crop_fit_around_membrane(
        self, cluster_index_tv=0, cluster_index_fit=0, distance_tolerance=50
    ):
        """use kdtree for distance calculation"""
        tree = spatial.cKDTree(self.cluster_list_tv[cluster_index_tv])
        groups = tree.query_ball_point(
            self.cluster_list_fits[cluster_index_fit], distance_tolerance
        )
        keep_list = np.unique([i for i, grp in enumerate(groups) if len(grp)])
        all_indices = np.arange(
            0, len(self.cluster_list_fits[cluster_index_fit]) - 1, 1
        )
        delete_list = list(set(all_indices) - set(keep_list))
        print(delete_list)
        self.cluster_list_fits[cluster_index_fit] = np.delete(
            self.cluster_list_fits[cluster_index_fit], delete_list, 0
        )
        return

    def write_output_mrc(self, positions, output_filename, offset=0):
        """writes a merged mrc file from the fit"""
        output_array = self.image_array * 0
        positions = positions / self.pixel_size
        for position in positions:
            x, y, z = position
            if x > self.boxlength[0] or x < 0:
                continue
            elif y > self.boxlength[1] or y < 0:
                continue
            elif z > self.boxlength[2] or z < 0:
                continue
            try:
                output_array[int(x) + offset, int(y) + offset, int(z) + offset] = 1
            except:
                continue

        out_image = ImageIO()
        out_image.write(file="{}".format(output_filename), data=output_array)
        return

    def compile_xyz_string(self, point_cloud):
        """Compiles an xyz file string for viz in py3DMol"""
        # TODO check if this needs to go into the main gui function
        xyz_file = "{}\n".format(len(np.asarray(point_cloud)))
        xyz_file += "pointcloud as xyz positions\n"
        for position in np.asarray(point_cloud):
            xyz_file += "C {} {} {}\n".format(position[0], position[1], position[2])
        return xyz_file

    def merge_clusters(self, cluster_indices=[]):
        """Merge two or more clusters by vstacking them"""
        if len(cluster_indices) < 2:
            return
        to_merge = []
        for i in cluster_indices:
            to_merge.append(self.cluster_list_tv[i])
        stacked_coordinates = np.vstack(np.asarray(to_merge))

        cluster_indices = np.sort(np.asarray(cluster_indices))[::-1]
        for i in cluster_indices:
            del self.cluster_list_tv[i]
        self.cluster_list_tv.append(stacked_coordinates)
        return

    def reload_original_values(self):
        """Reload the origial values"""
        self.cluster_list_tv = []
        for index in np.unique(self.intensity_list):
            subset = np.take(
                self.position_list, np.where(self.intensity_list == index)[0], axis=0
            )
            self.cluster_list_tv.append(subset)
        self.cluster_list_fits = []

    def delete_fit(self, fit_index):
        """delete a fit from data"""
        # self.cluster_list_fits = self.cluster_list_fits.pop(fit_index)
        del self.cluster_list_fits[fit_index]

    def delete_multiple_clusters(self, cluster_indices):
        """delete Multiple clusters"""
        cluster_indices = np.sort(np.asarray(cluster_indices))[::-1]
        for index in cluster_indices:
            self.delete_cluster(index)

    def delete_cluster(self, cluster_index):
        """delete a single cluster from data"""
        # self.cluster_list_fits = self.cluster_list_fits.pop(fit_index)
        del self.cluster_list_tv[cluster_index]

    def backup_step_to_previous(self):
        """takes the current value and overwrites the previous step"""
        self.cluster_list_tv_previous = self.cluster_list_tv.copy()
        self.cluster_list_fits_previous = self.cluster_list_fits.copy()

    def reload_previous_step(self):
        self.cluster_list_tv = self.cluster_list_tv_previous.copy()
        self.cluster_list_fits = self.cluster_list_fits_previous.copy()

    @staticmethod
    def write_xyz(point_cloud, output_filename):
        """Write an XYZ file for viewing in VMD"""
        xyz_file = "{}\n".format(len(np.asarray(point_cloud)))
        xyz_file += "pointcloud as xyz positions\n"
        for position in np.asarray(point_cloud):
            xyz_file += "C {} {} {}\n".format(position[0], position[1], position[2])
        text_file = open("{}".format(output_filename), "w")
        text_file.write(xyz_file)
        text_file.close()
        return

    @staticmethod
    def write_txt(point_cloud, output_filename):
        """Write a simple txt file to disc"""
        np.savetxt(output_filename, point_cloud)
        return

    def eigenvalue_outlier_removal(self, cluster_index, k_n=300, thresh=0.05):
        """Uses the covaiance based edge detection to remove points from the
        point cloud. Code adpated from:
        https://github.com/denabazazian/Edge_Extraction/blob/master/Difference_Eigenvalues.py
        """

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(self.cluster_list_tv[cluster_index])
        pcd1 = PyntCloud.from_instance("open3d", pcd_o3d)

        pcd_np = np.zeros((len(pcd1.points), 6))

        # find neighbors
        kdtree_id = pcd1.add_structure("kdtree")
        k_neighbors = pcd1.get_neighbors(k=k_n, kdtree=kdtree_id)

        # calculate eigenvalues
        ev = pcd1.add_scalar_field("eigen_values", k_neighbors=k_neighbors)

        x = pcd1.points["x"].values
        y = pcd1.points["y"].values
        z = pcd1.points["z"].values

        e1 = pcd1.points["e3(" + str(k_n + 1) + ")"].values
        e2 = pcd1.points["e2(" + str(k_n + 1) + ")"].values
        e3 = pcd1.points["e1(" + str(k_n + 1) + ")"].values

        sum_eg = np.add(np.add(e1, e2), e3)
        sigma = np.divide(e1, sum_eg)
        # sigma = e1
        sigma_value = sigma
        # pdb.set_trace()

        # visualize the edges
        sigma = sigma > thresh

        # Save the edges and point cloud
        thresh_min = sigma_value < thresh
        sigma_value[thresh_min] = 0
        thresh_max = sigma_value > thresh
        sigma_value[thresh_max] = 255

        pcd_np[:, 0] = x
        pcd_np[:, 1] = y
        pcd_np[:, 2] = z
        pcd_np[:, 3] = sigma_value

        face_np = np.delete(pcd_np, np.where(pcd_np[:, 3] != 0), axis=0)
        print(face_np)
        self.cluster_list_tv[cluster_index] = np.asarray(face_np)[:, :3]
        return

    def calculate_normals(self, cluster_indices=[], fit_indices=[]):
        """Calculate the normals of each voxel based on nearest neighbors
        uses a plane fit (very basic method)
        """
        self.analysis_properties["normal_selection"] = []
        self.analysis_properties["surface_normals"] = []
        all_positions = []
        for i in cluster_indices:
            all_positions.append(self.cluster_list_tv[i])
        for j in fit_indices:
            all_positions.append(self.cluster_list_fits[j])
        all_positions = np.vstack(all_positions)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_positions)
        pcd.estimate_normals()
        pcd.normalize_normals()
        pcd.orient_normals_consistent_tangent_plane(k=50)
        self.analysis_properties["normal_selection"] = np.asarray(all_positions)
        self.analysis_properties["surface_normals"] = np.asarray(pcd.normals)
        return

    def delete_normals(self):
        self.analysis_properties["normal_selection"] = []
        self.analysis_properties["surface_normals"] = []
        return

    def flip_normals(self):
        self.analysis_properties["surface_normals"] = self.analysis_properties[
            "surface_normals"
        ] * (-1)

    def interpolate_membrane_closed_surface(self, shape_type, cluster_index=0):
        """Least square fit for a perfect sphere and adding of points.
        For vesicles and spherical viruses.
        """
        interpxyz = PARAMETRIZATION_TYPE[shape_type].fit(np.asarray(self.cluster_list_tv)[cluster_index]).sample(100)
        self.cluster_list_fits.append(interpxyz)
        return

    def save_hdf(self, filename):
        """write all class variables into hdf5 file format"""
        with h5py.File(filename, "w") as hf:
            hf.attrs["mrc_filename"] = self.mrc_filename
            hf.attrs["shape"] = self.shape
            hf.attrs["pixel_size"] = self.pixel_size
            hf.attrs["boxlength"] = self.boxlength
            hf.attrs["lamina_R"] = self.lamina_R
            hf.create_dataset("image_array", data=self.image_array, compression="gzip")
            hf.create_dataset(
                "position_list", data=self.position_list, compression="gzip"
            )
            hf.create_dataset(
                "intensity_list", data=self.intensity_list, compression="gzip"
            )
            hf.create_dataset(
                "protein_positions_list",
                data=self.protein_positions_list,
                compression="gzip",
            )

            for key in self.analysis_properties.keys():
                hf.create_dataset(
                    "{}".format(key),
                    data=self.analysis_properties[str(key)],
                    compression="gzip",
                )

            cluster_list_tv = hf.create_group("cluster_list_tv")
            for idx, arr in enumerate(self.cluster_list_tv):
                cluster_list_tv.create_dataset(str(idx), data=arr, compression="gzip")

            cluster_list_fits = hf.create_group("cluster_list_fits")
            for idx, arr in enumerate(self.cluster_list_fits):
                cluster_list_fits.create_dataset(str(idx), data=arr, compression="gzip")

            cluster_list_tv_previous = hf.create_group("cluster_list_tv_previous")
            for idx, arr in enumerate(self.cluster_list_tv_previous):
                cluster_list_tv_previous.create_dataset(
                    str(idx), data=arr, compression="gzip"
                )

            cluster_list_fits_previous = hf.create_group("cluster_list_fits_previous")
            for idx, arr in enumerate(self.cluster_list_fits_previous):
                cluster_list_fits_previous.create_dataset(
                    str(idx), data=arr, compression="gzip"
                )
        return

    def load_hdf(self, filename):
        """read all class variables from hdf5 file format and populate instance"""
        with h5py.File(filename, "r") as hf:
            self.mrc_filename = hf.attrs["mrc_filename"]
            self.shape = hf.attrs["shape"]
            self.pixel_size = hf.attrs["pixel_size"]
            self.boxlength = hf.attrs["boxlength"]
            self.image_array = np.asarray(hf["image_array"])
            self.lamina_R = hf.attrs["lamina_R"]
            self.intensity_list = np.asarray(hf["intensity_list"])
            self.position_list = np.asarray(hf["position_list"])

            try:
                for key in self.analysis_properties.keys():
                    self.analysis_properties[key] = np.asarray(hf[str(key)]).tolist()
            except:
                pass

            self.cluster_list_tv = []
            for i in hf["cluster_list_tv"]:
                self.cluster_list_tv.append(np.asarray(hf["cluster_list_tv"][i]))

            self.cluster_list_fits = []
            for i in hf["cluster_list_fits"]:
                self.cluster_list_fits.append(np.asarray(hf["cluster_list_fits"][i]))

            self.cluster_list_tv_previous = []
            for i in hf["cluster_list_tv_previous"]:
                self.cluster_list_tv_previous.append(
                    np.asarray(hf["cluster_list_tv_previous"][i])
                )

            self.cluster_list_fits_previous = []
            for i in hf["cluster_list_fits_previous"]:
                self.cluster_list_fits_previous.append(
                    np.asarray(hf["cluster_list_fits_previous"][i])
                )

        return

    def extract_slice(self, filename, slice="center"):
        """Extract a slice from the original tomogram and visualize"""
        # TODO add manual selection of slices; currently disabled
        image = ImageIO()
        image.readMRC(filename, memmap=False)
        center_slice = int(image.shape[2] / 2)
        slice = np.flip(image.data[:, :, center_slice].T)
        self.raw_tomogram_slice = slice
        return

    # new analysis features for protein tab
    def load_protein_position(self, filename):
        """Load a protein position list (XYZ file)"""
        extension = filename.split(".")[-1]
        if extension == "csv":
            protein_positions = np.genfromtxt(filename, delimiter=",")
        elif extension == "txt":
            protein_positions = np.loadtxt(filename)
        elif extension == "mrc":
            pass
            # TODO load from MRC file
            # protein_positions =
        else:
            raise ValueError(
                "Only csv and txt file without headers can be loaded. Please provide a different file."
            )
        self.protein_positions_list = []
        self.protein_positions_list.append(protein_positions)
        return

    def delete_protein_position(self):
        """delete the positions from the gui"""
        self.protein_positions_list = []
        return

    def analyze_protein_membrane_min_distance(self, cluster_indices=[], fit_indices=[]):
        """analyze distance between particle positions and a membrane segmentation"""
        all_minimal_distances = []
        all_considered_membranes = []
        for i in cluster_indices:
            all_considered_membranes.append(self.cluster_list_tv[i])
        for j in fit_indices:
            all_considered_membranes.append(self.cluster_list_fits[j])
        all_considered_membranes = np.vstack(all_considered_membranes)

        print("running distance analysis")
        for point in self.protein_positions_list[0]:
            minimal_distances = np.min(cdist([point], all_considered_membranes), axis=1)
            all_minimal_distances.append(minimal_distances)
        print("finalize and plot")
        self.analysis_properties["minimal_distances"] = np.hstack(all_minimal_distances)
        # self.analysis_properties["minimal_distances"] = np.min(np.vstack(all_minimal_distances),axis=0)
        return

    def get_selected_sphere_radii(self, fit_indices=[]):
        """return a list of sphere radii based on a selection"""
        self.analysis_properties["radii"] = []
        radii = []
        for j in fit_indices:
            mean_point = np.mean(self.cluster_list_fits[j], axis=0)
            radius = np.linalg.norm(mean_point - self.cluster_list_fits[j][0])
            radii.append(radius)
        self.analysis_properties["radii"] = radii
        return

    @staticmethod
    def save_values_txt(value_array, filename):
        np.savetxt(filename, value_array)
        return


# TODO add another clustering method?
# TODO give access to other open3d features
# TODO add new protein stuff to state file hdf5 for saving etc.
