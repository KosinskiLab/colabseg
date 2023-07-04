#
# ColabSeg - Interactive segmentation GUI
#
# Marc Siggel, December 2021

import numpy as np
import open3d as o3d
import os
import math

import open3d as o3d
from pyto.io.image_io import ImageIO



def load_and_binarize_seg(filename, min_val=0):
    """load an mrc file and binarize it"""
    seg = io.load_tomo(filename)
    binary_seg = (seg > min_val).astype(int)
    return seg, binary_seg


def convert_tomo(seg, binary_seg, cbinsize, step=1):
    """convert tomo"""
    position_list = []
    intensity_list = []
    nx, ny, nz = binary_seg.shape
    for x in range(0, nx, step):
        for y in range(0, ny, step):
            for z in range(0, nz, step):
                if binary_seg[x, y, z] == 1:
                    position_list.append(
                        np.array([x * cbinsize, y * cbinsize, z * cbinsize])
                    )
                    intensity_list.append(seg[x, y, z])
    position_list = np.asarray(position_list)
    intensity_list = np.asarray(intensity_list)
    return position_list, intensity_list


def get_bbox(mrc_array, cbinsize):
    """return bounding box size"""
    nx, ny, nz = mrc_array.shape
    bbox = np.array([[0, 0, 0], [nx * cbinsize, ny * cbinsize, nz * cbinsize]])
    return bbox


def write_txt(output_name, position_array):
    """Write Txt file for processing"""
    np.savetxt("{}.txt".format(output_name), position_array)
    return


def array_to_open3d(xyz_array, downsample=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_array)
    # if downsample is not False:
    #     pcd = pcd.voxel_down_sample(voxel_size=downsample)
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100,std_ratio=1.0)
    # cl, ind = cl.remove_radius_outlier(nb_points=20, radius=200)
    # xyz_load = np.asarray(pcd_load.points)
    return pcd


def com_cluster_points(position_list, intensity_list, cutoff):
    """com clustering of points and labeling accoridng to intensity_list"""
    tag = np.zeros(len(position_list))
    com_list = []
    cluster_index = []
    # tag.append()
    # while loop instead
    acc = False
    while acc == False:
        if len(np.where(tag == 0)[0]) == 0:
            acc = True
            break
        randint = np.random.randint(0,len(np.where(tag == 0)[0]))
        randint = np.where(tag == 0)[0][randint]
        print(len(np.where(tag == 0)[0]))
        #randint = np.random.randint(0, len(position_list))
        #if 0 not in tag:
        #    acc = True
        if tag[randint] != 0:
            continue
        pos = position_list[randint]
        dist_arr = distance_array(position_list[randint],position_list)
        indices = np.where(dist_arr[0] < cutoff)[0]
        pos_cluster = position_list[dist_arr[0] < cutoff]
        cluster_labels = intensity_list[dist_arr[0] < 80]
        cluster_label = np.bincount(cluster_labels).argmax()
        center = np.average(pos_cluster,axis=0)
        for index in indices:
            tag[index] = 1
        com_list.append(center)
        cluster_index.append(cluster_label)

    cluster_index_list = np.asarray(cluster_index)
    com_list = np.asarray(com_list)
    return com_list, cluster_index_list


def separate_clusters(com_list, cluster_index_list):
    """Separate clusters based on TV input"""
    list_of_meshes = []

    for index in np.unique(cluster_index_list):
        subset = np.take(com_list, np.where(cluster_index_list == index)[0], axis=0)
        pcd = array_to_open3d(subset)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=20)
        )
        pcd.orient_normals_consistent_tangent_plane(10)
        # estimate radius for rolling ball
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 1.0 * avg_dist
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius * 0.5, radius * 1.0, radius * 1.5, radius * 2.0]))
        o3d.io.write_triangle_mesh("./output_cluster_{}.ply".format(index), mesh)
        write_xyz(np.asarray(pcd.points), "./output_cluster_{}.xyz".format(index))
        np.savetxt("./output_cluster_{}.txt".format(index), np.asarray(pcd.points))
        list_of_meshes.append(mesh)

    return list_of_meshes


def get_all_edge_lengths(mesh):
    """"calculate all edge lengths"""
    position_array = mesh.points()
    edge_vertices_array = ev.indices()
    edge_lengths = []
    for edge in edge_vertices_array:
        el = np.linalg.norm((position_array[[edge[1]]][0] - position_array[[edge[0]]][0]))
        edge_lengths.append(el)
    return edge_lengths


def write_xyz(point_cloud, filename):
    """Write an XYZ file for viewing in VMD"""
    xyz_file = "{}\n".format(len(np.asarray(point_cloud)))
    xyz_file += "pointcloud_as_xyz_positions\n"
    for position in np.asarray(point_cloud):
        xyz_file += "C {} {} {}\n".format(position[0], position[1], position[2])
    text_file = open("{}".format(filename), "w")
    text_file.write(xyz_file)
    text_file.close()
    return


# @jit(nopython=True)
def angle_between_degree(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(math.acos(max(-1.0, min(calc_dot(v1_u, v2_u), 1.0))))


def R_2vect(vector_orig, vector_fin):
    """Calculate the rotation matrix required to rotate from one vector to another.
    """

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / norm(vector_orig)
    vector_fin = vector_fin / norm(vector_fin)

    # The rotation axis (normalised).
    axis = np.cross(vector_orig, vector_fin)
    axis_len = norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = math.acos(np.dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = math.cos(angle)
    sa = math.sin(angle)

    # Calculate the rotation matrix elements.
    R = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    R[0, 0] = 1.0 + (1.0 - ca) * (x ** 2 - 1.0)
    R[0, 1] = -z * sa + (1.0 - ca) * x * y
    R[0, 2] = y * sa + (1.0 - ca) * x * z
    R[1, 0] = z * sa + (1.0 - ca) * x * y
    R[1, 1] = 1.0 + (1.0 - ca) * (y ** 2 - 1.0)
    R[1, 2] = -x * sa + (1.0 - ca) * y * z
    R[2, 0] = -y * sa + (1.0 - ca) * x * z
    R[2, 1] = x * sa + (1.0 - ca) * y * z
    R[2, 2] = 1.0 + (1.0 - ca) * (z ** 2 - 1.0)
    return R


# @jit(nopython=True)
def calc_cross(a, b, c):
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c


# @jit(nopython=True)
def calc_norm(a):
    return math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)


# @jit(nopython=True)
def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / calc_norm(vector)


# @jit(nopython=True)
def calc_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# @jit(nopython=True)
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.acos(max(-1.0, min(calc_dot(v1_u, v2_u), 1.0)))


def print_data_xyz(filename, mesh):
    """writes or appends a xyz file with the current simulation step."""
    exists = os.path.isfile("{}.xyz".format(filename))
    if exists == True:
        # print("appending file")
        f = open("{}.xyz".format(filename), "a+")
    else:
        # print("writing new file")
        f = open("{}.xyz".format(filename), "w+")
    f.write("{}\n".format(len(mesh.vertices())))
    f.write("#test\n")
    for i in mesh.vertices():
        f.write(
            "C\t{}\t{}\t{}\n".format(
                mesh.point(i)[0], mesh.point(i)[1], mesh.point(i)[2]
            )
        )
    f.close()
