#
# ColabSeg - Interactive segmentation GUI
#
# Marc Siggel, December 2021
# from pyto.io.image_io import ImageIO
import math
import scipy

import numpy as np

def plane_fit(atom_coordinates, order=1):
    """
    Fit a plane to given atom coordinates.

    Parameters
    ----------
    atom_coordinates : ndarray
        A 2D array of shape (n, 3) where each row is a 3D coordinate (x, y, z).
    order : int, optional
        Order of the plane fit.
        1 indicates linear and 2 indicates quadratic. Default is 1.

    Returns
    -------
    X : ndarray
        2D array of x-coordinates for the meshgrid.
    Y : ndarray
        2D array of y-coordinates for the meshgrid.
    Z : ndarray
        2D array of z-coordinates for the meshgrid.

    Notes
    -----
    The function uses least squares to fit a plane to the input coordinates.
    """
    data = atom_coordinates
    # print(data[:, 0])
    ymin = np.amin(data[:, 1])
    ymax = np.amax(data[:, 1])
    xmin = np.amin(data[:, 0])
    xmax = np.amax(data[:, 0])
    # print(xmin, xmax, ymin, ymax)
    X, Y = np.meshgrid(np.arange(xmin, xmax, 1), np.arange(ymin, ymax, 1))
    XX = X.flatten()
    YY = Y.flatten()

    order = 1  # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

        # evaluate it on grid
        Z = C[0] * X + C[1] * Y + C[2]

        # or expressed using matrix/vector product
        # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[
            np.ones(data.shape[0]),
            data[:, :2],
            np.prod(data[:, :2], axis=1),
            data[:, :2] ** 2,
        ]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

        # evaluate it on a grid
        Z = np.dot(
            np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX**2, YY**2], C
        ).reshape(X.shape)

    return X, Y, Z


def make_plot_array(xmin, xmax, ymin, ymax, interp=0.1):
    """
    Create a meshgrid based on the given boundaries and interpolation factor.

    Parameters
    ----------
    xmin : float
        Minimum x value.
    xmax : float
        Maximum x value.
    ymin : float
        Minimum y value.
    ymax : float
        Maximum y value.
    interp : float, optional
        Interpolation factor, less than 1 for downsampling. Default is 0.1.

    Returns
    -------
    xx : ndarray
        2D array of x-coordinates for the meshgrid.
    yy : ndarray
        2D array of y-coordinates for the meshgrid.
    """
    x = np.arange(xmin, xmax, 1.0 / interp)
    y = np.arange(ymin, ymax, 1.0 / interp)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def R_2vect(vector_orig, vector_fin):
    """
    Calculate the rotation matrix to rotate from one vector to another.

    Parameters
    ----------
    vector_orig : ndarray
        Original vector, shape (3,).
    vector_fin : ndarray
        Final vector, shape (3,).

    Returns
    -------
    R : ndarray
        Rotation matrix of shape (3, 3).

    Notes
    -----
    The rotation matrix is determined based on the cross product of the two vectors
    and the rotation angle.
    """

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / np.linalg.norm(vector_orig)
    vector_fin = vector_fin / np.linalg.norm(vector_fin)

    # The rotation axis (normalised).
    axis = np.cross(vector_orig, vector_fin)
    axis_len = np.linalg.norm(axis)
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
    R[0, 0] = 1.0 + (1.0 - ca) * (x**2 - 1.0)
    R[0, 1] = -z * sa + (1.0 - ca) * x * y
    R[0, 2] = y * sa + (1.0 - ca) * x * z
    R[1, 0] = z * sa + (1.0 - ca) * x * y
    R[1, 1] = 1.0 + (1.0 - ca) * (y**2 - 1.0)
    R[1, 2] = -x * sa + (1.0 - ca) * y * z
    R[2, 0] = -y * sa + (1.0 - ca) * x * z
    R[2, 1] = x * sa + (1.0 - ca) * y * z
    R[2, 2] = 1.0 + (1.0 - ca) * (z**2 - 1.0)
    return R


def create_sphere_points(radius, x0, y0, z0, n=72):
    """
    Create 3D points that fall on a defined sphere.

    Parameters
    ----------
    radius : float
        Radius of the sphere.
    x0 : float
        X-coordinate of the sphere center.
    y0 : float
        Y-coordinate of the sphere center.
    z0 : float
        Z-coordinate of the sphere center.
    n : int, optional
        Number of angle slices. 2*pi angle range is evenly split into n slices.
        Default is 72.

    Returns
    -------
    positions_xyz : ndarray
        Coordinates of 3D points on the sphere, shape (n*n, 3).

    References
    ----------
    .. [1] https://gist.github.com/WuyangLI/eb4cf9d2df6680ff16255732efd0d242
    """
    sp = np.linspace(0, 2.0 * np.pi, num=n)
    nx = sp.shape[0]
    u = np.repeat(sp, nx)
    v = np.tile(sp, nx)
    x = x0 + np.cos(u) * np.sin(v) * radius
    y = y0 + np.sin(u) * np.sin(v) * radius
    z = z0 + np.cos(v) * radius
    positions_xyz = np.column_stack([x, y, z])
    return positions_xyz


def lstsq_sphere_fitting(positions):
    """
    Fit a set of 3D points to a sphere using least squares.

    Parameters
    ----------
    positions : ndarray
        A 2D array of shape (n, 3) where each row represents a 3D coordinate (x, y, z).

    Returns
    -------
    float
        Radius of the fitted sphere.
    float
        X-coordinate of the sphere center.
    float
        Y-coordinate of the sphere center.
    float
        Z-coordinate of the sphere center.

    Notes
    -----
    The function constructs a matrix A based on the positions and uses least squares
    to determine the sphere's parameters.

    References
    ----------
    .. [1] https://gist.github.com/WuyangLI/4bf4b067fed46789352d5019af1c11b2
    """
    # add column of ones to pos_xyz to construct matrix A
    pos_xyz = positions
    row_num = pos_xyz.shape[0]
    A = np.ones((row_num, 4))
    A[:, 0:3] = pos_xyz

    # construct vector f
    f = np.sum(np.multiply(pos_xyz, pos_xyz), axis=1)

    sol, residules, rank, singval = np.linalg.lstsq(A, f)

    # solve the radius
    radius = math.sqrt(
        (sol[0] * sol[0] / 4.0)
        + (sol[1] * sol[1] / 4.0)
        + (sol[2] * sol[2] / 4.0)
        + sol[3]
    )

    return radius, sol[0] / 2.0, sol[1] / 2.0, sol[2] / 2.0
