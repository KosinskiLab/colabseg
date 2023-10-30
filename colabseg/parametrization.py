from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy import optimize


class Parametrization(ABC):
    """
    A strategy class to represent parametrizations of point clouds
    """

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, positions: np.ndarray, *args, **kwargs):
        """
        Fit a parametrization to a point cloud.

        Parameters
        ----------
        positions : np.ndarray
            Point coordinates with shape (n x 3)
        *args : List
            Additional arguments
        **kwargs : Dict
            Additional keywoard arguments.

        Returns
        -------
        Tuple
            Parametrization parameters
        """

    @abstractmethod
    def sample(self, n_samples: int, *args, **kwargs):
        """
        Samples points from the surface of the parametrization.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw
        *args : List
            Additional arguments
        **kwargs : Dict
            Additional keywoard arguments.

        Returns
        -------
        np.ndarray
            Sampled points.
        """


class Sphere(Parametrization):
    """
    Parametrize a point cloud as sphere.
    """

    def __init__(self, radius: np.ndarray, center: np.ndarray):
        """
        Initialize the Ellipsoid parametrization.

        Parameters
        ----------
        radius : np.ndarray
            Radius of the sphere
        center : np.ndarray
            Center of the sphere along each axis.
        """
        self.radius = radius
        self.center = center

    @classmethod
    def fit(cls, positions: np.ndarray) -> "Sphere":
        """
        Fit an sphere to a set of 3D points.

        Parameters
        ----------
        positions : np.ndarray
            Point coordinates with shape (n x 3)

        Returns
        -------
        Sphere
            Class instance with fitted parameters.

        References
        ----------
        .. [1]  https://gist.github.com/WuyangLI/4bf4b067fed46789352d5019af1c11b2

        """
        # add column of ones to pos_xyz to construct matrix A
        pos_xyz = positions
        row_num = pos_xyz.shape[0]
        A = np.ones((row_num, 4))
        A[:, 0:3] = pos_xyz

        # construct vector f
        f = np.sum(np.multiply(pos_xyz, pos_xyz), axis=1)

        sol, residules, rank, singval = np.linalg.lstsq(A, f, rcond=None)

        # solve the radius
        radius = np.sqrt(
            (sol[0] * sol[0] / 4.0)
            + (sol[1] * sol[1] / 4.0)
            + (sol[2] * sol[2] / 4.0)
            + sol[3]
        )
        return cls(
            radius=radius, center=np.array([sol[0] / 2.0, sol[1] / 2.0, sol[2] / 2.0])
        )

    def sample(
        self, n_samples: int, radius: np.ndarray = None, center: np.ndarray = None
    ) -> np.ndarray:
        """
        Samples points from the surface of a sphere.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw
        radius : np.ndarray, optional
            Radius of the sphere
        center : np.ndarray, optional
            Center of the sphere along each axis

        Returns
        -------
        np.ndarray
            Sampled points.
        """
        center = self.center if center is None else center
        radius = self.radius if radius is None else radius

        sp = np.linspace(0, 2.0 * np.pi, num=n_samples)
        x0, y0, z0 = center
        nx = sp.shape[0]
        u = np.repeat(sp, nx)
        v = np.tile(sp, nx)
        x = x0 + np.cos(u) * np.sin(v) * radius
        y = y0 + np.sin(u) * np.sin(v) * radius
        z = z0 + np.cos(v) * radius
        positions_xyz = np.column_stack([x, y, z])
        return positions_xyz


class Ellipsoid(Parametrization):
    """
    Parametrize a point cloud as ellipsoid.
    """

    def __init__(self, radii: np.ndarray, center: np.ndarray, orientations: np.ndarray):
        """
        Initialize the Ellipsoid parametrization.

        Parameters
        ----------
        radii : np.ndarray
            Radii of the ellipse along each axis
        center : np.ndarray
            Center of the ellipse along each axis
        orientations : np.ndarray
            Square orientation matrix
        """
        self.radii = radii
        self.center = center
        self.orientations = orientations

    @classmethod
    def fit(cls, positions) -> "Ellipsoid":
        """
        Fit an ellipsoid to a set of 3D points.

        Parameters
        ----------
        positions: np.ndarray
            Point coordinates with shape (n x 3)

        Returns
        -------
        Ellipsoid
            Class instance with fitted parameters.

        Raises
        ------
        NotImplementedError
            If the points are not 3D.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape[1] != 3 or len(positions.shape) != 2:
            raise NotImplementedError(
                "Only three-dimensional point clouds are supported."
            )

        def ellipsoid_loss(radii, data_points, center, orientations):
            transformed_points = np.dot(data_points - center, orientations)

            normalized_points = transformed_points / radii

            distances = np.sum(normalized_points**2, axis=1) - 1

            loss = np.sum(distances**2)
            return loss

        center = positions.mean(axis=0)
        positions_centered = positions - center

        cov_mat = np.cov(positions_centered, rowvar=False)
        evals, evecs = np.linalg.eigh(cov_mat)

        sort_indices = np.argsort(evals)[::-1]
        evals = evals[sort_indices]
        evecs = evecs[:, sort_indices]

        initial_radii = 2 * np.sqrt(evals)

        result = optimize.minimize(
            ellipsoid_loss,
            initial_radii,
            args=(positions, center, evecs),
            method="Nelder-Mead",
        )
        radii = result.x

        return cls(radii=radii, center=center, orientations=evecs)

    def sample(
        self,
        n_samples: int,
        radii: np.ndarray = None,
        center: np.ndarray = None,
        orientations: np.ndarray = None,
    ) -> np.ndarray:
        """
        Samples points from the surface of an ellisoid.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw
        radii : np.ndarray
            Radii of the ellipse along each axis
        center : np.ndarray
            Center of the ellipse along each axis
        orientations : np.ndarray
            Square orientation matrix

        Returns
        -------
        np.ndarray
            Sampled points.
        """
        radii = self.radii if radii is None else radii
        center = self.center if center is None else center
        orientations = self.orientations if orientations is None else orientations

        sp = np.linspace(0, 2.0 * np.pi, num=n_samples)
        nx = sp.shape[0]
        u = np.repeat(sp, nx)
        v = np.tile(sp, nx)

        x = np.sin(u) * np.cos(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(u)

        samples = np.vstack((x, y, z)).T
        samples = samples * radii
        samples = samples.dot(orientations.T)
        samples += center

        return samples


class Cylinder(Parametrization):
    """
    Parametrize a point cloud as cylinder.
    """

    def __init__(self, centers: np.ndarray, orientations: np.ndarray,
        radius: float, height : float):
        """
        Initialize the Cylinder parametrization.

        Parameters
        ----------
        centers : np.ndarray
            Center coordinates of the cylinder in X and Y.
        orientations : np.ndarray
            Square orientation matrix
        radius: float
            Radius of the cylinder.
        height : float
            Height of the cylinder.
        """
        self.centers = centers
        self.orientations = orientations
        self.radius = radius
        self.height = height


    @classmethod
    def fit(cls, positions: np.ndarray) -> "Cylinder":
        """
        Fit a 3D point cloud to a cylinder.

        Parameters
        ----------
        positions : np.ndarray
            Point coordinates with shape (n x 3)

        Returns
        -------
        Cylinder
            Class instance with fitted parameters.

        Raises
        ------
        ValueError
            If th number of initial parameters is not equal to five.
        NotImplementedError
            If the points are not 3D.
        """

        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape[1] != 3 or len(positions.shape) != 2:
            raise NotImplementedError(
                "Only three-dimensional point clouds are supported."
            )

        center = positions.mean(axis=0)
        positions_centered = positions - center

        cov_mat = np.cov(positions_centered, rowvar=False)
        evals, evecs = np.linalg.eigh(cov_mat)

        sort_indices = np.argsort(evals)[::-1]
        evals = evals[sort_indices]
        evecs = evecs[:, sort_indices]

        initial_radii = 2 * np.sqrt(evals)

        def cylinder_loss(radii, data_points, center, orientations):
            transformed_points = np.dot(data_points - center, orientations)

            normalized_points = transformed_points / radii

            distances = np.sum(normalized_points**2, axis=1) - 1

            loss = np.sum(distances**2)
            return loss

        result = optimize.minimize(
            cylinder_loss,
            np.max(initial_radii),
            args=(positions, center, evecs),
            method="Nelder-Mead",
        )

        rotated_points = positions_centered.dot(evecs)
        heights = rotated_points.max(axis = 0) - rotated_points.min(axis = 0)
        height = heights[np.argmax(np.abs(np.diff(heights))) + 1]

        return cls(radius=result.x, centers=center, orientations=evecs, height = height)

    def sample(
        self,
        n_samples: int,
        centers: np.ndarray = None,
        orientations: np.ndarray = None,
        radius: float = None,
        height : float = None,
    ) -> np.ndarray:
        """
        Sample points from the surface of a cylinder.

        Parameters
        ----------
        centers : np.ndarray
            Center coordinates of the cylinder in X and Y.
        orientations : np.ndarray
            Square orientation matrix
        radius: float
            Radius of the cylinder.
        height : float
            Height of the cylinder.

        Returns
        -------
        np.ndarray
            Array of sampled points from the cylinder surface.
        """
        centers = self.centers if centers is None else centers
        orientations = self.orientations if orientations is None else orientations
        radius = self.radius if radius is None else radius
        height = self.height if height is None else height

        theta = np.linspace(0, 2 * np.pi, n_samples)
        h = np.linspace(-height/2, height/2, n_samples)
        
        mesh = np.asarray(np.meshgrid(theta, h)).reshape(2, -1).T

        x = radius * np.cos(mesh[:,0])
        y = radius * np.sin(mesh[:,0])
        z = mesh[:,1]
        samples =  np.column_stack((x, y, z))

        samples = samples.dot(orientations.T)
        samples += centers

        return samples


PARAMETRIZATION_TYPE = {"sphere": Sphere, "ellipsoid": Ellipsoid, "cylinder": Cylinder}
