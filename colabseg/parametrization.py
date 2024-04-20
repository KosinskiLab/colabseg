from abc import ABC, abstractmethod

import numpy as np
import open3d as o3d
from scipy import optimize
from scipy.spatial import ConvexHull


class Parametrization(ABC):
    """
    A strategy class to represent parametrizations of point clouds
    """

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, positions: np.ndarray, *args, **kwargs) -> "Parametrization":
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
        Parametrization
            Parametrization instance.
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
        """

        center = positions.mean(axis=0)
        positions_centered = positions - center

        cov_mat = np.cov(positions_centered, rowvar=False)
        evals, evecs = np.linalg.eigh(cov_mat)

        sort_indices = np.argsort(evals)[::-1]
        evals = evals[sort_indices]
        evecs = evecs[:, sort_indices]

        initial_radii = 2 * np.sqrt(evals)

        def sphere_loss(params, data_points, orientations):
            radius, center = params[0], params[1:]
            transformed_points = np.dot(data_points - center, orientations)

            normalized_points = transformed_points / radius

            distances = np.sum(normalized_points**2, axis=1) - 1

            loss = np.sum(distances**2)
            return loss

        result = optimize.minimize(
            sphere_loss,
            np.array([np.max(initial_radii), *center]),
            args=(positions, evecs),
            method="Nelder-Mead",
        )
        radius, center = result.x[0], result.x[1:]

        return cls(radius=radius, center=center)

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

        indices = np.arange(0, n_samples, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_samples)
        theta = np.pi * (1 + 5**0.5) * indices

        positions_xyz = np.column_stack(
            [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
        )
        positions_xyz = np.multiply(positions_xyz, radius)
        positions_xyz = np.add(positions_xyz, center)

        return positions_xyz

    def compute_normal(self, points: np.ndarray) -> np.ndarray:
        """
        Compute the normal vector at a given point on the sphere surface.

        Parameters
        ----------
        points : np.ndarray
            Points on the sphere surface with shape n x d

        Returns
        -------
        np.ndarray
            Normal vectors at the given points
        """
        return (points - self.center) / self.radius

    def points_per_sampling(self, sampling_density: float) -> int:
        """
        Computes the apporximate number of random samples
        required to achieve a given sampling_density.

        Parameters
        ----------
        sampling_density : float
            Average distance between points.

        Returns
        -------
        int
            Number of required random samples.
        """
        n_points = np.multiply(
            np.square(np.pi),
            np.ceil(np.power(np.divide(self.radius, sampling_density), 2)),
        )
        return int(n_points)


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

        References
        ----------
        .. [1]  https://de.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape[1] != 3 or len(positions.shape) != 2:
            raise NotImplementedError(
                "Only three-dimensional point clouds are supported."
            )

        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        D = np.array(
            [
                x * x + y * y - 2 * z * z,
                x * x + z * z - 2 * y * y,
                2 * x * y,
                2 * x * z,
                2 * y * z,
                2 * x,
                2 * y,
                2 * z,
                1 - 0 * x,
            ]
        )
        d2 = np.array(x * x + y * y + z * z).T
        u = np.linalg.solve(D.dot(D.T), D.dot(d2))
        v = np.concatenate(
            [
                np.array([u[0] + 1 * u[1] - 1]),
                np.array([u[0] - 2 * u[1] - 1]),
                np.array([u[1] - 2 * u[0] - 1]),
                u[2:],
            ],
            axis=0,
        ).flatten()
        A = np.array(
            [
                [v[0], v[3], v[4], v[6]],
                [v[3], v[1], v[5], v[7]],
                [v[4], v[5], v[2], v[8]],
                [v[6], v[7], v[8], v[9]],
            ]
        )

        center = np.linalg.solve(-A[:3, :3], v[6:9])
        T = np.eye(4)
        T[3, :3] = center.T

        R = T.dot(A).dot(T.T)
        evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
        radii = np.sign(evals) * np.sqrt(1.0 / np.abs(evals))

        return cls(radii=radii, center=center, orientations=evecs)

    def sample(
        self,
        n_samples: int,
        radii: np.ndarray = None,
        center: np.ndarray = None,
        orientations: np.ndarray = None,
        sample_mesh: bool = False,
        mesh_init_factor: int = None,
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

        positions_xyz = np.zeros((n_samples, self.center.size))
        samples_drawn = 0
        np.random.seed(42)
        radii_fourth, r_min = np.power(radii, 4), np.min(radii)
        while samples_drawn < n_samples:
            point = np.random.normal(size=3)
            point /= np.linalg.norm(point)

            np.multiply(point, radii, out=point)

            p = r_min * np.sqrt(np.divide(np.square(point), radii_fourth).sum())
            u = np.random.uniform(0, 1)
            if u <= p:
                positions_xyz[samples_drawn] = point
                samples_drawn += 1

        positions_xyz = positions_xyz.dot(orientations.T)
        positions_xyz = np.add(positions_xyz, center)

        if sample_mesh:
            hull = ConvexHull(positions_xyz)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(positions_xyz[hull.vertices])
            mesh.triangles = o3d.utility.Vector3iVector(hull.simplices)

            if mesh_init_factor is None:
                point_cloud = mesh.sample_points_uniformly(
                    number_of_points=n_samples,
                )
            else:
                point_cloud = mesh.sample_points_poisson_disk(
                    number_of_points=n_samples,
                    init_factor=mesh_init_factor,
                )

            positions_xyz = np.asarray(point_cloud.points)

        return positions_xyz

        # n_points = int(np.ceil(int(np.sqrt(n_samples * 2))))

        # phi, theta = np.meshgrid(
        #     np.linspace(0, np.pi, n_points), np.linspace(0, 2*np.pi, n_points)
        # )
        # phi = phi.flatten()
        # theta = theta.flatten()

        # positions_xyz = np.column_stack([
        #     np.sin(phi) * np.cos(theta),
        #     np.sin(phi) * np.sin(theta),
        #     np.cos(phi)
        # ])
        # positions_xyz = np.multiply(positions_xyz, radii)
        # positions_xyz = positions_xyz.dot(orientations.T)
        # positions_xyz = np.add(positions_xyz, center)
        # return positions_xyz

    def compute_normal(self, points: np.ndarray) -> np.ndarray:
        """
        Compute the normal vector at a given point on the ellipsoid surface.

        Parameters
        ----------
        points : np.ndarray
            Points on the sphere surface with shape n x d

        Returns
        -------
        np.ndarray
            Normal vectors at the given points
        """
        # points_norm = (points - self.center) / self.radii

        norm_points = (points - self.center).dot(np.linalg.inv(self.orientations.T))
        normals = np.divide(np.multiply(norm_points, 2), np.square(self.radii))
        normals = np.dot(normals, self.orientations.T)
        normals /= np.linalg.norm(normals, axis=1)[:, None]

        return normals

    def points_per_sampling(self, sampling_density: float) -> int:
        """
        Computes the apporximate number of random samples
        required to achieve a given sampling_density.

        Parameters
        ----------
        sampling_density : float
            Average distance between points.

        Returns
        -------
        int
            Number of required random samples.
        """
        area_points = np.pi * np.square(sampling_density)
        area_ellipsoid = np.power(self.radii[0] * self.radii[1], 1.6075)
        area_ellipsoid += np.power(self.radii[0] * self.radii[2], 1.6075)
        area_ellipsoid += np.power(self.radii[1] * self.radii[2], 1.6075)

        area_ellipsoid = np.power(np.divide(area_ellipsoid, 3), 1 / 1.6075)
        area_ellipsoid *= 4 * np.pi

        n_points = np.ceil(np.divide(area_ellipsoid, area_points))
        return int(n_points)


class Cylinder(Parametrization):
    """
    Parametrize a point cloud as cylinder.
    """

    def __init__(
        self,
        centers: np.ndarray,
        orientations: np.ndarray,
        radius: float,
        height: float,
    ):
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

        def cylinder_loss(params, data_points, orientations):
            radii, center = params[0], params[1:]
            transformed_points = np.dot(data_points - center, orientations)

            normalized_points = transformed_points / radii

            distances = np.sum(normalized_points**2, axis=1) - 1

            loss = np.sum(distances**2)
            return loss

        result = optimize.minimize(
            cylinder_loss,
            np.array([np.max(initial_radii), *center]),
            args=(positions, evecs),
            method="Nelder-Mead",
        )
        radius, center = result.x[0], result.x[1:]
        rotated_points = positions_centered.dot(evecs)
        heights = rotated_points.max(axis=0) - rotated_points.min(axis=0)
        height = heights[np.argmax(np.abs(np.diff(heights))) + 1]
        return cls(radius=radius, centers=center, orientations=evecs, height=height)

    def sample(
        self,
        n_samples: int,
        centers: np.ndarray = None,
        orientations: np.ndarray = None,
        radius: float = None,
        height: float = None,
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

        n_samples = int(np.ceil(np.sqrt(n_samples)))
        theta = np.linspace(0, 2 * np.pi, n_samples)
        h = np.linspace(-height / 2, height / 2, n_samples)

        mesh = np.asarray(np.meshgrid(theta, h)).reshape(2, -1).T

        x = radius * np.cos(mesh[:, 0])
        y = radius * np.sin(mesh[:, 0])
        z = mesh[:, 1]
        samples = np.column_stack((x, y, z))

        samples = samples.dot(orientations.T)
        samples += centers

        return samples

    def points_per_sampling(self, sampling_density: float) -> int:
        """
        Computes the apporximate number of random samples
        required to achieve a given sampling_density.

        Parameters
        ----------
        sampling_density : float
            Average distance between points.

        Returns
        -------
        int
            Number of required random samples.
        """
        area_points = np.square(sampling_density)
        area = 2 * self.radius * (self.radius + self.height)

        n_points = np.ceil(np.divide(area, area_points))
        return int(n_points)


PARAMETRIZATION_TYPE = {"sphere": Sphere, "ellipsoid": Ellipsoid, "cylinder": Cylinder}
