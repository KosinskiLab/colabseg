from typing import List, Dict

import napari
import numpy as np

# from napari.types import ImageData
from napari.layers import Points
from magicgui import widgets, magicgui

# from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from .parametrization import Sphere, Ellipsoid, Cylinder


FOREGROUND_COLOR = "red"
BACKGROUND_COLOR = "blue"

spin_box = {"widget_type": "FloatSpinBox", "step": 1, "max": 1e10, "min": -1e10}

# @magicgui(
#     auto_call=True,
#     sigma={"widget_type": "FloatSlider", "max": 6},
#     mode={"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
#     layout="horizontal",
# )
# def gaussian_blur(layer: ImageData, sigma: float = 0, mode="nearest") -> ImageData:
#     # Apply a gaussian blur to 'layer'.
#     if sigma == 0:
#         return layer
#     if layer is not None:
#         return gaussian_filter(layer, sigma=sigma, mode=mode)


def generate_parametrization_widgets():
    @magicgui(
        auto_call=True,
        center_x=spin_box,
        center_y=spin_box,
        center_z=spin_box,
        radius=spin_box,
        layout="vertical",
    )
    def sphere(center_x: float, center_y: float, center_z: float, radius: float):
        parametrization = Sphere(
            center=np.array([center_z, center_y, center_x]), radius=np.array([radius])
        )
        return parametrization

    @magicgui(
        auto_call=True,
        center_x=spin_box,
        center_y=spin_box,
        center_z=spin_box,
        radius_z=spin_box,
        radius_y=spin_box,
        radius_x=spin_box,
        euler_x=spin_box,
        euler_y=spin_box,
        euler_z=spin_box,
        layout="vertical",
    )
    def ellipsoid(
        center_x: float,
        center_y: float,
        center_z: float,
        radius_x: float,
        radius_y: float,
        radius_z: float,
        euler_x: float,
        euler_y: float,
        euler_z: float,
    ):
        parametrization = Ellipsoid(
            center=np.array([center_z, center_y, center_x]),
            radii=np.array([radius_z, radius_y, radius_x]),
            orientations=[euler_z, euler_y, euler_x],
        )
        return parametrization

    @magicgui(
        auto_call=True,
        center_x=spin_box,
        center_y=spin_box,
        center_z=spin_box,
        radius=spin_box,
        height=spin_box,
        euler_x=spin_box,
        euler_y=spin_box,
        euler_z=spin_box,
        layout="vertical",
    )
    def cylinder(
        center_x: float,
        center_y: float,
        center_z: float,
        radius: float,
        height: float,
        euler_x: float,
        euler_y: float,
        euler_z: float,
    ):
        parametrization = Cylinder(
            center=np.array([center_z, center_y, center_x]),
            radius=radius,
            height=height,
            orientations=[euler_z, euler_y, euler_x],
        )
        return parametrization

    ret = {Sphere: sphere, Cylinder: cylinder, Ellipsoid: ellipsoid}
    return ret


class ColabSegNapariWidget(widgets.Container):
    """
    A widget for napari viewer to manage and visualize data from ColabSegData instances.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance to which this widget will be added.
    colabsegdata_instance : :py:class:`colabseg.new_gui_functions.ColabSegData`, optional
        An instance of ColabSegData containing the data to be visualized.

    Attributes
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    pixel_size : float
        Pixel size of the data, obtained from colabsegdata_instance.
    dropdown : magicgui.widgets.ComboBox
        Dropdown menu for selecting clusters to highlight.
    _previous_selection : str or None
        The previously selected cluster name.
    """

    def __init__(
        self,
        viewer,
        colabsegdata_instance: "ColabSegData",
        display_pixel_size: np.ndarray = None,
    ) -> "ColabSegNapariWidget":
        super().__init__(layout="vertical")

        self.viewer = viewer
        self.action_widgets = []

        if colabsegdata_instance is not None:
            pixel_size = colabsegdata_instance.pixel_size
            if display_pixel_size is not None:
                ratio = np.divide(display_pixel_size, pixel_size)
                pixel_size = np.multiply(pixel_size, ratio)
            self.pixel_size = pixel_size
            self.load_data(data=colabsegdata_instance)

        self._previous_selection = None
        self.dropdown = widgets.ComboBox(
            label="Highlight",
            choices=self._get_point_layers(),
            nullable=True,
        )

        self.fit_dropdown = widgets.ComboBox(
            label="Fit",
            choices=[
                i for i in range(len(colabsegdata_instance.cluster_list_fits_objects))
            ],
            nullable=True,
        )

        self.dropdown.changed.connect(self._on_cluster_selected)
        self.fit_dropdown.changed.connect(self._expose_parametrization_parameters)

        self.append(self.dropdown)
        self.append(self.fit_dropdown)

    def _get_point_layers(self) -> List[str]:
        """
        Returns a sorted list of point layer names from the napari viewer.

        Returns
        -------
        list
            A sorted list of strings of point layers present in the viewer.
        """
        return sorted(
            [layer.name for layer in self.viewer.layers if isinstance(layer, Points)]
        )

    def load_data(self, data: "ColabSegData") -> None:
        """
        Loads and visualizes data from a given ColabSegData instance. Points are
        divided by the pixel_size attribute of the ColabSegData instance. Currently,
        the attributes cluster_list_tv, cluster_list_fits, protein_positions_list
        from the ColabSegData instance are visualized as point clouds.

        Parameters
        ----------
        data : type
            An instance of ColabSegData containing the data to be visualized.
        """

        for index, cluster in enumerate(data.cluster_list_tv):
            if len(cluster) == 0:
                continue
            scaled_points = np.divide(cluster, self.pixel_size)
            self.viewer.add_points(
                scaled_points,
                name=f"Cluster_{index}",
                edge_width=0,
                size=1,
                symbol="disc",
                face_color=BACKGROUND_COLOR,
            )

        self._fit_points_sampled = [0] * len(data.cluster_list_fits)
        for index, cluster in enumerate(data.cluster_list_fits):
            if len(cluster) == 0:
                continue
            scaled_points = np.divide(cluster, self.pixel_size)
            self._fit_points_sampled[index] = scaled_points.shape[0]
            self.viewer.add_points(
                scaled_points,
                name=f"Fit_{index}",
                edge_width=0,
                size=1,
                symbol="ring",
                face_color=BACKGROUND_COLOR,
            )

        for index, cluster in enumerate(data.protein_positions_list):
            if len(cluster) == 0:
                continue
            scaled_points = np.divide(cluster, self.pixel_size)
            self.viewer.add_points(
                scaled_points,
                name=f"Protein_{index}",
                edge_width=0,
                size=3,
                symbol="star",
                face_color=BACKGROUND_COLOR,
            )
        self.parametrization_objects = data.cluster_list_fits_objects

    def _on_cluster_selected(self, event) -> None:
        """
        Callback function for dropdown selection changes. Highlights the selected cluster.

        Parameters
        ----------
        event
            The event triggered on changing the dropdown selection.
        """
        if self.dropdown.value is None:
            for point_cloud in self._get_point_layers():
                selected_layer = self.viewer.layers[point_cloud]
                selected_layer.face_color = BACKGROUND_COLOR
            return None

        selected_layer = self.viewer.layers[self.dropdown.value]
        selected_layer.face_color = FOREGROUND_COLOR

        if self._previous_selection is not None:
            previous_layer = self.viewer.layers[self._previous_selection]
            previous_layer.face_color = BACKGROUND_COLOR

        self._previous_selection = self.dropdown.value

    def export_data(self) -> Dict[str, List[np.ndarray]]:
        """
        Exports the point cloud data currently visualized in the napari viewer.
        The order of the output lists correspond to the input order. Deleted
        point clouds will be represented as empty lists.

        Returns
        -------
        dict of {str : list of numpy.ndarray}
            A dictionary where keys are point cloud class names and values
            are lists of numpy arrays representing point cloud coordinates.
        """
        point_clouds = self._get_point_layers()
        point_classes = {}
        for point_cloud in point_clouds:
            point_class, index = point_cloud.split("_")
            if point_class not in point_classes:
                point_classes[point_class] = 0
            point_classes[point_class] = max(point_classes[point_class], int(index))

        ret = {point_class: [[]] * (n + 1) for point_class, n in point_classes.items()}
        if "Fit" in point_classes:
            ret["Fits"] = [[]] * (point_classes["Fit"] + 1)

        for point_cloud in point_clouds:
            point_class, index = point_cloud.split("_")
            index = int(index)

            selected_layer = self.viewer.layers[point_cloud]
            original_points = np.multiply(selected_layer.data, self.pixel_size)

            ret[point_class][index] = original_points
            if point_class == "Fit":
                ret["Fit"][index] = self.parametrization_objects[index].sample(
                    self._fit_points_sampled[index]
                )

                ret["Fits"][index] = self.parametrization_objects[index]

        return ret

    def _expose_parametrization_parameters(self, event=None):
        for widget in self.action_widgets:
            self.remove(widget)
        self.action_widgets.clear()

        fit = self.fit_dropdown.value
        if fit is None:
            return None

        parametrization_object = self.parametrization_objects[fit]
        self._set_fit_defaults(parametrization_object)
        supported_parametrizations = generate_parametrization_widgets()

        if type(parametrization_object) not in supported_parametrizations.keys():
            raise ValueError(
                f"Parametrization type {parametrization_object} not supported."
                f" Supported are {','.join(list(supported_parametrizations.keys()))}"
            )

        function_widgets = supported_parametrizations[type(parametrization_object)]
        for widget in function_widgets:
            if widget.name in self.defaults:
                widget.value = self.defaults[widget.name]

        b1 = widgets.PushButton(value=True, text="Save")
        b2 = widgets.PushButton(value=True, text="Reset")

        b1.changed.connect(self._save_fit)
        b2.changed.connect(self._reset_fit)

        container = widgets.Container(widgets=[b1, b2], layout="horizontal")

        function_widgets.called.connect(self._plot_fit)

        self.action_widgets.append(function_widgets)
        self.action_widgets.append(container)
        self.append(function_widgets)
        self.append(container)

    def _set_fit_defaults(self, parametrization_object):
        self.defaults = {}
        if hasattr(parametrization_object, "center"):
            center = np.divide(parametrization_object.center, self.pixel_size)
            self.defaults["center_z"] = center[0]
            self.defaults["center_y"] = center[1]
            self.defaults["center_x"] = center[2]

        if hasattr(parametrization_object, "radius"):
            self.defaults["radius"] = np.divide(
                parametrization_object.radius, self.pixel_size[0]
            )

        if hasattr(parametrization_object, "radii"):
            radii = np.divide(parametrization_object.radii, self.pixel_size)
            self.defaults["radius_z"] = radii[0]
            self.defaults["radius_y"] = radii[1]
            self.defaults["radius_x"] = radii[2]

        if hasattr(parametrization_object, "orientations"):
            self.orientations_det = np.linalg.det(parametrization_object.orientations)
            euler_angles = Rotation.from_matrix(
                self.orientations_det * parametrization_object.orientations
            ).as_euler("zyx", degrees=True)

            self.defaults["euler_z"] = euler_angles[0]
            self.defaults["euler_y"] = euler_angles[1]
            self.defaults["euler_x"] = euler_angles[2]
        return None

    def _reset_fit(self, *args):
        for widget in self.action_widgets[0]:
            if widget.name in self.defaults:
                widget.value = self.defaults[widget.name]
        return None

    def _save_fit(self, *args):
        fit = self.fit_dropdown.value
        if fit is None:
            return None

        euler_angles = [None] * 3
        parametrization_object = self.parametrization_objects[fit]
        for widget in self.action_widgets[0]:
            if widget.name == "center_z":
                parametrization_object.center[0] = widget.value
            elif widget.name == "center_y":
                parametrization_object.center[1] = widget.value
            elif widget.name == "center_x":
                parametrization_object.center[2] = widget.value
            elif widget.name == "radius":
                parametrization_object.radius = widget.value
            elif widget.name == "radius_z":
                parametrization_object.radii[0] = widget.value
            elif widget.name == "radius_y":
                parametrization_object.radii[1] = widget.value
            elif widget.name == "radius_x":
                parametrization_object.radii[2] = widget.value
            elif widget.name == "euler_z":
                euler_angles[0] = widget.value
            elif widget.name == "euler_y":
                euler_angles[1] = widget.value
            elif widget.name == "radius_x":
                euler_angles[2] = widget.value

        if all([x is not None for x in euler_angles]):
            parametrization_object.orientations = (
                Rotation.from_euler(
                    angles=euler_angles, seq="zyx", degrees=True
                ).as_matrix()
                * self.orientations_det
            )

        if hasattr(parametrization_object, "center"):
            parametrization_object.center = np.multiply(
                parametrization_object.center, self.pixel_size
            )
        if hasattr(parametrization_object, "radii"):
            parametrization_object.radii = np.multiply(
                parametrization_object.radii, self.pixel_size
            )
        if hasattr(parametrization_object, "radius"):
            parametrization_object.radius = np.multiply(
                parametrization_object.radius, self.pixel_size[0]
            )

        self.parametrization_objects[fit] = parametrization_object
        self._set_fit_defaults(parametrization_object)

        return None

    def _plot_fit(self, *args):
        parametrization = args[0]

        if hasattr(parametrization, "orientations"):
            parametrization.orientations = (
                Rotation.from_euler(
                    angles=parametrization.orientations, seq="zyx", degrees=True
                ).as_matrix()
                * self.orientations_det
            )

        points = parametrization.sample(200)

        layer_name = f"Fit_{self.fit_dropdown.value}"
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = points
            return None

        self.viewer.add_points(
            points,
            face_color="red",
            name=f"Fit_{self.fit_dropdown.value}",
            edge_width=0,
            size=1,
            symbol="star",
        )
        # This is too slow for current napari versions
        # hull = ConvexHull(np.asarray(points))
        # surface = (hull.points, hull.simplices, np.ones(hull.points.shape[0]))
        # self.viewer.add_surface(
        #     surface,
        #     colormap="red",
        #     name=f"Fit_{self.fit_dropdown.value}",
        #     opacity=1,
        #     blending="translucent",
        #     shading="none",
        #     visible=True,
        # )


class NapariManager:
    """
    Manages the napari viewer and associated widgets for data visualization and interaction.

    Parameters
    ----------
    display_data : array-like, optional
        Data to be displayed in the napari viewer, typically image data.
    colabsegdata_instance : :py:class:`colabseg.new_gui_functions.ColabSegData`, optional
        An instance of ColabSegData for data visualization.

    Attributes
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    colabseg_widget : :py:class:`ColabSegNapariWidget`
        The widget for managing and visualizing ColabSegData instances.
    """

    def __init__(
        self,
        display_data: np.ndarray = None,
        colabsegdata_instance: "ColabSegData" = None,
        display_pixel_size: np.ndarray = None,
        **kwargs,
    ):
        self.viewer = napari.Viewer()

        if display_data is not None:
            self.viewer.add_image(
                data=display_data, name="Raw mrc file", colormap="gray"
            )

        self.colabseg_widget = ColabSegNapariWidget(
            viewer=self.viewer,
            colabsegdata_instance=colabsegdata_instance,
            display_pixel_size=display_pixel_size,
        )

        self.viewer.window.add_dock_widget(
            widget=self.colabseg_widget, name="Cluster", area="left"
        )

        # self.viewer.window.add_dock_widget(gaussian_blur, area = "bottom")
        # self.viewer.layers.events.changed.connect(gaussian_blur.reset_choices)

    def run(self):
        """
        Launches the napari viewer.
        """
        napari.run()

    def close(self):
        """
        Closes the napari viewer.
        """
        try:
            self.viewer.close()
            self.viewer.close_all()
        except Exception as e:
            print(e)
        self.viewer = None

    def export_data(self):
        """
        Exports the point cloud data from the colabseg_widget.

        Returns
        -------
        dict
            The point cloud data exported from the colabseg_widget.

        See Also
        --------
        :py:meth:`ColabSegNapariWidget.export_data`
        """
        return self.colabseg_widget.export_data()
