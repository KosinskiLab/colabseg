from typing import List, Dict

import numpy as np
import napari
from napari.layers import Points
from magicgui import widgets


FOREGROUND_COLOR = "red"
BACKGROUND_COLOR = "blue"


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
        self, viewer, colabsegdata_instance: "ColabSegData"
    ) -> "ColabSegNapariWidget":
        super().__init__(layout="vertical")

        self.viewer = viewer

        if colabsegdata_instance is not None:
            self.pixel_size = colabsegdata_instance.pixel_size
            self.load_data(data=colabsegdata_instance)

        self._previous_selection = None
        self.dropdown = widgets.ComboBox(
            label="Highlight",
            choices=self._get_point_layers(),
            nullable=True,
        )

        self.dropdown.changed.connect(self._on_cluster_selected)

        self.append(self.dropdown)

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
            scaled_points = np.divide(cluster, self.pixel_size)
            self.viewer.add_points(
                scaled_points,
                name=f"Cluster_{index}",
                edge_width=0,
                size=1,
                symbol="disc",
                face_color=BACKGROUND_COLOR,
            )

        for index, cluster in enumerate(data.cluster_list_fits):
            scaled_points = np.divide(cluster, self.pixel_size)
            self.viewer.add_points(
                scaled_points,
                name=f"Fit_{index}",
                edge_width=0,
                size=3,
                symbol="star",
                face_color=BACKGROUND_COLOR,
            )

        for index, cluster in enumerate(data.protein_positions_list):
            scaled_points = np.divide(cluster, self.pixel_size)
            self.viewer.add_points(
                scaled_points,
                name=f"Protein_{index}",
                edge_width=0,
                size=3,
                symbol="ring",
                face_color=BACKGROUND_COLOR,
            )

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
        for point_cloud in point_clouds:
            point_class, index = point_cloud.split("_")

            selected_layer = self.viewer.layers[point_cloud]
            original_points = np.multiply(selected_layer.data, self.pixel_size)

            ret[point_class][int(index)] = original_points
        print(ret)
        return ret


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
        **kwargs,
    ):
        self.viewer = napari.Viewer()

        if display_data is not None:
            self.viewer.add_image(
                data=display_data, name="Raw mrc file", colormap="gray_r"
            )

        self.colabseg_widget = ColabSegNapariWidget(
            viewer=self.viewer, colabsegdata_instance=colabsegdata_instance
        )

        self.viewer.window.add_dock_widget(
            widget=self.colabseg_widget, name="Cluster", area="left"
        )

    def run(self):
        """
        Launches the napari viewer.
        """
        napari.run()

    def close(self):
        """
        Closes the napari viewer.
        """
        self.viewer.close()

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
