import threading
import numpy as np
import napari
from napari.layers import Image, Points
from napari.utils.events import EventedList
from magicgui import widgets


FOREGROUND_COLOR = "red"
BACKGROUND_COLOR = "blue"


class ColabSegNapariWidget(widgets.Container):
    def __init__(self, viewer, colabsegdata_instance: type) -> "ColabSegNapariWidget":
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

    def _get_point_layers(self) -> None:
        return sorted(
            [layer.name for layer in self.viewer.layers if isinstance(layer, Points)]
        )

    def load_data(self, data: type) -> None:
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

    def export_data(self) -> {str : [np.ndarray]}:
        ret = {}

        point_clouds = self._get_point_layers()
        point_classes = list(set([x.split("_")[0] for x in point_clouds]))

        ret = {point_class : [ [ ] ] * len(point_clouds) for point_class in point_classes}
        for point_cloud in point_clouds:
            point_class, index = point_cloud.split("_")

            selected_layer = self.viewer.layers[point_cloud]
            original_points = np.multiply(selected_layer.data, self.pixel_size)

            ret[point_class][int(index)] = original_points

        return ret


class NapariManager:
    def __init__(self, display_data=None, colabsegdata_instance=None, **kwargs):
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
        napari.run()

    def close(self):
        self.viewer.close()

    def export_data(self):
        return self.colabseg_widget.export_data()
