#
# ColabSeg - Interactive segmentation GUI
#
# Marc Siggel, January 2023

from .utilities import plane_fit, make_plot_array, R_2vect, create_sphere_points, lstsq_sphere_fitting
import os
import h5py
from .image_io import ImageIO
import napari
import numpy as np
from .new_gui_functions import ColabSegData
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class NapariFramework(object):
    """docstring for NapariFramework. Testing environment to build GUI from here."""

    def __init__(self):
        super(NapariInteraction, self).__init__()
        self.viewer = napari.Viewer()
        self.data_structure = []


        self.data_structure.append(np.random.randint(0, 100, size=(10, 3)))
        self.napari_index.append(0)
        self.data_structure.append(np.random.randint(0, 100, size=(10, 3)))
        self.napari_index.append(1)
        self.data_structure.append(np.random.randint(0, 100, size=(10, 3)))
        self.napari_index.append(2)


    def initialize_data_from_colabseg(self):
        """This loads the data from scratch"""
        # todo add napari_index bookkeeping -> store in list use len(current)
        for index, entry in enumerate(self.data_structure):
            self.viewer.add_points(entry, size=10, name="colabseg_cluster_{}".format(index), metadata={"colabseg": True, "cluster_index": index})
        return

    def update_from_colabseg_to_napari(self):
        """Update Napari layer from Colabseg"""
        napari_index_current = []
        for napari_index, layer in enumerate(self.viewer.layers):
            if "colabseg" in layer.metadata.keys():
                napari_index_current.append(napari_index)


        for index, cluster in enumerate(self.data_structure):
            if index not in napari_index_current:
                self.viewer.add_points(cluster, size=10, name="colabseg_cluster_{}".format(index), metadata={"colabseg": True, "cluster_index": index})
            if (cluster == self.viewer.layers[napari_index_current[index]]).all() == False:
                self.viewer.layers[napari_index_current[index]] = cluster
            else:
                continue


        delete_list = []
        for napari_index, layer in enumerate(self.viewer.layers):
            if "colabseg" in layer.metadata.keys():
                if layer.metadata["cluster_index"] >= len(self.data_structure):
                    delete_list.append(napari_index)
        del self.viewer.layers[delete_list] # TOOO this doesn't work. do this differrently

        return

    def update_from_napari_to_colabseg(self):
        """Update data structure in Colabseg from Napari"""
        for layer in self.viewer.layers:
            if "colabseg" in layer.metadata.keys():


        # need to check if some layer was removed.
        # remove if not check if diff exists to keep it in sync
        # tie to delete on cluster

        return

    def highlight_selected_clusters_update(self, index):
        """Highlight the selected cluster in the gui same as in the jupyter notebook version"""
        # TODO this needs ot be tied to the clickback
        for layer in self.viewer.layers:
            if "colabseg" in layer.metadata.keys():
                if layer.metadata["cluster_index"] == index:
                    layer.face_color[:,0] = 1.0
                    layer.face_color[:,1] = 0.0
                    layer.face_color[:,2] = 0.0
                    layer.face_color[:,3] = 1.0
                    layer.refresh()
                else:
                    layer.face_color[:,0] = 0.9
                    layer.face_color[:,1] = 0.9
                    layer.face_color[:,2] = 0.9
                    layer.face_color[:,3] = 0.6
                    layer.refresh()
            else:
                continue
        return

    def hide_all_clusters(self, index):
        """hide all clusters in napari except the selected one."""
        for layer in self.viewer.layers:
            if "colabseg" in layer.metadata.keys():
                if layer.metadata["cluster_index"] == index:
                    layer.visible = 1
                    layer.refresh()
                else:
                    layer.visible = 0
                    layer.refresh()
            else:
                layer.visible = 0
                layer.refresh()

        return


    def modify_random_cluster(self):
        """testing functionality: make some arbitrary modification to the code."""

        return
    ## TODO: maintaining stability across colabseg and napari check update functions
    # TODO the user needs to be asked if he wants to edit certain points or not.
