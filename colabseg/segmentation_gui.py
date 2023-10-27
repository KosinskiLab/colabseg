#
# ColabSeg - Interactive segmentation GUI
#
# Marc Siggel, December 2021

import numpy as np
from ipywidgets import widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .new_gui_functions import ColabSegData
from .py3dmol_controls import seg_visualization

class JupyterFramework(object):
    """Docstring JupyterFramework for GUI"""

    def __init__(self):
        self.all_widgets = {}
        self.seg_visualization = seg_visualization()
        self.data_structure = ColabSegData()

    def gui_elements_loading(self):
        """Load Loading interface"""

        self.all_widgets["input_mrc"] = widgets.Text(
            value="test_file.mrc",
            placeholder="mrc or h5 file",
            description="Input Filename:",
            style={"description_width": "initial"},
            disabled=False,
        )

        self.all_widgets["load_mrc"] = widgets.Button(
            description="Load and Convert MRC file",
            disabled=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )
        self.all_widgets["load_mrc"].on_click(self.load_mrc_file)

        self.all_widgets["load_state_hdf"] = widgets.Button(
            description="Load state h5",
            disabled=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )
        self.all_widgets["load_state_hdf"].on_click(self.load_state_hdf)

        self.all_widgets["load_point_cloud"] = widgets.Button(
            description="Load point cloud txt",
            disabled=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )
        self.all_widgets["load_point_cloud"].on_click(self.load_from_point_cloud)

        self.all_widgets["load_stl_file"] = widgets.Button(
            description="Load stl points",
            disabled=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )
        self.all_widgets["load_stl_file"].on_click(self.load_from_stl_file)

        self.hbox_load = widgets.HBox(
            [
                self.all_widgets["input_mrc"],
                self.all_widgets["load_mrc"],
                self.all_widgets["load_point_cloud"],
                self.all_widgets["load_stl_file"],
                self.all_widgets["load_state_hdf"],
            ]
        )
        display(self.hbox_load)

    def gui_elements_cluster_analysis(self):
        """Load Loading interface"""
        self.all_widgets["cluster_sel"] = widgets.SelectMultiple(
            options=[x for x in range(0, len(self.data_structure.cluster_list_tv), 1)],
            rows=10,
            description="Clusters:",
            disabled=False,
        )
        self.all_widgets["cluster_sel"].observe(
            self.seg_visualization.highlight_clusters, names="value"
        )

        self.all_widgets["fit_sel"] = widgets.SelectMultiple(
            options=[
                x
                for x in range(
                    len(self.data_structure.cluster_list_tv),
                    len(self.data_structure.cluster_list_tv)
                    + len(self.data_structure.cluster_list_fits),
                    1,
                )
            ],
            rows=10,
            description="Fits:",
            disabled=False,
        )
        self.all_widgets["fit_sel"].observe(
            self.seg_visualization.highlight_fits, names="value"
        )

        self.all_widgets["rotate_flat"] = widgets.Button(description="Rotate Flat")
        self.all_widgets["rotate_flat"].on_click(self.rotate_flat)

        self.all_widgets["rotate_back"] = widgets.Button(description="Rotate Original")
        self.all_widgets["rotate_back"].on_click(self.rotate_back)

        self.all_widgets["output_filename"] = widgets.Text(
            value="output_test.mrc",
            placeholder="Type something",
            description="Output Filename:",
            style={"description_width": "initial"},
            disabled=False,
        )

        self.all_widgets["select_all_clusters"] = widgets.Checkbox(
            False,
            description="Save all",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="80px"),
        )
        self.all_widgets["save_as_integers"] = widgets.Checkbox(
            True,
            description="Binary",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="80px"),
        )

        self.all_widgets["save_clusters_mrc"] = widgets.Button(
            description="Save Selected MRC"
        )
        self.all_widgets["save_clusters_mrc"].on_click(self.save_clusters_mrc)

        self.all_widgets["save_clusters_txt"] = widgets.Button(
            description="Save Selected TXT"
        )
        self.all_widgets["save_clusters_txt"].on_click(self.save_clusters_txt)

        self.all_widgets["save_hdf"] = widgets.Button(description="Save State hdf5")
        self.all_widgets["save_hdf"].on_click(self.save_state_hdf)

        self.all_widgets["fit_rbf"] = widgets.Button(description="Fit RBF Fxn")
        self.all_widgets["fit_rbf"].on_click(self.fit_rbf)

        self.all_widgets["directionality_rbf"] = widgets.Dropdown(
            options=["xz", "yz", "xy"],
            value="xz",
            description="Membrane Dir.:",
            layout=widgets.Layout(width="150px"),
        )

        self.all_widgets["fit_sphere"] = widgets.Button(description="Fit Sphere")
        self.all_widgets["fit_sphere"].on_click(self.fit_sphere)

        self.all_widgets["fit_ellipsoid"] = widgets.Button(description="Fit Ellipsoid")
        self.all_widgets["fit_ellipsoid"].on_click(self.fit_ellipsoid)

        self.all_widgets["fit_cylinder"] = widgets.Button(description="Fit Cylinder")
        self.all_widgets["fit_cylinder"].on_click(self.fit_cylinder)

        self.all_widgets["crop_fit"] = widgets.Button(description="Crop fit around")
        self.all_widgets["crop_fit"].on_click(self.crop_fit)

        self.all_widgets["distance_tolerance"] = widgets.IntText(
            value=50,
            min=1,
            description="Dist Toleracne:",
            disabled=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="150px"),
        )

        self.all_widgets["delete_fit"] = widgets.Button(description="Delete fit")
        self.all_widgets["delete_fit"].on_click(self.delete_fit)

        self.all_widgets["trim_min"] = widgets.IntText(
            value=100, min=0, description="Trim Min:", disabled=False
        )
        self.all_widgets["trim_max"] = widgets.IntText(
            value=100, min=0, description="Trim Max:", disabled=False
        )
        self.all_widgets["trim_top_bottom"] = widgets.Button(
            description="Trim top bottom"
        )
        self.all_widgets["trim_top_bottom"].on_click(self.trim_top_bottom)

        self.all_widgets["merge_clusters"] = widgets.Button(
            description="Merge Clusters"
        )
        self.all_widgets["merge_clusters"].on_click(self.merge_clusters)

        self.all_widgets["delete_cluster"] = widgets.Button(
            description="Delete Clusters"
        )
        self.all_widgets["delete_cluster"].on_click(self.delete_cluster)

        self.all_widgets["undo_step"] = widgets.Button(description="Undo last step")
        self.all_widgets["undo_step"].on_click(self.undo_step)

        self.all_widgets["reset_input"] = widgets.Button(description="Rest to initial")
        self.all_widgets["reset_input"].on_click(self.reset_input)

        self.all_widgets["dbscan_eps"] = widgets.IntText(
            value=40,
            min=0,
            description="Neighbor Dist:",
            disabled=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="150px"),
        )
        self.all_widgets["dbscan_min_points"] = widgets.IntText(
            value=20,
            min=0,
            description="Min Points:",
            disabled=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="150px"),
        )

        self.all_widgets["dbscan_recluster"] = widgets.Button(
            description="DBSCAN clustering"
        )
        self.all_widgets["dbscan_recluster"].on_click(self.dbscan_recluster)

        self.all_widgets["outlier_removal"] = widgets.Button(
            description="Outlier removal"
        )
        self.all_widgets["outlier_removal"].on_click(self.outlier_removal)

        self.all_widgets["outlier_nb_neighbors"] = widgets.IntText(
            value=100,
            min=0,
            description="Num Neighbors:",
            disabled=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="150px"),
        )
        self.all_widgets["outlier_std_ratio"] = widgets.FloatText(
            value=0.2,
            min=0,
            description="Std Ratio:",
            disabled=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="150px"),
        )

        self.all_widgets["edge_outlier_removal"] = widgets.Button(
            description="Edge outlier removal"
        )
        self.all_widgets["edge_outlier_removal"].on_click(
            self.remove_eigenvalue_outliers
        )
        # # TODO: add hyperparameters for outlier removal

        self.all_widgets["merge_clusters"] = widgets.Button(
            description="Merge Clusters", style={"description_width": "initial"}
        )
        self.all_widgets["merge_clusters"].on_click(self.merge_clusters)

        self.all_widgets["load_raw_image_text"] = widgets.Text(
            value="raw_image.mrc",
            placeholder="Type something",
            description="Raw mrc file:",
            disabled=False,
        )
        self.all_widgets["load_raw_image_button"] = widgets.Button(
            description="Load Slice", style={"description_width": "initial"}
        )
        self.all_widgets["load_raw_image_button"].on_click(self.load_raw_image)
        self.all_widgets["load_raw_image"] = widgets.HBox(
            [
                self.all_widgets["load_raw_image_text"],
                self.all_widgets["load_raw_image_button"],
            ]
        )

        # add protein loading to GUI
        self.all_widgets["protein_filename"] = widgets.Text(
            value="protein_positions.txt",
            placeholder="Type something",
            description="Protein positions txt:",
            style={"description_width": "initial"},
            disabled=False,
        )
        self.all_widgets["load_proteins_button"] = widgets.Button(
            description="Load Protein File", style={"description_width": "initial"}
        )
        self.all_widgets["load_proteins_button"].on_click(
            self.execute_load_protein_position
        )

        self.all_widgets["delete_proteins_button"] = widgets.Button(
            description="Delete Protein", style={"description_width": "initial"}
        )
        self.all_widgets["delete_proteins_button"].on_click(
            self.execute_delete_protein_positions
        )

        # analyze minimal distance from selected membranes or fits
        self.all_widgets["analyze_minimal_distance_button"] = widgets.Button(
            description="Analyze Prot-Mem Distances",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )
        self.all_widgets["analyze_minimal_distance_button"].on_click(
            self.calculate_and_plot_distances_proteins
        )
        # analyze radii from selected fits
        # (should be sphere but will not stop you from doing it to non-spherical objects)
        self.all_widgets["analyze_sphere_radius_button"] = widgets.Button(
            description="Analyze Vesicle Radii",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )
        self.all_widgets["analyze_sphere_radius_button"].on_click(
            self.calculate_and_plot_sphere_radii
        )

        self.all_widgets["get_normals_button"] = widgets.Button(
            description="get normals",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )
        self.all_widgets["get_normals_button"].on_click(self.get_and_plot_normals)

        self.all_widgets["delete_normals_button"] = widgets.Button(
            description="delete normals",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )
        self.all_widgets["delete_normals_button"].on_click(self.delete_normals)

        self.all_widgets["flip_normals_button"] = widgets.Button(
            description="flip normals",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )
        self.all_widgets["flip_normals_button"].on_click(self.flip_normals)

        self.all_widgets["num_bins_button_distances"] = widgets.IntText(
            value=20,
            min=2,
            description="Num Bins Distance Plot:",
            disabled=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )
        self.all_widgets["num_bins_button_radii"] = widgets.IntText(
            value=20,
            min=2,
            description="Num Bins Radii Plot:",
            disabled=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )

        self.all_widgets["property_filename"] = widgets.Text(
            value="protein_positions.txt",
            placeholder="Type something",
            description="Output Filename",
            style={"description_width": "initial"},
            disabled=False,
        )

        self.all_widgets["save_radii_list_button"] = widgets.Button(
            description="Save Radii txt",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="150px"),
        )
        self.all_widgets["save_radii_list_button"].on_click(self.save_radii_file)

        self.all_widgets["save_radii_plot_button"] = widgets.Button(
            description="Save Radii Plot",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="150px"),
        )
        self.all_widgets["save_radii_plot_button"].on_click(self.save_radii_plot)

        self.all_widgets["protein_analyis_label"] = widgets.HTML(
            value="<b>Protein Analysis</b>"
        )
        self.all_widgets["radii_analysis_label"] = widgets.HTML(
            value="<b>Radii Analysis</b>"
        )
        self.all_widgets["normal_analyis_label"] = widgets.HTML(
            value="<b>Membrane Normal Analysis</b>"
        )

        self.all_widgets["save_distance_list_button"] = widgets.Button(
            description="Save Distance txt",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="150px"),
        )
        self.all_widgets["save_distance_list_button"].on_click(self.save_distance_file)

        self.all_widgets["save_distance_plot_button"] = widgets.Button(
            description="Save Distance Plot",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="150px"),
        )
        self.all_widgets["save_distance_plot_button"].on_click(self.save_distance_plot)

        self.all_widgets["save_normals_list_button"] = widgets.Button(
            description="Save Normals txt",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="150px"),
        )
        self.all_widgets["save_normals_list_button"].on_click(self.save_normals_file)

        self.all_widgets["save_analysis_line"] = widgets.HBox(
            [
                self.all_widgets["property_filename"],
                self.all_widgets["save_radii_list_button"],
                self.all_widgets["save_radii_plot_button"],
                self.all_widgets["save_distance_list_button"],
                self.all_widgets["save_distance_plot_button"],
                self.all_widgets["save_normals_list_button"],
            ]
        )
        self.all_widgets["save_tomogram_line"] = widgets.HBox(
            [
                self.all_widgets["select_all_clusters"],
                self.all_widgets["save_as_integers"],
                self.all_widgets["output_filename"],
                self.all_widgets["save_clusters_mrc"],
                self.all_widgets["save_clusters_txt"],
                self.all_widgets["save_hdf"],
            ]
        )

        if self.data_structure.raw_tomogram_slice == []:
            plot_output = widgets.Output()
            plot_output.clear_output()
            with plot_output:
                plt.figure(dpi=200)
            plot_output.clear_output()
            self.all_widgets["plot_output"] = plot_output

        if (
            self.data_structure.analysis_properties["minimal_distances"] == []
            and self.data_structure.analysis_properties["radii"] == []
        ):
            plot_output_analysis = widgets.Output()
            plot_output_analysis.clear_output()
            with plot_output_analysis:
                plt.figure(dpi=200)
            plot_output_analysis.clear_output()
            self.all_widgets["analysis_plot_output"] = plot_output_analysis

        self.all_widgets["raw_image_plt"] = self.all_widgets["plot_output"]
        self.all_widgets["analysis_plot_figure"] = self.all_widgets[
            "analysis_plot_output"
        ]
  

        self.all_widgets["control_stat_outlier_removal"] = widgets.HBox(
            [
                self.all_widgets["outlier_nb_neighbors"],
                self.all_widgets["outlier_std_ratio"],
                self.all_widgets["outlier_removal"],
            ]
        )
        self.all_widgets["control_dbscan_clustering"] = widgets.HBox(
            [
                self.all_widgets["dbscan_eps"],
                self.all_widgets["dbscan_min_points"],
                self.all_widgets["dbscan_recluster"],
            ]
        )
        self.all_widgets["protein_loading"] = widgets.HBox(
            [
                self.all_widgets["protein_filename"],
                self.all_widgets["load_proteins_button"],
                self.all_widgets["delete_proteins_button"],
            ]
        )
        self.all_widgets["normals_processing"] = widgets.HBox(
            [
                self.all_widgets["get_normals_button"],
                self.all_widgets["flip_normals_button"],
                self.all_widgets["delete_normals_button"],
            ]
        )

        self.all_widgets["saving"] = widgets.VBox(
            [
                self.all_widgets["save_tomogram_line"],
                self.all_widgets["save_analysis_line"],
            ]
        )
        self.all_widgets["fitting_rbf"] = widgets.HBox(
            [self.all_widgets["fit_rbf"], self.all_widgets["directionality_rbf"]]
        )
        self.all_widgets["cropping"] = widgets.HBox(
            [self.all_widgets["crop_fit"], self.all_widgets["distance_tolerance"]]
        )
        self.all_widgets["fit_closed_surface"] = widgets.HBox(
            [
                self.all_widgets["fit_sphere"],
                self.all_widgets["fit_ellipsoid"],
                self.all_widgets["fit_cylinder"]
            ]
        )
        self.all_widgets["fitting"] = widgets.VBox(
            [
                self.all_widgets["fitting_rbf"],
                self.all_widgets["fit_closed_surface"],
                self.all_widgets["cropping"],
                self.all_widgets["delete_fit"],
            ]
        )
        self.all_widgets["trimming"] = widgets.HBox(
            [
                self.all_widgets["trim_max"],
                self.all_widgets["trim_min"],
                self.all_widgets["trim_top_bottom"],
            ]
        )
        self.all_widgets["rotating"] = widgets.HBox(
            [self.all_widgets["rotate_flat"], self.all_widgets["rotate_back"]]
        )
        self.all_widgets["undoing"] = widgets.VBox(
            [
                self.all_widgets["undo_step"],
                self.all_widgets["reset_input"],
                self.all_widgets["delete_cluster"],
                self.all_widgets["merge_clusters"],
            ]
        )
        self.all_widgets["outliers"] = widgets.VBox(
            [
                self.all_widgets["edge_outlier_removal"],
                self.all_widgets["control_stat_outlier_removal"],
                self.all_widgets["control_dbscan_clustering"],
            ]
        )
        self.all_widgets["raw_image"] = widgets.VBox(
            [self.all_widgets["load_raw_image"], self.all_widgets["raw_image_plt"]]
        )

        self.all_widgets["analysis"] = widgets.VBox(
            [
                self.all_widgets["normal_analyis_label"],
                self.all_widgets["normals_processing"],
                self.all_widgets["protein_analyis_label"],
                self.all_widgets["protein_loading"],
                self.all_widgets["num_bins_button_distances"],
                self.all_widgets["analyze_minimal_distance_button"],
                self.all_widgets["radii_analysis_label"],
                self.all_widgets["num_bins_button_radii"],
                self.all_widgets["analyze_sphere_radius_button"],
                self.all_widgets["analysis_plot_figure"],
            ]
        )

        self.all_widgets["tab_nest"] = widgets.Tab()
        cluster_tab = widgets.HBox(
            [self.all_widgets["cluster_sel"], self.all_widgets["undoing"]]
        )
        edit_tab = widgets.VBox(
            [self.all_widgets["trimming"], self.all_widgets["rotating"]]
        )
        point_edit_tab = widgets.HBox([self.all_widgets["outliers"]])
        fit_tab = widgets.HBox(
            [self.all_widgets["fit_sel"], self.all_widgets["fitting"]]
        )
        save_tab = self.all_widgets["saving"]
        raw_image = self.all_widgets["raw_image"]
        analysis_tab = self.all_widgets["analysis"]

        self.all_widgets["tab_nest"].children = [
            cluster_tab,
            edit_tab,
            point_edit_tab,
            fit_tab,
            analysis_tab,
            save_tab,
            raw_image,
        ]

        titles = [
            "Cluster Selection",
            "Lamella Editing",
            "Points Editing",
            "Fits",
            "Analysis",
            "Save",
            "Raw Image",
        ]
        for i, title in enumerate(titles):
            self.all_widgets["tab_nest"].set_title(i, title)
        display(self.all_widgets["tab_nest"])

        # organize into tabs
        # 3D visualizatiton GUI here
        self.seg_visualization.load_all_models(
            self.data_structure.cluster_list_tv, start_index=0
        )
        if len(self.data_structure.cluster_list_fits) > 0:
            self.seg_visualization.load_all_models_fit(
                self.data_structure.cluster_list_fits,
                start_index=len(self.data_structure.cluster_list_tv),
            )
        self.seg_visualization.add_bounding_box(self.data_structure.boxlength)
        if len(self.data_structure.protein_positions_list) > 0:
            self.seg_visualization.load_protein_positions(
                self.data_structure.protein_positions_list, start_index=0
            )
        if len(self.data_structure.analysis_properties["normal_selection"]) > 0:
            self.seg_visualization.load_normal_positions(
                self.data_structure.analysis_properties["normal_selection"],
                self.data_structure.analysis_properties["surface_normals"],
            )

    def boot_gui(self):
        """Initial booting of the gui"""
        clear_output()
        # sets the downsampling value for the py3dmol module for large point clouds.
        self.update_downsampling()
        self.seg_visualization.view.removeAllModels()
        self.seg_visualization.view.removeAllShapes()
        self.gui_elements_cluster_analysis()
        self.seg_visualization.view.update()
        return

    def reload_gui(self):
        """Reloads the entire gui after applying changes"""
        clear_output()
        # sets the downsampling value for the py3dmol module for large point clouds.
        self.update_downsampling()
        self.seg_visualization.view.removeAllModels()
        self.seg_visualization.view.removeAllShapes()
        self.gui_elements_cluster_analysis()
        self.seg_visualization.view.update()
        self.seg_visualization.view.show()
        return

    def load_mrc_file(self, obj):
        """Load MRC file and populate backend"""
        # overwrite previous instance
        self.data_structure = ColabSegData()
        ## TODO: check if file is really mrc or mrc.gz extension

        self.data_structure.load_tomogram(self.all_widgets["input_mrc"].value)
        # self.data_structure.convert_tomo(step=self.all_widgets["step_size"].value)
        self.data_structure.convert_tomo()
        self.data_structure.get_lamina_rotation_matrix()
        return

    def load_from_point_cloud(self, obj):
        self.data_structure = ColabSegData()
        self.data_structure.load_point_cloud(self.all_widgets["input_mrc"].value)
        self.data_structure.get_lamina_rotation_matrix()
        return

    def load_from_stl_file(self, obj):
        self.data_structure = ColabSegData()
        self.data_structure.load_stl_file(self.all_widgets["input_mrc"].value)
        self.data_structure.get_lamina_rotation_matrix()
        return

    def load_viz(self, obj):
        """Load Visualizations"""
        self.seg_visualization.load_model_from_file(self.all_widgets["input_mrc"].value)
        return

    def rotate_flat(self, obj):
        """Rotate Lamina Flat onto xy plane"""
        self.data_structure.backup_step_to_previous()
        self.data_structure.plain_fit_and_rotate_lamina(backward=False)
        self.reload_gui()
        # self.seg_visualization.load_all_models(self.data_structure.cluster_list_tv)
        # self.seg_visualization.add_bounding_box(self.data_structure.boxlength)
        # self.seg_visualization.highlight_clusters
        return

    def rotate_back(self, obj):
        """rotate Lamina back to original position"""
        self.data_structure.backup_step_to_previous()
        self.data_structure.plain_fit_and_rotate_lamina(backward=True)
        # self.seg_visualization.load_all_models(self.data_structure.cluster_list_tv)
        # self.seg_visualization.add_bounding_box(self.data_structure.boxlength)
        # self.seg_visualization.highlight_clusters
        self.reload_gui()
        return

    def trim_top_bottom(self, obj):
        """Above and below a certain value"""
        if (
            len(self.all_widgets["cluster_sel"].value) == 0
            and len(self.all_widgets["fit_sel"].value) == 0
        ):
            return
        self.data_structure.backup_step_to_previous()
        trim_min = self.all_widgets["trim_min"].value
        trim_max = self.all_widgets["trim_max"].value

        if len(self.all_widgets["cluster_sel"].value) != 0:
            self.data_structure.trim_cluster_egdes_cluster(
                self.all_widgets["cluster_sel"].value, trim_min, trim_max
            )
        # TODO WARNING NEED TO SUBTRACT THE VALUE
        if len(self.all_widgets["fit_sel"].value) != 0:
            self.data_structure.trim_cluster_egdes_fit(
                self.fit_idx_conv()[0], trim_min, trim_max
            )
        # self.seg_visualization.load_all_models(self.data_structure.cluster_list_tv)
        # self.seg_visualization.add_bounding_box(self.data_structure.boxlength)
        self.reload_gui()
        return

    def load_state_hdf(self, obj):
        """load state as hdf5"""
        self.data_structure = ColabSegData()
        self.data_structure.load_hdf(self.all_widgets["input_mrc"].value)
        return

    def save_state_hdf(self, obj):
        """save state as hdf5 file"""
        self.data_structure.save_hdf(self.all_widgets["output_filename"].value)
        return

    def save_clusters_mrc(self, obj):
        """Saves an MRC file with the selected clusters and fits"""
        positions = []

        if self.all_widgets["select_all_clusters"].value == True:
            for cluster_tv in self.data_structure.cluster_list_tv:
                positions.append(cluster_tv)
            for cluster_fit in self.data_structure.cluster_list_fits:
                positions.append(cluster_fit)
            positions = np.vstack(np.asarray(positions))
            self.data_structure.write_output_mrc(
                positions, self.all_widgets["output_filename"].value
            )
            print("saving mrc as {}".format(self.all_widgets["output_filename"].value))

        if self.all_widgets["select_all_clusters"].value == False:
            for i in self.all_widgets["cluster_sel"].value:
                positions.append(self.data_structure.cluster_list_tv[i])
            for j in self.all_widgets["fit_sel"].value:
                fit_index = int(j - len(self.data_structure.cluster_list_tv))
                positions.append(self.data_structure.cluster_list_fits[fit_index])
            positions = np.vstack(np.asarray(positions))
            self.data_structure.write_output_mrc(
                positions, self.all_widgets["output_filename"].value
            )
            print("saving mrc as {}".format(self.all_widgets["output_filename"].value))

        return

    def save_clusters_txt(self, obj):
        """Saves a TXT file with the selected clusters and fits"""
        # merged or separate
        positions = []
        if self.all_widgets["select_all_clusters"].value == True:
            for cluster_tv in self.data_structure.cluster_list_tv:
                positions.append(cluster_tv)
            for cluster_fit in self.data_structure.cluster_list_fits:
                positions.append(cluster_fit)
            positions = np.vstack(np.asarray(positions))

        if self.all_widgets["select_all_clusters"].value == False:
            for i in self.all_widgets["cluster_sel"].value:
                positions.append(self.data_structure.cluster_list_tv[i])
            for j in self.all_widgets["fit_sel"].value:
                positions.append(self.data_structure.cluster_list_fit[j])
            positions = np.vstack(np.asarray(positions))

        self.data_structure.write_txt(
            positions, self.all_widgets["output_filename"].value
        )
        return

    def merge_clusters(self, obj):
        """Merge selected clusters"""
        if len(self.all_widgets["cluster_sel"].value) < 2:
            print("Not enough selected!")
            print("Please select at least 2 cluster for merging")
        self.data_structure.merge_clusters(self.all_widgets["cluster_sel"].value)
        self.reload_gui()
        return

    def reset_input(self, obj):
        """Reset input"""
        self.data_structure.backup_step_to_previous()
        self.data_structure.reload_original_values()
        self.reload_gui()
        return

    def undo_step(self, obj):
        """undo last step"""
        self.data_structure.reload_previous_step()
        self.reload_gui()
        return

    def fit_rbf(self, obj):
        """fit RBF to selected clusters"""
        self.data_structure.backup_step_to_previous()
        if len(self.all_widgets["cluster_sel"].value) != 1:
            print("Nothing or too many clusters selected!")
            print("Please select a single cluster for the fitting procedure!")
            return
        self.data_structure.interpolate_membrane_rbf(
            self.all_widgets["cluster_sel"].value,
            skip_to_downsample="auto",
            directionality=self.all_widgets["directionality_rbf"].value,
        )
        self.reload_gui()
        return

    def fit_sphere(self, obj):
        """Fit lstsq sphere to selected cluster"""
        self.data_structure.backup_step_to_previous()
        if len(self.all_widgets["cluster_sel"].value) != 1:
            print("Nothing or too many clusters selected!")
            print("Please select a single cluster for the fitting procedure!")
            return
        self.data_structure.interpolate_membrane_closed_surface(
            "sphere",
            self.all_widgets["cluster_sel"].value
        )
        self.reload_gui()
        return
    
    def fit_ellipsoid(self, obj):
            """Fit lstsq ellipsoid to selected cluster"""
            self.data_structure.backup_step_to_previous()
            if len(self.all_widgets["cluster_sel"].value) != 1:
                print("Nothing or too many clusters selected!")
                print("Please select a single cluster for the fitting procedure!")
                return
            self.data_structure.interpolate_membrane_closed_surface(
                "ellipsoid",
                self.all_widgets["cluster_sel"].value
            )
            self.reload_gui()
            return

    def fit_cylinder(self, obj):
        """Fit lstsq ellipsoid to selected cluster"""
        self.data_structure.backup_step_to_previous()
        if len(self.all_widgets["cluster_sel"].value) != 1:
            print("Nothing or too many clusters selected!")
            print("Please select a single cluster for the fitting procedure!")
            return
        self.data_structure.interpolate_membrane_closed_surface(
            "cylinder",
            self.all_widgets["cluster_sel"].value
        )
        self.reload_gui()
        return

    def delete_fit(self, obj):
        """deletes a selected fit"""
        self.data_structure.backup_step_to_previous()
        if len(self.all_widgets["fit_sel"].value) != 1:
            print("Nothing or too many clusters selected!")
            print("Please select a single cluster for the deleting procedure!")
            return
        self.data_structure.delete_fit(self.fit_idx_conv()[0])
        self.reload_gui()
        return

    def dbscan_recluster(self, obj):
        """run DBSCAN clustering and re-assign clusters"""
        self.data_structure.backup_step_to_previous()
        if len(self.all_widgets["cluster_sel"].value) != 1:
            print("Nothing or too many clusters selected!")
            print("Please select a single cluster for the DBSCAN clustering!")
            return
        self.data_structure.dbscan_clustering(
            self.all_widgets["cluster_sel"].value[0],
            eps=self.all_widgets["dbscan_eps"].value,
            min_points=self.all_widgets["dbscan_min_points"].value,
        )
        self.reload_gui()
        return

    def outlier_removal(self, obj):
        """remove statistical outliers"""
        # TODO: add selection parameters to interface
        self.data_structure.backup_step_to_previous()
        if len(self.all_widgets["cluster_sel"].value) != 1:
            print("Nothing or too many clusters selected!")
            print("Please select a single cluster for the outlier removal!")
            return
        self.data_structure.statistical_outlier_removal(
            self.all_widgets["cluster_sel"].value[0],
            nb_neighbors=self.all_widgets["outlier_nb_neighbors"].value,
            std_ratio=self.all_widgets["outlier_std_ratio"].value,
        )
        self.reload_gui()
        return

    def remove_eigenvalue_outliers(self, obj):
        """remove edge outliers"""
        # TODO: add lambda variable to the selection
        self.data_structure.backup_step_to_previous()
        if len(self.all_widgets["cluster_sel"].value) != 1:
            print("Nothing or too many clusters selected!")
            print("Please select a single cluster for the eigenvalue removal!")
            return
        self.data_structure.eigenvalue_outlier_removal(
            self.all_widgets["cluster_sel"].value[0]
        )
        self.reload_gui()
        return

    def crop_fit(self, obj):
        """Crop fit around fit"""
        self.data_structure.backup_step_to_previous()
        if len(self.all_widgets["cluster_sel"].value) != 1:
            print("Nothing or too many clusters selected!")
            print("Please select a single cluster for the cropping!")
            return
        if len(self.all_widgets["fit_sel"].value) != 1:
            print("Nothing or too many clusters selected!")
            print("Please select a single fit for the cropping!")
            return
        self.data_structure.crop_fit_around_membrane(
            cluster_index_tv=self.all_widgets["cluster_sel"].value[0],
            cluster_index_fit=self.fit_idx_conv()[0],
            distance_tolerance=self.all_widgets["distance_tolerance"].value,
        )
        self.reload_gui()
        return

    def delete_cluster(self, obj):
        self.data_structure.backup_step_to_previous()
        if len(self.all_widgets["cluster_sel"].value) == 0:
            print("Nothing selected!")
            print("Please select at least one cluster for deleting")
            return
        self.data_structure.delete_multiple_clusters(
            self.all_widgets["cluster_sel"].value
        )
        self.reload_gui()
        return

    def fit_idx_conv(self):
        """Helper function to convert the index of a fit"""
        fit_indices = np.asarray(self.all_widgets["fit_sel"].value) - len(
            self.data_structure.cluster_list_tv
        )
        return fit_indices

    def load_raw_image(self, obj):
        """Load a raw mrc slice"""
        self.data_structure.extract_slice(self.all_widgets["load_raw_image_text"].value)
        m = self.data_structure.raw_tomogram_slice
        vmin = m.min()
        vmax = m.max()
        self.all_widgets["plot_output"] = widgets.Output()
        with self.all_widgets["plot_output"]:
            plt.figure(dpi=100)
            plt.imshow(m, cmap=cm.Greys_r, vmin=vmin, vmax=vmax)
            plt.axis("off")
            plt.show()

        self.all_widgets["raw_image_plt"] = self.all_widgets["plot_output"]
        self.reload_gui()
        return

    def update_downsampling(self):
        """Downsample large point clouds.
        general viz variables.
        """
        num_points = len(self.data_structure.position_list)
        downsampling_value = int(np.round(num_points / 100000))
        if downsampling_value == 0:
            downsampling_value = 1
        if downsampling_value != self.seg_visualization.downsample:
            print("updating downsampling value to: {}".format(downsampling_value))
        self.seg_visualization.downsample = downsampling_value
        return

    def execute_load_protein_position(self, obj):
        """Load the position file to calculate distances"""
        try:
            self.data_structure.load_protein_position(
                self.all_widgets["protein_filename"].value
            )
        except:
            print(
                "File: {} not found!".format(self.all_widgets["protein_filename"].value)
            )
            return
        self.reload_gui()
        return

    def execute_delete_protein_positions(self, obj):
        self.data_structure.delete_protein_position()
        self.reload_gui()
        return

    def calculate_and_plot_distances_proteins(self, obj):
        """calculate all minimal distances between a protein and a sphere"""
        if (
            len(self.all_widgets["cluster_sel"].value) == 0
            and len(self.all_widgets["fit_sel"].value) == 0
        ):
            print("No cluster or fit selected")
            return
        if len(self.data_structure.protein_positions_list) == 0:
            print("No proteins loaded! load a protein to proceed")
            return
        fit_indices = self.fit_idx_conv()
        cluster_indices = self.all_widgets["cluster_sel"].value

        self.data_structure.analyze_protein_membrane_min_distance(
            cluster_indices=cluster_indices, fit_indices=fit_indices
        )
        self.all_widgets["analysis_plot_output"] = widgets.Output()
        with self.all_widgets["analysis_plot_output"]:
            plt.figure(dpi=100)
            plt.hist(
                self.data_structure.analysis_properties["minimal_distances"],
                bins=self.all_widgets["num_bins_button_distances"].value,
                density=False,
            )
            plt.ylabel("Count")
            plt.xlabel("Distance [$\AA$]")
            plt.title("Distribution of minimal protein-membrane distances")
            plt.show()

        self.all_widgets["analysis_plot_figure"] = self.all_widgets[
            "analysis_plot_output"
        ]
        self.reload_gui()
        return

    def calculate_and_plot_sphere_radii(self, obj):
        """Call calculation of all sphere radii"""
        if len(self.all_widgets["fit_sel"].value) == 0:
            print("No sphere fit selected. Only works on sphere fits...")
            return
        fit_indices = self.fit_idx_conv()
        self.data_structure.get_selected_sphere_radii(fit_indices=fit_indices)
        self.all_widgets["analysis_plot_output"] = widgets.Output()
        with self.all_widgets["analysis_plot_output"]:
            plt.figure(dpi=100)
            plt.hist(
                self.data_structure.analysis_properties["radii"],
                bins=self.all_widgets["num_bins_button_radii"].value,
                density=False,
            )
            plt.ylabel("Count")
            plt.xlabel("Radii [$\AA$]")
            plt.title("Distribution of radii")
            plt.show()

        self.all_widgets["analysis_plot_figure"] = self.all_widgets[
            "analysis_plot_output"
        ]
        self.reload_gui()
        return

    def save_radii_file(self, obj):
        if self.data_structure.analysis_properties["radii"] == []:
            print("No distances analyzed! Got analysis tab and analyze.")
        self.data_structure.save_values_txt(
            self.data_structure.analysis_properties["radii"],
            self.all_widgets["property_filename"].value,
        )
        return

    def save_distance_file(self, obj):
        if self.data_structure.analysis_properties["minimal_distances"] == []:
            print("No distances analyzed! Go to analysis tab and analyze.")
            return
        self.data_structure.save_values_txt(
            self.data_structure.analysis_properties["minimal_distances"],
            self.all_widgets["property_filename"].value,
        )
        return

    def save_normals_file(self, obj):
        if self.data_structure.analysis_properties["normal_selection"] == []:
            print("No normals computed!  Go to analysis tab and analyze.")
            return
        normal_position_vector_array = np.hstack(
            [
                self.data_structure.analysis_properties["normal_selection"],
                self.data_structure.analysis_properties["surface_normals"],
            ]
        )
        self.data_structure.save_values_txt(
            normal_position_vector_array, self.all_widgets["property_filename"].value
        )
        return

    def save_radii_plot(self, obj):
        if self.data_structure.analysis_properties["radii"] == []:
            print("No distances analyzed! Got analysis tab and analyze.")
            return
        plt.figure(dpi=100)
        plt.hist(
            self.data_structure.analysis_properties["radii"],
            bins=self.all_widgets["num_bins_button_radii"].value,
            density=False,
        )
        plt.ylabel("Count")
        plt.xlabel("Radii [$\AA$]")
        plt.title("Distribution of radii")
        plt.show()
        return

    def save_distance_plot(self, obj):
        if self.data_structure.analysis_properties["minimal_distances"] == []:
            print("No distances analyzed! Go to analysis tab and analyze.")
            return
        plt.figure(dpi=100)
        plt.hist(
            self.data_structure.analysis_properties["minimal_distances"],
            bins=self.all_widgets["num_bins_button_distances"].value,
            density=False,
        )
        plt.ylabel("Count")
        plt.xlabel("Distance [$\AA$]")
        plt.title("Distribution of minimal protein-membrane distances")
        plt.show()
        return

    def get_and_plot_normals(self, obj):
        if (
            len(self.all_widgets["cluster_sel"].value) == 0
            and len(self.all_widgets["fit_sel"].value) == 0
        ):
            print("No cluster or fit selected")
            return
        self.data_structure.calculate_normals(
            cluster_indices=self.all_widgets["cluster_sel"].value,
            fit_indices=self.all_widgets["fit_sel"].value,
        )
        self.reload_gui()
        return

    def delete_normals(self, obj):
        if len(self.data_structure.analysis_properties["surface_normals"]) == 0:
            print("no normals calculated. Use get normals first")
            return
        self.data_structure.delete_normals()
        self.reload_gui()
        return

    def flip_normals(self, obj):
        if len(self.data_structure.analysis_properties["surface_normals"]) == 0:
            print("no normals calculated. Use get normals first")
            return
        self.data_structure.flip_normals()
        self.reload_gui()
