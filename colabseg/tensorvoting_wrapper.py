#
# ColabSeg - Interactive segmentation GUI
#
# Marc Siggel, December 2021

import subprocess
import numpy as np
import os
import ipywidgets
from ipywidgets import interact, fixed, IntSlider, widgets
from tqdm.notebook import tqdm
import multiprocessing


def generate_tensor_voting_gui():
    """Generates tensorvoting GUI"""
    all_values = {}

    def run_tv_code(obj):
        """This is a wrapper for tensorvoting"""
        print("TV running on {}".format(all_values["base_name"].value))
        # TODO check file exists

        if os.path.isfile(all_values["base_name"].value) is False:
            raise ValueError("File doesn't exist!")

        base_name = os.path.splitext(all_values["base_name"].value)[0]

        print("1/7 running scale_space")
        subprocess.run(
            [
                "{}/scale_space".format(all_values["tensor_voting_path"].value),
                "-t {}".format(all_values["cpus"].value),
                "-s {}".format(all_values["scale_space"].value),
                "{}.mrc".format(base_name),
                "{}_sspace.mrc".format(base_name),
            ]
        )
        print("2/7 running dtvoting 1")
        subprocess.run(
            [
                "{}/dtvoting".format(all_values["tensor_voting_path"].value),
                "-t {}".format(all_values["cpus"].value),
                "-s {}".format(all_values["tv1_value"].value),
                "{}_sspace.mrc".format(base_name),
                "{}_tv1.mrc".format(base_name),
            ]
        )
        print("3/7 running running surfaceness")
        subprocess.run(
            [
                "{}/surfaceness".format(all_values["tensor_voting_path"].value),
                "-t {}".format(all_values["cpus"].value),
                "-s {}".format(all_values["m_gaussian_pre"].value),
                "-p {}".format(all_values["m_gaussian_post"].value),
                "-m {}".format(all_values["m_thresh"].value),
                "{}_tv1.mrc".format(base_name),
                "{}_surf1.mrc".format(base_name),
            ]
        )
        print("4/7 running dtvoting 2")
        subprocess.run(
            [
                "{}/dtvoting".format(all_values["tensor_voting_path"].value),
                "-t {}".format(all_values["cpus"].value),
                "-w",
                "-s {}".format(all_values["tv2_value"].value),
                "{}_surf1.mrc".format(base_name),
                "{}_tv2.mrc".format(base_name),
            ]
        )
        print("5/7 running surfaceness saliency")
        subprocess.run(
            [
                "{}/surfaceness".format(all_values["tensor_voting_path"].value),
                "-t {}".format(all_values["cpus"].value),
                "-S",
                "-s {}".format(all_values["s_gaussian_pre"].value),
                "-p {}".format(all_values["s_gaussian_post"].value),
                "{}_tv2.mrc".format(base_name),
                "{}_surf2.mrc".format(base_name),
            ]
        )
        print("6/7 running running threshholding")
        subprocess.run(
            [
                "{}/thresholding".format(all_values["tensor_voting_path"].value),
                "-t {}".format(all_values["cpus"].value),
                "-l {}".format(all_values["thresholding_threshhold"].value),
                "{}_surf2.mrc".format(base_name),
                "{}_thresh.mrc".format(base_name),
            ]
        )
        print("7/7 running global_analysis clustering")
        subprocess.run(
            [
                "{}/global_analysis".format(all_values["tensor_voting_path"].value),
                "-v 2",
                "-3 {}".format(all_values["cluster_cutoff"].value),
                "{}_thresh.mrc".format(base_name),
                "{}_global2.mrc".format(base_name),
            ]
        )
        # subprocess.run(["/Users/masiggel/embl/tensorvoting/TomoSegMemTV_Apr2020_osx/bin/global_analysis", "-v 2", "-3 10", "{}_thresh.mrc".format(base_name), "{}_global2.mrc".format(base_name)])
        subprocess.run(["gzip", "{}_global2.mrc".format(base_name)])
        print("DONE!")
        print(
            "Your file is written to this working directory {} under name {}".format(
                os.getcwd(), "{}_global2.mrc.gz".format(base_name)
            )
        )
        print("### IMPORTANT: PLEASE CITE###")
        print(
            "Martinez-Sanchez, A.; Garcia, I.; Asano, S.; Lucic, V.; Fernandez, J.-J."
        )
        print(
            "Robust Membrane Detection Based on Tensor Voting for Electron Tomography"
        )
        print(
            "J. Struct. Biol. 2014, 186 (1), 49–61. https://doi.org/10.1016/j.jsb.2014.02.015"
        )

        if all_values["remove_intermediates"].value == True:
            print("cleaning intermediate files!")
            os.remove("{}_sspace.mrc".format(base_name))
            os.remove("{}_tv1.mrc".format(base_name))
            os.remove("{}_surf1.mrc".format(base_name))
            os.remove("{}_tv2.mrc".format(base_name))
            os.remove("{}_surf2.mrc".format(base_name))
            os.remove("{}_thresh.mrc".format(base_name))
            os.remove("{}_global2.mrc".format(base_name))

    def reset_optimal_default_values(obj):
        """Reset the optimal values according to my work"""
        all_values["scale_space"].value = 2
        all_values["tv1_value"].value = 20
        all_values["m_gaussian_pre"].value = 1.0
        all_values["m_gaussian_post"].value = 0.5
        all_values["m_thresh"].value = 0.1
        all_values["tv2_value"].value = 15
        all_values["s_gaussian_pre"].value = 1.0
        all_values["s_gaussian_post"].value = 0.0
        all_values["thresholding_threshhold"].value = 0.04
        all_values["cluster_cutoff"].value = 10000

    def print_current(obj):
        for key in all_values:
            print(all_values[key].value)

    all_values["tensor_voting_path"] = widgets.Text(
        placeholder="PATH/TO/TOMOSEG",
        description="TV Path:",
        style={"description_width": "initial"},
        disabled=False,
    )

    all_values["base_name"] = widgets.Text(
        value="input_filename.mrc",
        placeholder="Type something",
        description="Input Filename:",
        style={"description_width": "initial"},
        disabled=False,
    )

    num_cpus = multiprocessing.cpu_count()
    # print(num_cpus)
    all_values["cpus"] = widgets.IntSlider(
        value=num_cpus - 1,
        min=1,
        max=num_cpus,
        step=1,
        description="CPUs:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        style={"description_width": "initial"},
        readout_format="d",
    )

    all_values["scale_space"] = widgets.FloatSlider(
        value=2.0,
        min=0,
        max=3,
        step=0.1,
        description="sspace -s:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        style={"description_width": "initial"},
        readout_format=".1f",
    )

    all_values["tv1_value"] = widgets.IntSlider(
        value=20,
        min=1,
        max=30,
        step=1,
        description="TV1 slider -s:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        style={"description_width": "initial"},
        readout_format="d",
    )

    all_values["m_gaussian_pre"] = widgets.FloatSlider(
        value=1.0,
        min=0,
        max=3,
        step=0.1,
        description="Gaussian input -s:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        style={"description_width": "initial"},
        readout_format=".1f",
    )

    all_values["m_gaussian_post"] = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=3,
        step=0.1,
        description="Gaussian post -p:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        style={"description_width": "initial"},
        readout_format=".1f",
    )

    all_values["m_thresh"] = widgets.FloatSlider(
        value=0.01,
        min=0,
        max=3,
        step=0.01,
        description="Membrane Thresh:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        style={"description_width": "initial"},
        readout_format=".2f",
    )

    all_values["tv2_value"] = widgets.IntSlider(
        value=15,
        min=1,
        max=30,
        step=1,
        description="TV2 slider:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        style={"description_width": "initial"},
        readout_format="d",
    )

    all_values["s_gaussian_pre"] = widgets.FloatSlider(
        value=1.0,
        min=0,
        max=3,
        step=0.1,
        description="Gaussian pre -s:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        style={"description_width": "initial"},
        readout_format=".1f",
    )

    all_values["s_gaussian_post"] = widgets.FloatSlider(
        value=0.0,
        min=0,
        max=3,
        step=0.1,
        description="Gaussian post -p:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        style={"description_width": "initial"},
        readout_format=".1f",
    )

    all_values["thresholding_threshhold"] = widgets.FloatSlider(
        value=0.04,
        min=0,
        max=1,
        step=0.01,
        description="Thresh -l:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        style={"description_width": "initial"},
        readout_format=".1f",
    )

    all_values["cluster_cutoff"] = widgets.IntSlider(
        value=10000,
        min=1,
        max=50000,
        step=1,
        description="TV slider",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        style={"description_width": "initial"},
        readout_format=".1f",
    )

    all_values["remove_intermediates"] = widgets.Checkbox(
        value=False,
        description="Remove Intermediates",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="120px"),
    )

    hbox_tv_path = all_values["tensor_voting_path"]
    hbox_base_name = all_values["base_name"]
    hbox_cpus = all_values["cpus"]
    hbox_ssspace = all_values["scale_space"]
    hbox_tv1 = all_values["tv1_value"]
    hbox_surface1 = widgets.HBox(
        [
            all_values["m_gaussian_pre"],
            all_values["m_gaussian_post"],
            all_values["m_thresh"],
        ]
    )
    hbox_tv2 = all_values["tv2_value"]
    hbox_surface2 = widgets.HBox(
        [all_values["s_gaussian_pre"], all_values["s_gaussian_post"]]
    )
    hbox_thresh = all_values["thresholding_threshhold"]
    hbox_cluster_cutoff = all_values["cluster_cutoff"]
    hbox_remove_intermediates = all_values["remove_intermediates"]

    vbox_all = widgets.VBox(
        [
            hbox_tv_path,
            hbox_base_name,
            hbox_cpus,
            hbox_ssspace,
            hbox_tv1,
            hbox_surface1,
            hbox_tv2,
            hbox_surface2,
            hbox_thresh,
            hbox_cluster_cutoff,
            hbox_remove_intermediates,
        ]
    )
    # hbox = widgets.HBox([all_values["cpus"], all_values["sspace"]])
    display(vbox_all)

    start_run = widgets.Button(description="Run TV Pipeline")
    start_run.on_click(run_tv_code)
    display(start_run)

    set_defaults = widgets.Button(description="Set optmized default")
    set_defaults.on_click(reset_optimal_default_values)
    display(set_defaults)
