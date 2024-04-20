#
# ColabSeg - Interactive segmentation GUI
#
# Marc Siggel, December 2021
# Adapted by Lorenz Lamm for MemBrain-seg, July 2023

import os
from ipywidgets import widgets
from ipywidgets import Layout

from membrain_seg.segmentation.segment import segment as membrain_segment


def generate_membrainseg_gui():
    """Generates MemBrain-seg GUI"""
    all_values = {}

    def run_membrain_code(obj):
        """This is a wrapper for MemBrain-seg"""
        print("MemBrain-seg running on {}".format(all_values["tomo_file"].value))
        if os.path.isfile(all_values["tomo_file"].value) is False:
            raise ValueError("Tomogram doesn't exist!")

        print("Segmenting your tomogram using MemBrain-seg.")
        out_file_seg = membrain_segment(
            tomogram_path=all_values["tomo_file"].value,
            ckpt_path=all_values["membrain_model"].value,
            out_folder=os.getcwd(),
            store_connected_components=all_values["compute_connected_components"].value,
            connected_component_thres=(None if all_values["connected_component_thres"].value == "" else float(all_values["connected_component_thres"].value)),
            test_time_augmentation=all_values["test_time_augmentation"].value
        )

        print("Your file is written to", out_file_seg)


    box_layout = Layout(display='flex',
                    flex_flow='row',
                    align_items='stretch',
                    width='800px') 

    all_values["membrain_model"] = widgets.Text(
    placeholder='PATH/TO/MEMBRAIN_MODEL',
    description='MemBrain-seg model:',
    style = {'description_width': 'initial'},
    layout = box_layout,
    disabled=False)

    all_values["tomo_file"] = widgets.Text(
    placeholder='input_tomo.mrc',
    description='Input Filename:',
    style = {'description_width': 'initial'},
    layout = box_layout,
    disabled=False)

    all_values["compute_connected_components"] = widgets.Checkbox(
    value=False,
    description='Should connected components be computed to separate membranes?',
    style = {'description_width': 'initial'},
    layout = widgets.Layout(width='800px'))

    all_values["connected_component_thres"] = widgets.Text(
    placeholder='0',
    description='Connected components smaller than this will be removed.',
    style = {'description_width': 'initial'},
    layout = box_layout,
    disabled=False)

    all_values["test_time_augmentation"] = widgets.Checkbox(
    value=False,
    description='Should 8-fold test-time augmentation be performed (takes longer)?',
    style = {'description_width': 'initial'},
    layout = widgets.Layout(width='800px'))

    hbox_ckpt_path = all_values["membrain_model"]
    hbox_tomo_file =  all_values["tomo_file"]
    conn_comp_pred =  all_values["compute_connected_components"]
    conn_comp_thres =  all_values["connected_component_thres"]
    test_time_aug =  all_values["test_time_augmentation"]

    vbox_all = widgets.VBox([
        hbox_ckpt_path,
        hbox_tomo_file,
        conn_comp_pred,
        conn_comp_thres,
        test_time_aug
        ])
    #hbox = widgets.HBox([all_values["cpus"], all_values["sspace"]])
    display(vbox_all)

    start_run = widgets.Button(description="Run MemBrain-seg")
    start_run.on_click(run_membrain_code)
    display(start_run)
