#
# ColabSeg - Interactive segmentation GUI
#
# Marc Siggel, December 2021
# Adapted by Lorenz Lamm for MemBrain-seg, July 2023

import os
from ipywidgets import widgets
from ipywidgets import Layout

from membrain_seg.segmentation.segment import segment as membrain_segment
from membrain_seg.tomo_preprocessing.pixel_size_matching.match_pixel_size import match_pixel_size as membrain_match_pixel_size


def generate_membrainseg_gui():
    """Generates MemBrain-seg GUI"""
    all_values = {}

    def run_tv_code(obj):
        """This is a wrapper for MemBrain-seg"""
        print("MemBrain-seg running on {}".format(all_values["tomo_file"].value))
        if os.path.isfile(all_values["tomo_file"].value) is False:
            raise ValueError("Tomogram doesn't exist!")
        # if os.path.isfile(all_values["membrain_model"].value) is False:
        #     raise ValueError("MemBrain model doesn't exist!")

        if all_values["rescale_tomo"].value:
            print("Rescaling your tomogram. This can take several minutes.")
            px_scale_out = os.path.splitext(os.path.basename(all_values["tomo_file"].value))[0] + "_px_scaled.mrc"
            membrain_match_pixel_size(
                input_tomogram=all_values["tomo_file"].value,
                output_path=px_scale_out,
                pixel_size_in=(None if all_values["input_px"].value == "" else float(all_values["input_px"].value)),
                pixel_size_out=float(all_values["output_px"].value),
                disable_smooth=False
            )

        base_name = os.path.splitext(all_values["tomo_file"].value)[0]
        print("Segmenting your tomogram using MemBrain-seg.")
        out_file_seg = membrain_segment(
            tomogram_path=all_values["tomo_file"].value,
            ckpt_path=all_values["membrain_model"].value,
            out_folder=os.getcwd()
        )

        if all_values["remove_intermediates"].value:
            os.remove(px_scale_out)
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


    all_values["rescale_tomo"] = widgets.Checkbox(
    value=False,
    description='Rescale tomogram?',
    style = {'description_width': 'initial'},
    layout = widgets.Layout(width='200px'))


    all_values["input_px"] = widgets.Text(
    placeholder='default',
    description='pixel size of tomogram (if not specified, pixel size is read from header.)',
    style = {'description_width': 'initial'},
    layout = box_layout,
    disabled=False)

    all_values["output_px"] = widgets.Text(
    value="10.0",
    placeholder='10.0',
    description='target pixel size of tomogram. (i.e. to which pixel size should tomogram be scaled?)',
    style = {'description_width': 'initial'},
    layout = box_layout,
    disabled=False)

    all_values["remove_intermediates"] = widgets.Checkbox(
    value=False,
    description='Remove Intermediates (px matching)',
    style = {'description_width': 'initial'},
    layout = widgets.Layout(width='200px'))

    hbox_ckpt_path = all_values["membrain_model"]
    hbox_tomo_file =  all_values["tomo_file"]
    rescale_tomo = all_values["rescale_tomo"]
    input_px = all_values["input_px"]
    output_px = all_values["output_px"]
    remove_intermediates = all_values["remove_intermediates"]

    vbox_all = widgets.VBox([
        hbox_ckpt_path,
        hbox_tomo_file,
        rescale_tomo,
        remove_intermediates,
        input_px,
        output_px,
        ])
    #hbox = widgets.HBox([all_values["cpus"], all_values["sspace"]])
    display(vbox_all)

    start_run = widgets.Button(description="Run MemBrain-seg")
    start_run.on_click(run_tv_code)
    display(start_run)

