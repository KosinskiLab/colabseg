# Colabseg Tool: Jupyter-based Membrane Segmentation Manipulation
---
![Build and Testing](https://github.com/KosinskiLab/colabseg/actions/workflows/python-app.yml/badge.svg) [![python3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-360/) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## General Remarks

**WARNING: This installation guide is only valid for MacOS or linux**

---
## Preliminary Installation and use on Linux machines
Make an environment with anaconda and open it:

```
conda create --name YOUR_ENV_NAME python==3.8 pip
source activate YOUR_ENV_NAME
```
Run pip in the folder where `setup.py` is located:
```
pip install .
```

This also installs all necessary dependencies.


## Installation MacOS:

Make an environment with anaconda and open it:

```
conda create --name YOUR_ENV_NAME python==3.8 pip
source activate YOUR_ENV_NAME
```
Run pip in the folder where `setup.py` is located:
```
pip install .
```

This also installs all necessary dependencies.

Then add this environment as jupyter kernel and then boot a new jupyter notebook or better the demo notebook `colabseg_demo_notebook.ipynb` in the colabseg folder:

```
python -m ipykernel install --user --name=YOUR_ENV_NAME
jupyter notebook colabseg_demo_notebook.ipynb
```
Make sure to pick the correct environment as kernel to have access to the installed software. Execute the cells in order. It is possible to skip the tensorvoting step if segmented data is already available. Then simply load the `.mrc` file and load. Alternatively, you can load a `.h5` file which is a specific state file of the software which contains all the metadata of the classes.

When loading a new file it is advised to either restart the kernel and start from the top of the notebook. Or at least re-run the file loading cell. This will purge any existing data and avoid potential issues in the experimental stage.

## Detailed User Guide:
Detailed tutorial and user guide can be found [here](https://kosinskilab.github.io/colabseg/).
