import sys

try:
    from setuptools import setup, find_packages
except:
    raise ImportError("setuptools is required. Install it with: pip install setuptools")


# if sys.version_info < (3.9, 0):
#    sys.exit('Python 3.9 is required! please install in your environment.')

VERSION = "0.0.1"
DESCRIPTION = "Colab Seg "
LONG_DESCRIPTION = (
    "Segmentation toolbox GUI for membranes from cryo-electron tomography images"
)

# Setting up
setup(
    name="colabseg",
    version="0.0.1",
    author="Marc Siggel",
    author_email="marc.siggel@embl-hamburg.de",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    python_requires=">=3.6, <=3.9",
    install_requires=[
        "py3Dmol>2.0",
        "numpy==1.22.3",
        "h5py",
        "scipy",
        "tqdm",
        "scipy",
        "matplotlib",
        "pandas",
        "open3d",
        "ipython",
        "jupyter",
        "future",
        "pyntcloud",
    ],
    keywords=["python", "first package"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
    ],
)
