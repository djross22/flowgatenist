# FlowGateNIST

FlowGateNIST is a Python package for automated flow cytometry data analysis designed for use with small cells like bacteria and yeast. 

Automated flow cytometry data analysis with FlowGateNIST has four steps: 
1. Flow Cytometry Standard (FCS) data import. FlowGateNIST starts by converting data from the FCS format to a Pandas DataFrame for easy manipulation in Python. 
2. Automated cell gating. FlowGateNIST uses a GMM approach and a comparison between measured cell samples and buffer blank samples for automated gating to discriminate between events that are most likely to be cells vs. events that are most likely to be background. 
3. Automated singlet gating. FlowGateNIST then uses comparisons between the height, area, and width parameters of flow cytometry events to automatically discriminate between singlet and multiplet events. 
4. Calibration of signals with fluorescent beads. In addition to automated gating, FlowGateNIST uses a multi-dimensional GMM applied to data for fluorescence calibration beads to convert measured fluorescence signals to comparable units. 

## Cite FlowGateNIST
If you use FlowGateNIST for a scientific publication, please cite the following paper:

Automated Analysis of Bacterial Flow Cytometry Data with FlowGateNIST, David Ross, XXXXX NN, pp. XXX-YYY, 2021.

Bibtex entry:
```
@article{flowgatenist,
 title={Automated Analysis of Bacterial Flow Cytometry Data with FlowGateNIST},
 author={Ross, D.},
 journal={XXXX},
 volume={NN},
 pages={XXX--YYY},
 year={2021}
}
```

## Getting Started
### Install Python
We recommend installing Python with the latest Annaconda distribution from: https://www.anaconda.com/products/individual.

### Download and install FlowGateNIST
Download the FlowGateNIST package from GitHub ([link](https://github.com/djross22/flowgatenist)) and save the source code in a local directory on your computer (e.g., on a Windows PC, C:\Users\username\Documents\Python Scripts\flowgatenist).

Open a Command Prompt (Windows) or the Terminal applicaiton (Mac) and navigate to the directory where you saved the FlowcCalNIST source code. Then install the FlowGateNIST package in editable mode using the command, "pip install -e ."
```
C:\Users\username\Documents\Python Scripts\flowgatenist>pip install -e .
```
Note that the "." at the end is important, as it indicates that the package should be installed from the current directory.

#### Edit the configuration files
The directory "\flowgatenist\flowgatenist\Config Files" is tracked by the repository. So, to use custom configuration files, make a local copy of the directory named: "\flowgatenist\flowgatenist\Local Config Files"

Then edit the files in the Local Config Directory:

The "top_directory.txt" file should contain the name of the top-level directory which will contain the data and the analysis memory for FlowGateNIST.<br>
We recommend using FlowGateNIST with a directory structure similarly to this:<br>
<img src="./example data/images/Figure_1.jpg" width=250 >

FlowGateNIST uses past results to initialize some of the analysis steps (e.g. Gaussian mixture model fits). These past results are automatically saved in an analysis memory directory that will created as a sub-directory within the top-level directory.<br>
The FlowGateNIST algorithms find the top-level directory (and the analysis memory) by moving upward in the directory tree from the directory containing the flow cytometry data. So, with Windows, for example, if you keep the flow cytometry data in sub-directories within "My Documents", the "top_directory.txt" file could simply contain the directory name "Documents". In that case, the "Flow Cytometry Analysis Memory" directory will be automatically createds as a sub-directory to the Documents directory (e.g. C:\Users\username\Documents\Flow Cytometry Analysis Memory).<br>

The "bead_calibration_data.csv" file should contain the calibration data for the fluorescent beads that you use for calibration of fluorescence values measured with flow cytometry. The file contains three columns, for the blu, yellow, and violet laser channels of the cytometer. The data in each column are the MEF (molecules of equivalent fluorophore) for each channel. The values in the file included with this repository are for the Sphereotech Rainbow Calibration Particles, catalog number RCP-30-5A, lot number AJ01.

### Toutorial: example flow cytometry analysis
This repository contains three Jupyter notebooks (in the "\example data\Jupyter notebooks" sub-directory) that demonstrate the use of FlowGateNIST with the included example data:
- Notebook 1.FCS conversion and auto gating.ipynb
- Notebook 2.Calibration with fluorescent beads.ipynb
- Notebook 3.Plot and save dose-response curves.ipynb

These three notebooks contain the typical work flow used by our lab. Each notebook produces a set of diagnostic plots that we examine to make sure the algorithms worked correctly before moving on to the next notebook.
