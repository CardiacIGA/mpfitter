# Error Based Deformation Method

This is the repository that contains the Error Based Deformation Method (EBDM), developed by L. Verberne and supervised by R. Willems, C.V. Verhoosel (TU/e) and O. van der Sluis (TU/e & Philips). The EBDM enables the user to fit or deform a predefined template geometry of the left ventricle to a sparse-point-cloud data set. 

## Installation

The module can be installed in the following steps:

1. Clone the repository into your local folder
2. Open a command window (cmd) or anaconda prompt, and navigate to the local folder
3. Install the module by typing: `pip install -e .`

Future module updates: When new updates are pushed to the repository by the maintainer, you only have to pull them into your local folder. Reinstallation is not required.

## Workflow (Echocardiogram fit) WILL BE MOVED TO ECHO-SEGMENTER

The EBDM algorithm is designed to be applied to an arbitrary point cloud of the left ventricle. The point cloud can be of any density, i.e. dense or sparse, but should have a clear truncation at the base. in order to obtain a deformed or fitted geometry, the following workflow is designed:

1. Data selection (optional)
It is common to first select a that will be segmented. For this project, we rely on echocardiogram (echo) data, but other high-resolution image data is also possible (not implemented in the segmentation folder). If a high-resolution point cloud is provided, skip step 1-3. The user is responsible for selecting an appropriate echo image that will be segmented

2. Data extraction
The 2D echo data is manually segmented in order to obtain data points. The segmentation involves clicking/tracing the endocardium first and secondly the epicardium while starting and ending at the base. An example is included in the example/ folder.

3. Point cloud generation
The 2D segmented echo data are combined in 3D space to form a (sparse) point cloud. Make sure to specify which echo views are to be combined. Since the probe location is unknown, we have to assume that the views perfectly align with the angles as described in the book of [Hamer en Pieper: Praktische echocardiografie](https://link.springer.com/book/10.1007/978-90-368-0752-4). This is primarily the case for the long-axis views. The short axis view will be positioned such that it fits the already positioned long-axis views most optimally. 

4. Fit the template
The provided point cloud is now used as an input for the EBDM algorithm based on a predefined left ventricle (NURBS) template.

## Acknowledgements:
The original code and method is developed by L. Verberne during his MSc. thesis. The method is further improved by R. Willems during his PhD and pior to the publication. Support was provided by Clemens V. Verhoosel and Olaf van der Sluis as part of the [COMBAT-VT project](https://combatvt.nl).

## Citing this module:
Please consider citing this module when using it:

Willems, R., Verberne, L., van der Sluis, O., & Verhoosel, C. V. (2024). Echocardiogram-based ventricular isogeometric cardiac analysis using multi-patch fitted NURBS. Computer Methods in Applied Mechanics and Engineering, 425, 116958. DOI:https://doi.org/10.1016/j.cma.2024.116958