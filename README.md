## Synopsis

This project was started as a deformation-based scientific visualization research by Xin Tong from The Ohio State University during his PhD study under Prof. Han-Wei Shen. Another student, Cheng Li, later joined this research and extended it. This project is a library containing a lot of toolkits, which can be used for other scientific visualization researches and developments.

## Code Example

This project follows the Model–View–Controller design. Generally you need to first prepare objects of data (Model); then prepare objects of the Renderable class (View), and connect it with data objects; besides QT interaction, you can prepare objects of Interactor class (Controller), and connect it with data objects. At last, connect the Renderable objects and the Interactor objects with an object of GLWidget (inherited from QOpenGLWidget).

The program ImmersiveVolumeVis is a short program better for starters.

## Installation

Use CMake to build the project.

The required libraries for this project include QT, NVidia GPU (CUDA), and glm (http://glm.g-truc.net/). It has been tested on a Windows environment with QT 5.5 and CUDA 7.5, using Microsoft Visual Studio 2013 and 64-bit built.

Optional settings include:

Leap Motion controller: set USE_LEAP=ON in CMake
OSVR headset: set USE_OSVR=ON in CMake
Touchscreen: set USE_TOUCHSCREEN=ON in CMake (not tested yet)
HTC VIVE controller: set USE_CONTROLLER=ON in CMake (not tested yet)

## Contributors

Xin Tong, email: tongxin829 at gmail dot com
Cheng Li, email: li dot 4076 at gmail dot com


## Citation

This program is used for the following publication. Citing it will be appreciated if you use the code for your publication.

X. Tong; C. Li; H. W. Shen, "GlyphLens: View-dependent Occlusion Management in the Interactive Glyph Visualization," in IEEE Transactions on Visualization and Computer Graphics ( Volume: 23, Issue: 1, Jan. 2017, Page(s): 891 - 900)
doi: 10.1109/TVCG.2016.2599049
keywords: {Context;Data visualization;Lenses;Probes;Shape;Three-dimensional displays;Visualization;View-dependent visualization;focus + context techniques;glyph-based techniques;human-computer interaction;manipulation and deformation},
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7539643&isnumber=4359476