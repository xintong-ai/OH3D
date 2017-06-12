Synopsis

This project was started as a deformation-based scientific visualization research by Xin Tong from The Ohio State University during his PhD study under Prof. Han-Wei Shen. Another student, Cheng Li, later joined this research and extended it. This project is a library containing a lot of toolkits, which can be used for other scientific visualization researches and developments.


Code Example

This project can create an executable program for each subfolder in the 'programs' folder. Other folders are used to build libraries used by these executable programs. The program TutorialVis is a short program better for starters. The executable programs follow the Model–View–Controller design. /programs/TutorialVis/windows.cpp contains rich comments to explain this recommanded way to build programs.

Besides TutorialVis in the 'programs' folder, ParticleVis is used in citations [1] and [2]; TensorVis is used in citation [1]; VolumeVis is used in citation [2]. (please note low-level settings might have been changed since the papers are published. Refer to previous commits of the project for exact reproduction.) Other programs are test programs which are not well maintained and are not recommanded


Installation

Use CMake to build the project.

The required libraries for this project include QT, NVidia GPU (CUDA), and glm (http://glm.g-truc.net/). It has been tested on a Windows environment with QT 5.5 and CUDA 7.5, using Microsoft Visual Studio 2013 and 64-bit built.

Optional settings include:

BUILD_TUTORIAL: build /programs/TutorialVis.

BUILD_TEST: build some programs in 'programs' folder which are still under test.

USE_LEAP: use the Leap Motion controller. Need to install related libraries from (https://www.leapmotion.com/).

USE_OSVR: use the OSVR headset. Need to install related libraries from (http://www.osvr.org/).

USE_TOUCHSCREEN: use touchscreen (not tested yet)

USE_CONTROLLER: use HTC VIVE controller (not tested yet).


Contributors

Xin Tong, email: tongxin829 at gmail dot com

Cheng Li, email: li dot 4076 at gmail dot com


Citation

This program is used for the following publications. Citing it will be appreciated if you use the code for your publication.

[1] X. Tong; C. Li; H. W. Shen, "GlyphLens: View-dependent Occlusion Management in the Interactive Glyph Visualization," in IEEE Transactions on Visualization and Computer Graphics ( Volume: 23, Issue: 1, Jan. 2017, Page(s): 891 - 900) doi: 10.1109/TVCG.2016.2599049 keywords: {Context;Data visualization;Lenses;Probes;Shape;Three-dimensional displays;Visualization;View-dependent visualization;focus + context techniques;glyph-based techniques;human-computer interaction;manipulation and deformation}, URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7539643&isnumber=4359476

[2] C. Li, X. Tong, and H.-W. Shen. Virtual retractor: An interactive data exploration system using physically based deformation. In Visualization Symposium (PacificVis), IEEE Pacific, 2017. (in press)