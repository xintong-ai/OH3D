## Synopsis

This project was started as a deformation-based scientific visualization research by Xin Tong from The Ohio State University during his PhD study under Prof. Han-Wei Shen. Another student, Cheng Li, later joined this research and extended it. This project is a library containing a lot of toolkits, which can be used for other scientific visualization researches and developments.

## Code Example

## Installation
Use CMake to build the project.
 
To compile on the computer that does not have NVIDIA GPU, use the options:
-DUSE_DEFORM=OFF -DUSE_VOLUME_RENDERING=OFF -DCMAKE_BUILD_TYPE=DEBUG

To generate Xcode:
cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSE_DEFORM=OFF -DUSE_VOLUME_RENDERING=OFF -G Xcode 

## Contributors

Xin Tong, email: tongxin829@gmail.com
Cheng Li

## Citation
This program is used for the following publication. Citing it will be appreciated if you use the code.

X. Tong; C. Li; H. W. Shen, "GlyphLens: View-dependent Occlusion Management in the Interactive Glyph Visualization," in IEEE Transactions on Visualization and Computer Graphics , vol.PP, no.99, pp.1-1
doi: 10.1109/TVCG.2016.2599049
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7539643&isnumber=4359476
