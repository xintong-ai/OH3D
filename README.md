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

Xin Tong and Cheng Li

## License
It is currently close source. Only students in The Ohio State Univeristy can use it. Xin is thinking about making it open source under BSD licence in the future.
