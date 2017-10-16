#ifndef PLYVTK_READER_H
#define PLYVTK_READER_H

#include <fstream>
#include <iostream>
#include <memory>
#define _USE_MATH_DEFINES
#include "math.h"
#include "vector_types.h"
class PolyMesh;
class vtkPolyData;

class PlyVTKReader
{
public:
	void readPLYByVTK(const char* fname, PolyMesh* polyMesh);
	//the reason to use VTK is the common ply reader cannot handle too big ply files
	//unfortunately, by now I could not use the vtk reader to read normal info correctly, so the normal will be recomputed

private:
	//VTK/Examples/Cxx/PolyData/PolyDataExtractNormals
	bool GetPointNormals(vtkPolyData* polydata);

};



#endif