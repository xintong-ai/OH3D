#ifndef VTP_READER_H
#define VTP_READER_H

#include <fstream>
#include <iostream>
#include <memory>
#define _USE_MATH_DEFINES
#include "math.h"
#include "vector_types.h"
class PolyMesh;
class vtkPolyData;

class VTPReader
{
public:
	void readFile(const char* fname, PolyMesh* polyMesh);
	
private:
	//VTK/Examples/Cxx/PolyData/PolyDataExtractNormals
	bool GetPointNormals(vtkPolyData* polydata);

};



#endif