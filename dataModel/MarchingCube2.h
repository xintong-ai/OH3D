#ifndef MRACHING_CUBE2_H
#define MRACHING_CUBE2_H

#include <memory>

#include <numeric>
#include <math.h>

#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkMarchingCubes.h>
#include <vtkPolyData.h>

class PolyMesh;

class MarchingCube2
{
public:
	std::shared_ptr<PolyMesh> polyMesh = 0;

	float isoValue = -1;
	bool needCompute = false;
	MarchingCube2(const char * fname, std::shared_ptr<PolyMesh> p, float value = 0.001);

	void newIsoValue(float v);


private:
	vtkSmartPointer<vtkMarchingCubes> surface;
	vtkSmartPointer<vtkImageData> inputImage;
	vtkSmartPointer<vtkPolyData> vtpdata;

	void computeIsosurface();
	void updatePoly();

};

#endif //ARROW_RENDERABLE_H