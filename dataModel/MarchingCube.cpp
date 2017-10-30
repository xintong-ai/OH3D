#include "MarchingCube.h"
#include <PolyMesh.h>
#include <Algorithm>

#include <vtkXMLImageDataReader.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkPoints.h>
#include <vtkCell.h>



#include <vtkPolyDataReader.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkSmartPointer.h>


#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataNormals.h>
#include <vtkPointData.h>
#include <vtkSphereSource.h>
#include <vtkCell.h>


MarchingCube::MarchingCube(const char * fname, std::shared_ptr<PolyMesh> p, float value)
{
	polyMesh = p;
	isoValue0 = value;

	if (forNav){
		isoValue0 = 0.0006;
		isoValue1 = 0.0011;
		isoValue2 = 0.0014;
	}
	vtkSmartPointer<vtkXMLImageDataReader> reader =
		vtkSmartPointer<vtkXMLImageDataReader>::New();
	reader->SetFileName(fname);
	reader->Update();

	vtkSmartPointer<vtkImageData> img = reader->GetOutput();
	int3 dataSizes;
	img->GetDimensions(&(dataSizes.x));

	vtkSmartPointer<vtkFloatArray> array2 = vtkFloatArray::SafeDownCast(img->GetPointData()->GetArray("XW 2=    CO2     ")); // !!!!!!!!!!!!!!!! currently only for this case !!!!!!!!!!!!!!!


	double spacing[3] = { 1, 1, 1 };
	double origin[3] = { 0, 0, 0 };
	int dim[3] = { dataSizes.x, dataSizes.y, dataSizes.z };
	inputImage = vtkSmartPointer<vtkImageData>::New();
	inputImage->SetSpacing(spacing);
	inputImage->SetDimensions(dim);
	inputImage->SetOrigin(origin);
#if VTK_MAJOR_VERSION <= 5
	inputImage->SetScalarTypeToUnsignedChar();
	inputImage->AllocateScalars();
#else
	inputImage->AllocateScalars(VTK_FLOAT, 1);
#endif
	// fill the image with foreground voxels:
	unsigned char inval = 0;
	unsigned char outval = 1;
	vtkIdType count = inputImage->GetNumberOfPoints();
	for (vtkIdType i = 0; i < count; ++i)
	{
		inputImage->GetPointData()->GetScalars()->SetTuple1(i, inval);
	}
	for (int k = 0; k < dataSizes.z; k++)
	{
		for (int j = 0; j < dataSizes.y; j++)
		{
			for (int i = 0; i < dataSizes.x; i++)
			{
				int ind = k*dataSizes.y * dataSizes.x + j*dataSizes.x + i;
				float p = array2->GetValue(ind);
				inputImage->GetPointData()->GetScalars()->SetTuple1(ind, p);
			}
		}
	}

	surface = vtkSmartPointer<vtkMarchingCubes>::New();
	surface->SetInputData(inputImage);
	surface->ComputeNormalsOn();
	surface->SetValue(0, isoValue0);
	surface->SetValue(1, isoValue1);
	if (forNav){
		surface->SetValue(2, isoValue2);
	}

	vtpdata = vtkSmartPointer<vtkPolyData>::New();


	computeIsosurface();

	needCompute = false;
	updatePoly();
	polyMesh->SetPosRange(make_float3(0, 0, 0), make_float3(dataSizes.x, dataSizes.y, dataSizes.z));  //one time call should be enough
}

void MarchingCube::computeIsosurface()
{
	surface->Update();
	vtpdata = surface->GetOutput();

	//vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
	//writer->SetFileName("iso.vtp");
	//writer->SetInputData(surface->GetOutput());
	//writer->Write();
}

void MarchingCube::newIsoValue(float v, int index)
{
	surface->SetValue(index, v);
	surface->Update();
	vtpdata = surface->GetOutput();
	updatePoly();
}

void MarchingCube::updatePoly()
{
	int vertexcount = vtpdata->GetNumberOfPoints();
	int facecount = vtpdata->GetNumberOfCells();

	if (vertexcount > polyMesh->vertexcount || facecount > polyMesh->facecount)
	{
		polyMesh->~PolyMesh();

		polyMesh->vertexcount = vertexcount;
		polyMesh->facecount = facecount;

		polyMesh->vertexCoords = new float[3 * polyMesh->vertexcount];
		polyMesh->vertexNorms = new float[3 * polyMesh->vertexcount];
		polyMesh->indices = new unsigned[3 * polyMesh->facecount];

		polyMesh->vertexColorVals = new float[polyMesh->vertexcount * 2]; //times 2 to prepare for newly added vertices
		memset((void*)(polyMesh->vertexColorVals + vertexcount), 0, sizeof(float)* vertexcount);//the rest will always be set to 0 regardless of v
	}
	else{
		polyMesh->vertexcount = vertexcount;
		polyMesh->facecount = facecount;
	}


	//vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
	//writer->SetFileName("test.vtp");
	//writer->SetInputData(vtpdata);
	//writer->Write();

	//three possible ways to get the normals
	vtkFloatArray* normalDataFloat = vtkFloatArray::SafeDownCast(vtpdata->GetPointData()->GetNormals());
	//vtkFloatArray* normalDataFloat = vtkFloatArray::SafeDownCast(vtpdata->GetPointData()->GetArray("Normals_"));
	//vtkDataArray* normalsGeneric = polydata->GetPointData()->GetNormals(); 
	vtkFloatArray* scalarDataFloat = vtkFloatArray::SafeDownCast(vtpdata->GetPointData()->GetScalars(0));

	vtkFloatArray* normalDataFloat2 = vtkFloatArray::SafeDownCast(vtpdata->GetPointData()->GetArray("Scalars_"));

	float vmin = std::min(isoValue0, isoValue1), vmax = std::max(isoValue0, isoValue1);
	if (forNav){
		vmax = isoValue2;
	}

	if (vmax - vmin < 0.000001) vmax = vmin + 1;

	for (int i = 0; i < polyMesh->vertexcount; i++) {
		double coord[3];
		vtpdata->GetPoint(i, coord);

		polyMesh->vertexCoords[3 * i] = coord[0];
		polyMesh->vertexCoords[3 * i + 1] = coord[1];
		polyMesh->vertexCoords[3 * i + 2] = coord[2];
		polyMesh->vertexNorms[3 * i] = normalDataFloat->GetComponent(i, 0);
		polyMesh->vertexNorms[3 * i + 1] = normalDataFloat->GetComponent(i, 1);
		polyMesh->vertexNorms[3 * i + 2] = normalDataFloat->GetComponent(i, 2);

		float zz = scalarDataFloat->GetValue(i);
		polyMesh->vertexColorVals[i] = (zz - vmin) / (vmax - vmin);
	}


	for (int i = 0; i < polyMesh->facecount; i++) {
		if (vtpdata->GetCell(i)->GetNumberOfPoints() != 3){
			std::cout << "readed PLY data contains non-triangles. the current program cannot handle" << std::endl;
			exit(0);
		}

		polyMesh->indices[3 * i] = vtpdata->GetCell(i)->GetPointId(0);
		polyMesh->indices[3 * i + 1] = vtpdata->GetCell(i)->GetPointId(1);
		polyMesh->indices[3 * i + 2] = vtpdata->GetCell(i)->GetPointId(2);
	}
}
