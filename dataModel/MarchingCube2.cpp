#include "MarchingCube2.h"
#include <PolyMesh.h>

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


MarchingCube2::MarchingCube2(const char * fname, std::shared_ptr<PolyMesh> p, float value)
{
	polyMesh = p;
	isoValue = value;
	
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
inputImage = 	vtkSmartPointer<vtkImageData>::New();
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
	surface->SetValue(0, 0.001);

	vtpdata = vtkSmartPointer<vtkPolyData>::New();


	computeIsosurface();

	needCompute = false;
	updatePoly();
	polyMesh->SetPosRange(make_float3(0, 0, 0), make_float3(dataSizes.x, dataSizes.y, dataSizes.z));  //one time call should be enough
}	

void MarchingCube2::computeIsosurface()
{
	surface->Update();
	vtpdata = surface->GetOutput();

	//vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
	//writer->SetFileName("iso.vtp");
	//writer->SetInputData(surface->GetOutput());
	//writer->Write();
}

void MarchingCube2::newIsoValue(float v)
{
	surface->SetValue(0, v);
	surface->Update();
	vtpdata = surface->GetOutput();
	updatePoly();
}

void MarchingCube2::updatePoly()
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


	for (int i = 0; i < polyMesh->vertexcount; i++) {
		double coord[3];
		vtpdata->GetPoint(i, coord);

		polyMesh->vertexCoords[3 * i] = coord[0];
		polyMesh->vertexCoords[3 * i + 1] = coord[1];
		polyMesh->vertexCoords[3 * i + 2] = coord[2];
		polyMesh->vertexNorms[3 * i] = normalDataFloat->GetComponent(i, 0);
		polyMesh->vertexNorms[3 * i + 1] = normalDataFloat->GetComponent(i, 1);
		polyMesh->vertexNorms[3 * i + 2] = normalDataFloat->GetComponent(i, 2);
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
