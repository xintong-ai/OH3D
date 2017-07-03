#include "VTPReader.h"
#include "PolyMesh.h"

#include <vtkXMLPolyDataReader.h>
#include <vtkSmartPointer.h>


#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataNormals.h>
#include <vtkPointData.h>
#include <vtkSphereSource.h>

void VTPReader::readFile(const char* filename, PolyMesh* polyMesh)
{
	vtkSmartPointer<vtkXMLPolyDataReader> reader =
		vtkSmartPointer<vtkXMLPolyDataReader>::New();
	reader->SetFileName(filename);
	reader->Update();

	vtkSmartPointer<vtkPolyData> data = vtkSmartPointer<vtkPolyData>::New();
	data = reader->GetOutput();
	
	std::cout << "vertexcount " << data->GetNumberOfPoints() << std::endl;
	std::cout << "facecount: " << data->GetNumberOfCells() << std::endl;

	
	// Try to read normals directly
	bool hasPointNormals = GetPointNormals(data);

	if (!hasPointNormals)
	{
//		std::cout << "No point normals were found. Computing normals..." << std::endl;
//
//		// Generate normals
//		vtkSmartPointer<vtkPolyDataNormals> normalGenerator = vtkSmartPointer<vtkPolyDataNormals>::New();
//#if VTK_MAJOR_VERSION <= 5
//		normalGenerator->SetInput(polydata);
//#else
//		normalGenerator->SetInputData(data);
//#endif
//		normalGenerator->ComputePointNormalsOn();
//		normalGenerator->ComputeCellNormalsOff();
//		/*
//		// Optional settings
//		normalGenerator->SetFeatureAngle(0.1);
//		normalGenerator->SetSplitting(1);
//		normalGenerator->SetConsistency(0);
//		normalGenerator->SetAutoOrientNormals(0);
//		normalGenerator->SetComputePointNormals(1);
//		normalGenerator->SetComputeCellNormals(0);
//		normalGenerator->SetFlipNormals(0);
//		normalGenerator->SetNonManifoldTraversal(1);
//		*/
//		normalGenerator->SetSplitting(0);
//		normalGenerator->SetConsistency(1);
//
//		normalGenerator->Update();
//
//		data = normalGenerator->GetOutput();
//
//		// Try to read normals again
//		hasPointNormals = GetPointNormals(data);
//
//		std::cout << "On the second try, has point normals? " << hasPointNormals << std::endl;
//		if (!hasPointNormals){
//			std::cout << "fail computing normals" << std::endl;
//			exit(0);
//		}
	}
	else
	{
		std::cout << "Point normals were found!" << std::endl;
	}


	polyMesh->vertexcount = data->GetNumberOfPoints();
	polyMesh->facecount = data->GetNumberOfCells();

	polyMesh->vertexCoords = new float[3 * polyMesh->vertexcount];
	polyMesh->vertexNorms = new float[3 * polyMesh->vertexcount];

	vtkFloatArray* normalDataFloat = vtkFloatArray::SafeDownCast(data->GetPointData()->GetArray("Normals"));

	for (int i = 0; i < polyMesh->vertexcount; i++) {
		double coord[3];
		data->GetPoint(i, coord);

		polyMesh->vertexCoords[3 * i] = coord[0];
		polyMesh->vertexCoords[3 * i + 1] = coord[1];
		polyMesh->vertexCoords[3 * i + 2] = coord[2];
		polyMesh->vertexNorms[3 * i] = normalDataFloat->GetComponent(i, 0);
		polyMesh->vertexNorms[3 * i + 1] = normalDataFloat->GetComponent(i, 1);
		polyMesh->vertexNorms[3 * i + 2] = normalDataFloat->GetComponent(i, 2);
	}

	polyMesh->indices = new unsigned[3 * polyMesh->facecount];

	for (int i = 0; i < polyMesh->facecount; i++) {
		if (data->GetCell(i)->GetNumberOfPoints() != 3){
			std::cout << "readed PLY data contains non-triangles. the current program cannot handle" << std::endl;
			exit(0);
		}

		polyMesh->indices[3 * i] = data->GetCell(i)->GetPointId(0);
		polyMesh->indices[3 * i + 1] = data->GetCell(i)->GetPointId(1);
		polyMesh->indices[3 * i + 2] = data->GetCell(i)->GetPointId(2);
	}
	polyMesh->find_center_and_range();

}



bool VTPReader::GetPointNormals(vtkPolyData* polydata)
{
	//std::cout << "In GetPointNormals: " << polydata->GetNumberOfPoints() << std::endl;
	//std::cout << "Looking for point normals..." << std::endl;

	// Count points
	vtkIdType numPoints = polydata->GetNumberOfPoints();
	std::cout << "There are " << numPoints << " points." << std::endl;

	// Count triangles
	vtkIdType numPolys = polydata->GetNumberOfPolys();
	std::cout << "There are " << numPolys << " polys." << std::endl;

	////////////////////////////////////////////////////////////////
	// Double normals in an array
	vtkDoubleArray* normalDataDouble =
		vtkDoubleArray::SafeDownCast(polydata->GetPointData()->GetArray("Normals"));

	if (normalDataDouble)
	{
		int nc = normalDataDouble->GetNumberOfTuples();
		std::cout << "There are " << nc
			<< " components in normalDataDouble" << std::endl;
		return true;
	}

	////////////////////////////////////////////////////////////////
	// Double normals in an array
	vtkFloatArray* normalDataFloat =
		vtkFloatArray::SafeDownCast(polydata->GetPointData()->GetArray("Normals"));

	if (normalDataFloat)
	{
		int nc = normalDataFloat->GetNumberOfTuples();
		std::cout << "There are " << nc
			<< " components in normalDataFloat" << std::endl;
		return true;
	}

	////////////////////////////////////////////////////////////////
	// Point normals
	vtkDoubleArray* normalsDouble =
		vtkDoubleArray::SafeDownCast(polydata->GetPointData()->GetNormals());

	if (normalsDouble)
	{
		std::cout << "There are " << normalsDouble->GetNumberOfComponents()
			<< " components in normalsDouble" << std::endl;
		return true;
	}

	////////////////////////////////////////////////////////////////
	// Point normals
	vtkFloatArray* normalsFloat =
		vtkFloatArray::SafeDownCast(polydata->GetPointData()->GetNormals());

	if (normalsFloat)
	{
		std::cout << "There are " << normalsFloat->GetNumberOfComponents()
			<< " components in normalsFloat" << std::endl;
		return true;
	}

	/////////////////////////////////////////////////////////////////////
	// Generic type point normals
	vtkDataArray* normalsGeneric = polydata->GetPointData()->GetNormals(); //works
	if (normalsGeneric)
	{
		std::cout << "There are " << normalsGeneric->GetNumberOfTuples()
			<< " normals in normalsGeneric" << std::endl;

		double testDouble[3];
		normalsGeneric->GetTuple(0, testDouble);

		std::cout << "Double: " << testDouble[0] << " "
			<< testDouble[1] << " " << testDouble[2] << std::endl;

		// Can't do this:
		/*
		float testFloat[3];
		normalsGeneric->GetTuple(0, testFloat);

		std::cout << "Float: " << testFloat[0] << " "
		<< testFloat[1] << " " << testFloat[2] << std::endl;
		*/
		return true;
	}


	// If the function has not yet quit, there were none of these types of normals
	std::cout << "Normals not found!" << std::endl;
	return false;

}
