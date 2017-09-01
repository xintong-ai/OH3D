#include <memory>
#include <string>
#include <iostream>
#include <numeric>

//#include <cuda_runtime.h>
//#include <helper_cuda.h>

#include "Volume.h"
#include "RawVolumeReader.h"
#include "DataMgr.h"
#include "VecReader.h"

#include "myDefineRayCasting.h"

#include <itkImage.h>
#include <helper_timer.h>

#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkImplicitModeller.h>
#include <vtkSphereSource.h>
#include <vtkPolyData.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkImageData.h>
//#include <vtkXMLImageDataWriter.h>
#include <vtkNIFTIImageWriter.h>
#include <vtkImplicitModeller.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkImageResample.h>
#include "itkResampleImageFilter.h"
#include <vtkPLYReader.h>
#include <vtkPolyDataConnectivityFilter.h>
#include <vtkAbstractArray.h>
#include <vtkFieldData.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkLongLongArray.h>
#include <vtkIntArray.h>
#include <vtkPoints.h>
#include <vtkTriangle.h>
#include <vtkCell.h>
#include <vtkCellArray.h>
#include <vtkContourGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include <vtkUnstructuredGrid.h>

#include <helper_math.h>



#include <vtkXMLUnstructuredGridReader.h>
#include <vtkSmartPointer.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkFloatArray.h>
#include <vtkPolyDataNormals.h>

using namespace std;

const int PHASES_COUNT = 1; //phase 1 is for creating cell volume; phase 2 is for creating skeletion; phase 3 is for creating bilateral volume



void createSphere()
{
	vtkSmartPointer<vtkSphereSource> sphereSource =
		vtkSmartPointer<vtkSphereSource>::New();
	sphereSource->SetCenter(30.0, 30.0, 30.0);
	sphereSource->SetRadius(25);
	sphereSource->LatLongTessellationOff();
	sphereSource->SetThetaResolution(30);
	sphereSource->SetPhiResolution(30);

	sphereSource->Update();
	vtkSmartPointer<vtkPolyData> polydata = sphereSource->GetOutput();

	vtkSmartPointer<vtkXMLPolyDataWriter> writer =
		vtkSmartPointer<vtkXMLPolyDataWriter>::New();
	writer->SetFileName("sphere.vtp");
#if VTK_MAJOR_VERSION <= 5
	writer->SetInput(polydata);
#else
	writer->SetInputData(polydata);
#endif

	writer->Write();
}

//for blood cell data, separate each cell
void labelPoly()
{
	vtkSmartPointer<vtkPolyData> data = vtkSmartPointer<vtkPolyData>::New();

	std::shared_ptr<DataMgr> dataMgr;
	dataMgr = std::make_shared<DataMgr>();
	string dataPath = dataMgr->GetConfig("POLY_DATA_PATH");

	if (std::string(dataPath).find(".ply") != std::string::npos){
		vtkSmartPointer<vtkPLYReader> reader =
			vtkSmartPointer<vtkPLYReader>::New();
		reader->SetFileName(dataPath.c_str());
		data = reader->GetOutput();
		reader->Update();
	}
	else{
		vtkSmartPointer<vtkXMLPolyDataReader> reader =
			vtkSmartPointer<vtkXMLPolyDataReader>::New();
		reader->SetFileName(dataPath.c_str());
		data = reader->GetOutput();
		reader->Update();
	}

	vtkSmartPointer<vtkPolyDataConnectivityFilter> connectivityFilter =
		vtkSmartPointer<vtkPolyDataConnectivityFilter>::New();
	connectivityFilter->SetInputData(data);
	connectivityFilter->SetExtractionModeToAllRegions();
	connectivityFilter->ColorRegionsOn();
	connectivityFilter->Update();


	vtkSmartPointer<vtkXMLPolyDataWriter> writer =
		vtkSmartPointer<vtkXMLPolyDataWriter>::New();
	writer->SetFileName("labeled.vtp");
	writer->SetInputData(connectivityFilter->GetOutput());
	writer->Write();

	vtkSmartPointer<vtkPolyData> res = connectivityFilter->GetOutput();
	vtkLongLongArray *arrayRegionId = (vtkLongLongArray *)((res->GetPointData())->GetArray("RegionId"));
	vtkPoints * points = res->GetPoints();

	int n = arrayRegionId->GetNumberOfTuples();
	cout << "count of points: " << n << endl;
	long long range[2];
	arrayRegionId->GetValueRange(range);
	cout << "range of region id: " << range[0] << " " << range[1] << endl;
	int nRegion = range[1] - range[0] + 1;

	//check if points belonging to the same region are consecutive or not
	long long curid = 0;
	for (int i = 0; i < n; i++){
		long long d = arrayRegionId->GetValue(i);
		if (curid == d){
		}
		else if (curid + 1 == d){
			curid++;
		}
		else {
			cout << "not consecutive region id at " << i << " with id " << curid << endl;
			exit(0); //not implemented currently
		}
	}
	cout << "consecutive region ids for vertices checked" << endl;

	vector<int> count(nRegion, 0);
	vector<double3> posSum(nRegion, make_double3(0, 0, 0));
	vector<float3> minPos(nRegion, make_float3(999999, 999999, 999999));
	vector<float3> maxPos(nRegion, make_float3(-999999, -999999, -999999));

	for (int i = 0; i < n; i++){
		long long d = arrayRegionId->GetValue(i);
		double * coord = points->GetPoint(i);
		posSum[d] = make_double3(posSum[d].x + coord[0], posSum[d].y + coord[1], posSum[d].z + coord[2]);
		count[d]++;
		minPos[d] = make_float3(fmin(minPos[d].x, coord[0]), fmin(minPos[d].y, coord[1]), fmin(minPos[d].z, coord[2]));
		maxPos[d] = make_float3(fmax(maxPos[d].x, coord[0]), fmax(maxPos[d].y, coord[1]), fmax(maxPos[d].z, coord[2]));
	}
	vector<float3> posAve(nRegion);
	for (int i = 0; i < nRegion; i++){
		posAve[i] = make_float3(posSum[i].x / count[i], posSum[i].y / count[i], posSum[i].z / count[i]);
		//cout << "min " << minPos[i].x << " " << minPos[i].y << " " << minPos[i].z << endl;
		//cout << "max " << maxPos[i].x << " " << maxPos[i].y << " " << maxPos[i].z << endl;
	}


	vector<int> countFace(nRegion, 0);
	curid = 0;
	int m = res->GetNumberOfCells();
	for (int i = 0; i < m; i++){
		int vertexId = data->GetCell(i)->GetPointId(0);
		int d = arrayRegionId->GetValue(vertexId);
		if (curid == d){
			countFace[d]++;
		}
		else if (curid + 1 == d){
			curid++;
			countFace[d]++;
		}
		else {
			cout << "not consecutive region id at face " << i << " with id " << curid << endl;
			exit(0); //not implemented currently
		}
	}
	cout << "consecutive region ids for faces checked" << endl;


	int temp = 1;
	FILE * fp = fopen("polyMeshRegions.mytup", "wb");
	fwrite(&temp, sizeof(int), 1, fp);

	fwrite(&nRegion, sizeof(int), 1, fp);
	int nc = 14;
	fwrite(&nc, sizeof(int), 1, fp);
	int startv = 0, startf = 0;
	for (int i = 0; i < nRegion; i++){
		float startvf = startv;
		float endvf = startv + count[i] - 1;
		float startff = startf;
		float endff = startf + countFace[i] - 1;
		fwrite(&(posAve[i].x), sizeof(float3), 1, fp);
		fwrite(&startff, sizeof(float), 1, fp); //range of faces of the current region
		fwrite(&endff, sizeof(float), 1, fp);
		fwrite(&startvf, sizeof(float), 1, fp); //range of vertices of the current region
		fwrite(&endvf, sizeof(float), 1, fp);
		fwrite(&(minPos[i].x), sizeof(float3), 1, fp); //bounding box of the current region
		fwrite(&(maxPos[i].x), sizeof(float3), 1, fp);
		float tempi = i;
		fwrite(&tempi, sizeof(int), 1, fp); //id of the region

		startv = startv + count[i];
		startf = startf + countFace[i];
	}

}


void reduceBloodCell()	//also generate the normal
{
	int startTs = 6, endTs = 32;
	float yThr = 112;
	for (int i = startTs; i <= endTs; i++){
		stringstream ss;
		ss << setw(4) << setfill('0') << i;
		string s = ss.str();

		string inputFileName = "D:/Data/Lin/Flow Simulations with Red Blood Cells/uDeviceX/ply/rbcs-" + s + ".ply";

		vtkSmartPointer<vtkPolyData> data = vtkSmartPointer<vtkPolyData>::New();
		if (std::string(inputFileName).find(".ply") != std::string::npos){
			vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
			reader->SetFileName(inputFileName.c_str());
			data = reader->GetOutput();
			reader->Update();
		}
		else{
			std::cout << "file name not defined" << std::endl;
			exit(0);
		}
		std::cout << "vertexcount " << data->GetNumberOfPoints() << std::endl;
		std::cout << "facecount: " << data->GetNumberOfCells() << std::endl;

		//assume this filter will not change original points and cells
		vtkSmartPointer<vtkPolyDataConnectivityFilter> connectivityFilter =
			vtkSmartPointer<vtkPolyDataConnectivityFilter>::New();
		connectivityFilter->SetInputData(data);
		connectivityFilter->SetExtractionModeToAllRegions();
		connectivityFilter->ColorRegionsOn();
		connectivityFilter->Update();

		vtkSmartPointer<vtkPolyData> res = connectivityFilter->GetOutput();
		vtkIntArray *arrayRegionId = (vtkIntArray *)((res->GetPointData())->GetArray("RegionId"));
		//vtkPoints * points = res->GetPoints();

		int n = arrayRegionId->GetNumberOfTuples();
		//cout << "count of points: " << n << endl;
		int range[2];
		arrayRegionId->GetValueRange(range);
		//cout << "range of region id: " << range[0] << " " << range[1] << endl;
		int nRegion = range[1] - range[0] + 1;

		//check if points belonging to the same region are consecutive or not
		long long curid = 0;
		for (int i = 0; i < n; i++){
			long long d = arrayRegionId->GetValue(i);
			if (curid == d){
			}
			else if (curid + 1 == d){
				curid++;
			}
			else {
				cout << "not consecutive region id at " << i << " with id " << curid << endl;
				exit(0); //not implemented currently
			}
		}
		cout << "consecutive region ids for vertices checked" << endl;

		vector<int> count(nRegion, 0);
		vector<double3> posSum(nRegion, make_double3(0, 0, 0));
		for (int i = 0; i < n; i++){
			long long d = arrayRegionId->GetValue(i);
			double * coord = data->GetPoint(i);
			posSum[d] = make_double3(posSum[d].x + coord[0], posSum[d].y + coord[1], posSum[d].z + coord[2]);
			count[d]++;
		}
		vector<float3> posAve(nRegion);
		for (int i = 0; i < nRegion; i++){
			posAve[i] = make_float3(posSum[i].x / count[i], posSum[i].y / count[i], posSum[i].z / count[i]);
		}



		vector<int> table(data->GetNumberOfPoints(), -1);
		int newid = 0;

		//http://www.vtk.org/Wiki/VTK/Examples/Cxx/PolyData/DeletePoint
		int vertexcount = data->GetNumberOfPoints();
		vtkSmartPointer<vtkPoints> newPoints =
			vtkSmartPointer<vtkPoints>::New();
		for (int i = 0; i < vertexcount; i++) {
			//double coord[3];
			//data->GetPoint(i, coord);
			if (!(posAve[arrayRegionId->GetValue(i)].y > yThr)){
				double coord[3];
				data->GetPoint(i, coord);
				newPoints->InsertNextPoint(coord);
				table[i] = newid;
				newid++;
			}
		}

		vtkSmartPointer<vtkCellArray> triangles =
			vtkSmartPointer<vtkCellArray>::New();
		int facecount = data->GetNumberOfCells();
		for (int i = 0; i < facecount; i++) {
			bool needDel = false;
			int id[3];
			for (int j = 0; j < 3; j++){
				id[j] = data->GetCell(i)->GetPointId(j);
			}

			if (posAve[arrayRegionId->GetValue(id[0])].y > yThr){
		
			}
			else{
				vtkSmartPointer<vtkTriangle> triangle =
					vtkSmartPointer<vtkTriangle>::New();
				for (int j = 0; j < 3; j++){
					if (table[id[j]] < 0){
						cout << "removing error" << endl;
						exit(0);
					}
					triangle->GetPointIds()->SetId(j, table[id[j]]);
				}
				triangles->InsertNextCell(triangle);
			}
		}

		data->GetPoints()->ShallowCopy(newPoints);
		data->SetPolys(triangles);

		vtkSmartPointer<vtkPolyDataNormals> normalGenerator = vtkSmartPointer<vtkPolyDataNormals>::New();
#if VTK_MAJOR_VERSION <= 5
		normalGenerator->SetInput(data);
#else
		normalGenerator->SetInputData(data);
#endif
		normalGenerator->ComputePointNormalsOn();
		normalGenerator->ComputeCellNormalsOff();
		normalGenerator->SetSplitting(0);
		normalGenerator->SetConsistency(1);
		normalGenerator->Update();
		data = normalGenerator->GetOutput();


		vtkSmartPointer<vtkXMLPolyDataWriter> writer4 = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
		writer4->SetFileName(("D:/Data/Lin/reducedBloodCell/reduced-rbcs-" + s + ".vtp").c_str());
		writer4->SetInputData(data);
		writer4->Write();


		std::cout << "vertexcount " << data->GetNumberOfPoints() << std::endl;
		std::cout << "facecount: " << data->GetNumberOfCells() << std::endl;
	}



	//wall
	string inputFileName = "D:/Data/Lin/reducedBloodCell/wall.vtp";

	vtkSmartPointer<vtkPolyData> data = vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkXMLPolyDataReader> reader = vtkSmartPointer<vtkXMLPolyDataReader>::New();
	reader->SetFileName(inputFileName.c_str());
	data = reader->GetOutput();
	reader->Update();

	std::cout << "vertexcount " << data->GetNumberOfPoints() << std::endl;
	std::cout << "facecount: " << data->GetNumberOfCells() << std::endl;



	//vtkFloatArray* normalDataFloat = vtkFloatArray::SafeDownCast(data->GetPointData()->GetArray("Normals"));
	//polyMesh->vertexNorms[3 * i] = normalDataFloat->GetComponent(i, 0);
	//vector<float3> wallnormal(data->GetNumberOfPoints());

	vector<int> table(data->GetNumberOfPoints(), -1);
	int newid = 0;
	//http://www.vtk.org/Wiki/VTK/Examples/Cxx/PolyData/DeletePoint
	int vertexcount = data->GetNumberOfPoints();
	vtkSmartPointer<vtkPoints> newPoints = vtkSmartPointer<vtkPoints>::New();
	for (int i = 0; i < vertexcount; i++) {
		double coord[3];
		data->GetPoint(i, coord);
		if (!(coord[1] > yThr)){
			double coord[3];
			data->GetPoint(i, coord);
			newPoints->InsertNextPoint(coord);
			table[i] = newid;
			newid++;
		}
	}
	

	vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();
	int facecount = data->GetNumberOfCells();
	for (int i = 0; i < facecount; i++) {
		bool needDel = false;
		int id[3];
		double yOfVertices[3];
		for (int j = 0; j < 3; j++){
			id[j] = data->GetCell(i)->GetPointId(j);
			double coord[3];
			data->GetPoint(id[j], coord);
			yOfVertices[j] = coord[1];
		}

		if (yOfVertices[0] > yThr || yOfVertices[1] > yThr || yOfVertices[2] > yThr){

		}
		else{
			vtkSmartPointer<vtkTriangle> triangle =
				vtkSmartPointer<vtkTriangle>::New();
			for (int j = 0; j < 3; j++){
				triangle->GetPointIds()->SetId(j, table[id[j]]);
			}
			triangles->InsertNextCell(triangle);
		}
	}

	vtkSmartPointer<vtkPolyData> datanew = vtkSmartPointer<vtkPolyData>::New();
	datanew->SetPoints(newPoints);
	//datanew->GetPoints()->ShallowCopy(newPoints);
	datanew->SetPolys(triangles);

	vtkSmartPointer<vtkPolyDataNormals> normalGenerator = vtkSmartPointer<vtkPolyDataNormals>::New();
#if VTK_MAJOR_VERSION <= 5
	normalGenerator->SetInput(polydata);
#else
	normalGenerator->SetInputData(datanew);
#endif
	normalGenerator->ComputePointNormalsOn();
	normalGenerator->ComputeCellNormalsOff();
	normalGenerator->SetSplitting(0);
	normalGenerator->SetConsistency(1);
	normalGenerator->Update();
	datanew = normalGenerator->GetOutput();



	vtkSmartPointer<vtkXMLPolyDataWriter> writer4 = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
	writer4->SetFileName("D:/Data/Lin/reducedBloodCell/reduced-wall.vtp");
	writer4->SetInputData(datanew);
	writer4->Write();


	std::cout << "vertexcount " << datanew->GetNumberOfPoints() << std::endl;
	std::cout << "facecount: " << datanew->GetNumberOfCells() << std::endl;
}

void markReducedBloodCell()
{
	int startTs = 6, endTs = 32;

	int i = startTs;
	stringstream ss;
	ss << setw(4) << setfill('0') << i;
	string s = ss.str();

	string inputFileName = "D:/Data/Lin/reducedBloodCell/reduced-rbcs-" + s + ".vtp";

	vtkSmartPointer<vtkPolyData> data = vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkXMLPolyDataReader> reader = vtkSmartPointer<vtkXMLPolyDataReader>::New();
	reader->SetFileName(inputFileName.c_str());
	data = reader->GetOutput();
	reader->Update();

	//assume this filter will not change original points and cells
	vtkSmartPointer<vtkPolyDataConnectivityFilter> connectivityFilter =
		vtkSmartPointer<vtkPolyDataConnectivityFilter>::New();
	connectivityFilter->SetInputData(data);
	connectivityFilter->SetExtractionModeToAllRegions();
	connectivityFilter->ColorRegionsOn();
	connectivityFilter->Update();
	vtkSmartPointer<vtkPolyData> res = connectivityFilter->GetOutput();

	vtkSmartPointer<vtkXMLPolyDataWriter> writer4 = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
	writer4->SetFileName(("D:/Data/Lin/reducedBloodCell/marked-reduced-rbcs-" + s + ".vtp").c_str());
	writer4->SetInputData(res);
	writer4->Write();



	vtkIntArray *arrayRegionId = (vtkIntArray *)((res->GetPointData())->GetArray("RegionId"));
	int n = arrayRegionId->GetNumberOfTuples();
	int range[2];
	arrayRegionId->GetValueRange(range);
	int nRegion = range[1] - range[0] + 1;
	if (range[0] != 0){	//the function relies on the assumption that the result of vtkPolyDataConnectivityFilter is using consecutive ids starting from 0.
		std::cout << "not consecutive ids!!" << std::endl;
		exit(0);
	}

	//no need to check if points belonging to the same region are consecutive or not, since it has been checked in reduceBloodCell()

	vector<int> count(nRegion, 0);
	vector<double3> posSum(nRegion, make_double3(0, 0, 0));
	vector<float3> minPos(nRegion, make_float3(999999, 999999, 999999));
	vector<float3> maxPos(nRegion, make_float3(-999999, -999999, -999999));

	for (int i = 0; i < n; i++){
		long long d = arrayRegionId->GetValue(i);
		double * coord = data->GetPoint(i);
		posSum[d] = make_double3(posSum[d].x + coord[0], posSum[d].y + coord[1], posSum[d].z + coord[2]);
		minPos[d] = make_float3(fmin(minPos[d].x, coord[0]), fmin(minPos[d].y, coord[1]), fmin(minPos[d].z, coord[2]));
		maxPos[d] = make_float3(fmax(maxPos[d].x, coord[0]), fmax(maxPos[d].y, coord[1]), fmax(maxPos[d].z, coord[2])); 
		count[d]++;
	}
	vector<float3> posAve(nRegion);
	for (int i = 0; i < nRegion; i++){
		posAve[i] = make_float3(posSum[i].x / count[i], posSum[i].y / count[i], posSum[i].z / count[i]);
	}

	vector<float3> lastPosAve = posAve;
	vector<int> lastIds(nRegion);
	std::iota(std::begin(lastIds), std::end(lastIds), 0);

	int nextAvalableId = nRegion;

	//record tuple file
	vector<int> countFace(nRegion, 0);
	int curid = 0;
	int m = res->GetNumberOfCells();
	for (int i = 0; i < m; i++){
		int vertexId = data->GetCell(i)->GetPointId(0);
		int d = arrayRegionId->GetValue(vertexId);
		if (curid == d){
			countFace[d]++;
		}
		else if (curid + 1 == d){
			curid++;
			countFace[d]++;
		}
		else {
			cout << "not consecutive region id at face " << i << " with id " << curid << endl;
			exit(0); //not implemented currently
		}
	}
	int temp = 1;
	FILE * fp = fopen(("D:/Data/Lin/reducedBloodCell/marked-reduced-rbcs-" + s + "-polyMeshRegions.mytup").c_str(), "wb");
	fwrite(&temp, sizeof(int), 1, fp);
	fwrite(&nRegion, sizeof(int), 1, fp);
	int nc = 14;
	fwrite(&nc, sizeof(int), 1, fp);
	int startv = 0, startf = 0;
	for (int i = 0; i < nRegion; i++){
		float startvf = startv;
		float endvf = startv + count[i] - 1;
		float startff = startf;
		float endff = startf + countFace[i] - 1;
		fwrite(&(posAve[i].x), sizeof(float3), 1, fp);
		fwrite(&startff, sizeof(float), 1, fp); //range of faces of the current region
		fwrite(&endff, sizeof(float), 1, fp);
		fwrite(&startvf, sizeof(float), 1, fp); //range of vertices of the current region
		fwrite(&endvf, sizeof(float), 1, fp);
		fwrite(&(minPos[i].x), sizeof(float3), 1, fp); //for bounding box of the current region
		fwrite(&(maxPos[i].x), sizeof(float3), 1, fp);
		float tempi = i;
		fwrite(&tempi, sizeof(int), 1, fp); //id of the region
		startv = startv + count[i];
		startf = startf + countFace[i];
	}

	for (int i = startTs+1; i <= endTs; i++){
		std::cout << "processing time step " << i << std::endl;

		stringstream ss;
		ss << setw(4) << setfill('0') << i;
		string s = ss.str();

		string inputFileName = "D:/Data/Lin/reducedBloodCell/reduced-rbcs-" + s + ".vtp";

		vtkSmartPointer<vtkPolyData> data = vtkSmartPointer<vtkPolyData>::New();
		vtkSmartPointer<vtkXMLPolyDataReader> reader = vtkSmartPointer<vtkXMLPolyDataReader>::New();
		reader->SetFileName(inputFileName.c_str());
		data = reader->GetOutput();
		reader->Update();

		vtkSmartPointer<vtkPolyDataConnectivityFilter> connectivityFilter =
			vtkSmartPointer<vtkPolyDataConnectivityFilter>::New();
		connectivityFilter->SetInputData(data);
		connectivityFilter->SetExtractionModeToAllRegions();
		connectivityFilter->ColorRegionsOn();
		connectivityFilter->Update();
		vtkSmartPointer<vtkPolyData> res = connectivityFilter->GetOutput();

		vtkIntArray *arrayRegionId = (vtkIntArray *)((res->GetPointData())->GetArray("RegionId"));
		int n = arrayRegionId->GetNumberOfTuples();
		int range[2];
		arrayRegionId->GetValueRange(range);
		int nRegion = range[1] - range[0] + 1;
		if (range[0] != 0){
			std::cout << "not consecutive ids!!" << std::endl;
			exit(0);
		}
		//no need to check if points belonging to the same region are consecutive or not, since it has been checked in reduceBloodCell()
		vector<int> count(nRegion, 0);
		vector<double3> posSum(nRegion, make_double3(0, 0, 0));
		vector<float3> minPos(nRegion, make_float3(999999, 999999, 999999));
		vector<float3> maxPos(nRegion, make_float3(-999999, -999999, -999999));

		for (int i = 0; i < n; i++){
			long long d = arrayRegionId->GetValue(i);
			double * coord = data->GetPoint(i);
			posSum[d] = make_double3(posSum[d].x + coord[0], posSum[d].y + coord[1], posSum[d].z + coord[2]);
			minPos[d] = make_float3(fmin(minPos[d].x, coord[0]), fmin(minPos[d].y, coord[1]), fmin(minPos[d].z, coord[2]));
			maxPos[d] = make_float3(fmax(maxPos[d].x, coord[0]), fmax(maxPos[d].y, coord[1]), fmax(maxPos[d].z, coord[2])); 
			count[d]++;
		}
		vector<float3> posAve(nRegion);
		for (int i = 0; i < nRegion; i++){
			posAve[i] = make_float3(posSum[i].x / count[i], posSum[i].y / count[i], posSum[i].z / count[i]);
		}

		//for record tuple file
		vector<int> countFace(nRegion, 0);
		int curid = 0;
		int m = res->GetNumberOfCells();
		for (int i = 0; i < m; i++){
			int vertexId = data->GetCell(i)->GetPointId(0);
			int d = arrayRegionId->GetValue(vertexId);
			if (curid == d){
				countFace[d]++;
			}
			else if (curid + 1 == d){
				curid++;
				countFace[d]++;
			}
			else {
				cout << "not consecutive region id at face " << i << " with id " << curid << endl;
				exit(0); //not implemented currently
			}
		}



		vector<int> idChangeMap(nRegion, -1);

		int numLastIds = lastIds.size();
		vector<float> closestDisSquFromLast(numLastIds, -1);
		vector<int> idOfClosestDisSquFromLast(numLastIds, -1);

		for (int i = 0; i < nRegion; i++){
			float minDisSquare = FLT_MAX;
			int idOfMinDis;
			for (int j = 0; j < numLastIds; j++){
				float disSquare =  pow(posAve[i].x - lastPosAve[j].x, 2) + pow(posAve[i].y - lastPosAve[j].y, 2) + pow(posAve[i].z - lastPosAve[j].z, 2);
				if (disSquare < minDisSquare){
					minDisSquare = disSquare;
					idOfMinDis = j; //essentially the id is not j, but lastIds[j]. here only the index j is important
				}
			}

			if (idOfClosestDisSquFromLast[idOfMinDis] < 0){
				idOfClosestDisSquFromLast[idOfMinDis] = i;
				closestDisSquFromLast[idOfMinDis] = minDisSquare;
			}
			else{
				if (closestDisSquFromLast[idOfMinDis] > minDisSquare){
					idOfClosestDisSquFromLast[idOfMinDis] = i;
					closestDisSquFromLast[idOfMinDis] = minDisSquare;
				}
			}
		}

		for (int j = 0; j < numLastIds; j++){
			if (idOfClosestDisSquFromLast[j] >= 0){
				//id change from idOfClosestDisSquFromLast[j] -> j
				idChangeMap[idOfClosestDisSquFromLast[j]] = lastIds[j];
			}
		}
		for (int i = 0; i < nRegion; i++){
			if (idChangeMap[i] < 0){
				idChangeMap[i] = nextAvalableId;
				nextAvalableId++;
			}
		}
		for (int i = 0; i < n; i++){
			int idOri = arrayRegionId->GetValue(i);
			arrayRegionId->SetValue(i, idChangeMap[idOri]);
		}

		vtkSmartPointer<vtkXMLPolyDataWriter> writer4 = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
		writer4->SetFileName(("D:/Data/Lin/reducedBloodCell/marked-reduced-rbcs-" + s + ".vtp").c_str());
		writer4->SetInputData(res);
		writer4->Write();

		lastPosAve = posAve;
		lastIds = idChangeMap;



		
		//record tuple file
		int temp = 1;
		FILE * fp = fopen(("D:/Data/Lin/reducedBloodCell/marked-reduced-rbcs-" + s + "-polyMeshRegions.mytup").c_str(), "wb");
		fwrite(&temp, sizeof(int), 1, fp);
		fwrite(&nRegion, sizeof(int), 1, fp);
		int nc = 14;
		fwrite(&nc, sizeof(int), 1, fp);
		int startv = 0, startf = 0;
		for (int i = 0; i < nRegion; i++){
			float startvf = startv;
			float endvf = startv + count[i] - 1;
			float startff = startf;
			float endff = startf + countFace[i] - 1;
			fwrite(&(posAve[i].x), sizeof(float3), 1, fp);
			fwrite(&startff, sizeof(float), 1, fp); //range of faces of the current region
			fwrite(&endff, sizeof(float), 1, fp);
			fwrite(&startvf, sizeof(float), 1, fp); //range of vertices of the current region
			fwrite(&endvf, sizeof(float), 1, fp);
			fwrite(&(minPos[i].x), sizeof(float3), 1, fp); //for bounding box of the current region
			fwrite(&(maxPos[i].x), sizeof(float3), 1, fp);
			float tempi = idChangeMap[i];
			fwrite(&tempi, sizeof(int), 1, fp); //id of the region
			startv = startv + count[i];
			startf = startf + countFace[i];
		}
	}
}


int main(int argc, char **argv)
{
	//createSphere();
	labelPoly();

	//reduceBloodCell();
	//markReducedBloodCell();

	return 0;
}
