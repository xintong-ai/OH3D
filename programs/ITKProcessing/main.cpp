#include <memory>
#include <string>
#include <iostream>
#include <float.h>

//#include <cuda_runtime.h>
//#include <helper_cuda.h>

#include "Volume.h"
#include "RawVolumeReader.h"
#include "DataMgr.h"
#include "VecReader.h"

#include "myDefineRayCasting.h"
#include "PolyMesh.h"

#include "imageProcessing.h"
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
#include <vtkPointData.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataToImageStencil.h>
#include <vtkImageStencil.h>
#include <vtkImageStencil.h>
#include <vtkMetaImageWriter.h>
#include <vtkPLYReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkCell.h>
#include <itkConnectedComponentImageFilter.h>

using namespace std;

const int PHASES_COUNT = 1; //phase 1 is for creating cell volume; phase 2 is for creating skeletion; phase 3 is for creating bilateral volume

void computeInitCellVolume(std::shared_ptr<Volume> v, std::shared_ptr<Volume> resVolume, std::shared_ptr<RayCastingParameters> rcp)
{
	std::cout << "computing channel volume..." << std::endl;

	int3 dataSizes = v->size;

	for (int k = 0; k < dataSizes.z; k++)
	{
		for (int j = 0; j < dataSizes.y; j++)
		{
			for (int i = 0; i < dataSizes.x; i++)
			{
				int ind = k*dataSizes.y * dataSizes.x + j*dataSizes.x + i;
				if (v->values[ind] < rcp->transFuncP2){
					resVolume->values[ind] = 1;
				}
				else{
					resVolume->values[ind] = 0;
				}
			}
		}
	}

	std::cout << "finish computing channel volume..." << std::endl;
	return;
}


ImageType::Pointer createITKImageFromVolume(shared_ptr<Volume> volume)
{
	//compute skeleton volume and itk skelComponented image from the beginning
	std::cout << "computing skeletion volume..." << std::endl;
	const unsigned int numberOfPixels = volume->size.x * volume->size.y * volume->size.z;
	PixelType * localBuffer = new PixelType[numberOfPixels];
	for (int i = 0; i < numberOfPixels; i++){
		localBuffer[i] = volume->values[i];
	}

	///////////////import to itk image
	const bool importImageFilterWillOwnTheBuffer = false; //probably can change to true for faster speed?
	typedef itk::BinaryThinningImageFilter3D< ImageType, ImageType > ThinningFilterType;
	ImportFilterType::Pointer importFilter = ImportFilterType::New();

	ImageType::SizeType imsize;
	imsize[0] = volume->size.x;
	imsize[1] = volume->size.y;
	imsize[2] = volume->size.z;
	ImportFilterType::IndexType start;
	start.Fill(0);
	ImportFilterType::RegionType region;
	region.SetIndex(start);
	region.SetSize(imsize);
	importFilter->SetRegion(region);
	const itk::SpacePrecisionType origin[3] = { imsize[0], imsize[1], imsize[2] };
	importFilter->SetOrigin(origin);
	const itk::SpacePrecisionType _spacing[3] = { volume->spacing.x, volume->spacing.y, volume->spacing.z };
	importFilter->SetSpacing(_spacing);
	importFilter->SetImportPointer(localBuffer, volume->size.x *  volume->size.y *  volume->size.z, importImageFilterWillOwnTheBuffer);
	importFilter->Update();

	return importFilter->GetOutput();
}


void vtkPolyDataShift(vtkSmartPointer<vtkPolyData> originalMesh, float3 shift)
{
	int vertexcount = originalMesh->GetNumberOfPoints();

	for (int i = 0; i < vertexcount; i++) {
		double coord[3];
		originalMesh->GetPoint(i, coord);
		coord[0] += shift.x;
		coord[1] += shift.y;
		coord[2] += shift.z;
		originalMesh->GetPoints()->SetPoint(i, coord);
	}
}


//the purpose of shift is to leave enough margin from (0,0,0) to the lower bound of vertex coordinates
//the purpose of spacing is to make its channel volume suitable to have a spacing (1,1,1), for the convenience of future processing
void vtkPolyDataShiftAndRespacing(vtkSmartPointer<vtkPolyData> originalMesh, float3 shift, float3 spacing)
{
	int vertexcount = originalMesh->GetNumberOfPoints();

	for (int i = 0; i < vertexcount; i++) {
		double coord[3];
		originalMesh->GetPoint(i, coord);
		coord[0] = coord[0] / spacing.x + shift.x;
		coord[1] = coord[1] / spacing.y + shift.y;
		coord[2] = coord[2] / spacing.z + shift.z;
		originalMesh->GetPoints()->SetPoint(i, coord);
	}
}


void processVolumeData()
{
	//reading data
	int3 dims;
	float3 spacing;

	std::shared_ptr<DataMgr> dataMgr;
	dataMgr = std::make_shared<DataMgr>();
	//const std::string dataPath = dataMgr->GetConfig("VOLUME_DATA_PATH")
	dataPath = dataMgr->GetConfig("VOLUME_DATA_PATH");
	std::shared_ptr<RayCastingParameters> rcp = std::make_shared<RayCastingParameters>();
	std::string subfolder;

	Volume::rawFileInfo(dataPath, dims, spacing, rcp, subfolder);
	DataType volDataType = RawVolumeReader::dtUint16;
	bool labelFromFile;
	RawVolumeReader::rawFileReadingInfo(dataPath, volDataType, labelFromFile);

	shared_ptr<Volume> inputVolume = std::make_shared<Volume>(true);
	if (std::string(dataPath).find(".vec") != std::string::npos){
		std::shared_ptr<VecReader> reader;
		reader = std::make_shared<VecReader>(dataPath.c_str());
		reader->OutputToVolumeByNormalizedVecMag(inputVolume);
		reader.reset();
	}
	else{
		std::shared_ptr<RawVolumeReader> reader;
		reader = std::make_shared<RawVolumeReader>(dataPath.c_str(), dims, volDataType);
		reader->OutputToVolumeByNormalizedValue(inputVolume);
		reader.reset();
	}
	inputVolume->spacing = spacing;

	setParameter();
	//computing cell volume

	shared_ptr<Volume> initCellVolume = std::make_shared<Volume>();
	initCellVolume->setSize(inputVolume->size);
	initCellVolume->dataOrigin = inputVolume->dataOrigin;
	initCellVolume->spacing = inputVolume->spacing;
	computeInitCellVolume(inputVolume, initCellVolume, rcp);

	ImageType::Pointer initCellImg = createITKImageFromVolume(initCellVolume);

	ImageType::Pointer connectedImg;

	int maxComponentMark;
	refineCellVolume(initCellImg, dims, spacing, connectedImg, maxComponentMark);

	if (PHASES_COUNT == 1){
		return;
	}


	//compute skeletons
	ImageType::Pointer skelComponentedImg;
	std::cout << "finish computing skeletion volume..." << std::endl;
	skelComputing(connectedImg, initCellImg, dims, spacing, skelComponentedImg, maxComponentMark);

	std::vector<std::vector<float3>> viewArrays;
	//sample the skeleton to set the views
	findViews(skelComponentedImg, maxComponentMark, dims, spacing, viewArrays);

	if (PHASES_COUNT == 2){
		return;
	}


	//compute the bilateralVolume
	float* bilateralVolumeRes = 0;
	inputVolume->computeBilateralFiltering(bilateralVolumeRes, 2, 0.2);
	FILE * fp = fopen("bilat.raw", "wb");
	fwrite(bilateralVolumeRes, sizeof(float), inputVolume->size.x*inputVolume->size.y*inputVolume->size.z, fp);
	fclose(fp);
	delete[]bilateralVolumeRes;
}

//this function has not been throughly tested
inline float disToTri(float3 p, float3 p1, float3 p2, float3 p3, float thr)
{
	float3 e12 = p2 - p1;
	float3 e23 = p3 - p2;
	float3 e31 = p1 - p3;

	float3 n = normalize(cross(e12, - e31));
	float disToPlane = dot(p - p1, n);
	if (abs(disToPlane) >= thr){
		return thr + 1; //no need further computation
	}
	float3 proj = p - n*disToPlane;

	bool isInside = false;
	if (dot(cross(e12, -e31), cross(e12, proj - p1)) >= 0){
		if (dot(cross(e23, -e12), cross(e23, proj - p2)) >= 0){
			if (dot(cross(e31, -e23), cross(e31, proj - p3)) >= 0){
				isInside = true;
			}
		}
	}
	if (isInside){
		return abs(disToPlane);
	}
	float disInPlaneSqu = min(min(dot(proj - p1, proj - p1), dot(proj - p2, proj - p2)), dot(proj - p3, proj - p3));
	float d = dot(proj - p1, e12);
	if (d > 0 && d < dot(e12, e12)){
		float projL = d / length(e12);
		disInPlaneSqu = min(disInPlaneSqu, dot(proj - p1, proj - p1) - projL*projL);
	}
	d = dot(proj - p2, e23);
	if (d > 0 && d < dot(e23, e23)){
		float projL = d / length(e23);
		disInPlaneSqu = min(disInPlaneSqu, dot(proj - p2, proj - p2) - projL*projL);
	}
	d = dot(proj - p3, e31);
	if (d > 0 && d < dot(e31, e31)){
		float projL = d / length(e31);
		disInPlaneSqu = min(disInPlaneSqu, dot(proj - p3, proj - p3) - projL*projL);
	}
	return sqrt(disInPlaneSqu + disToPlane*disToPlane);
}

void computeDistanceMap(itk::Image< float, 3 >::Pointer image, vtkSmartPointer<vtkPolyData> data)
{
	//in this function, only process vtkPolyData that has been shifted and respaced
	float3 originf3 = make_float3(0, 0, 0);
	float3 spacing = make_float3(1.0, 1.0, 1.0);


	int facecount = data->GetNumberOfCells();

	ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();

	for (int i = 0; i < facecount; i++) {
		if (data->GetCell(i)->GetNumberOfPoints() != 3){
			std::cout << "readed poly data contains non-triangles. the current program cannot handle" << std::endl;
			exit(0);
		}

		int i1 = data->GetCell(i)->GetPointId(0);
		int i2 = data->GetCell(i)->GetPointId(1);
		int i3 = data->GetCell(i)->GetPointId(2);

		double coord[3];
		data->GetPoint(i1, coord);
		float3 p1 = make_float3(coord[0], coord[1], coord[2]);
		data->GetPoint(i2, coord);
		float3 p2 = make_float3(coord[0], coord[1], coord[2]);
		data->GetPoint(i3, coord);
		float3 p3 = make_float3(coord[0], coord[1], coord[2]);

		float3 v1 = p1 / spacing + originf3;
		float3 v2 = p2 / spacing + originf3;
		float3 v3 = p3 / spacing + originf3;

		float bbMargin = 3;
		int xstart = max(min(min(v1.x, v2.x), v3.x) - bbMargin, 0);
		int xend = min(ceil(max(max(v1.x, v2.x), v3.x) + bbMargin), size[0] - 1);
		int ystart = max(min(min(v1.y, v2.y), v3.y) - bbMargin, 0);
		int yend = min(ceil(max(max(v1.y, v2.y), v3.y) + bbMargin), size[1] - 1);
		int zstart = max(min(min(v1.z, v2.z), v3.z) - bbMargin, 0);
		int zend = min(ceil(max(max(v1.z, v2.z), v3.z) + bbMargin), size[2] - 1);

		for (unsigned int z = zstart; z <= zend; z++)
		{
			for (unsigned int y = ystart; y <= yend; y++)
			{
				for (unsigned int x = xstart; x <= xend; x++)
				{
					ImageType::IndexType pixelIndex;
					pixelIndex[0] = x;
					pixelIndex[1] = y;
					pixelIndex[2] = z;

					float cur = image->GetPixel(pixelIndex);
					float v = disToTri(make_float3(x, y, z) * spacing + originf3, p1, p2, p3, cur);
					if (v < cur){
						image->SetPixel(pixelIndex, v);
					}
				}
			}
		}
	}
}


void thresholdImage(itk::Image< float, 3 >::Pointer image, float disThr)
{
	ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
	for (unsigned int z = 0; z < size[2]; z++)
	{
		for (unsigned int y = 0; y < size[1]; y++)
		{
			for (unsigned int x = 0; x< size[0]; x++)
			{
				ImageType::IndexType pixelIndex;
				pixelIndex[0] = x;
				pixelIndex[1] = y;
				pixelIndex[2] = z;

				if (image->GetPixel(pixelIndex) <= disThr){
					image->SetPixel(pixelIndex, 0);
				}
				else{
					image->SetPixel(pixelIndex, 1);
				}
			}
		}
	}
}


//the purpose for this function is, using a larger threshold for regions outside the polymesh, therefore the mesh will be deformed earlier when the eye is approaching
void thresholdImageExteriorExtension(itk::Image< float, 3 >::Pointer image, float disThr, float disThrExt) //specifically for moortgat's data
{
	ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();

	itk::Image< int, 3 >::Pointer labelImage = itk::Image< int, 3 >::New();
	itk::Index<3> start; start.Fill(0);
	ImageType::RegionType region(start, size);
	labelImage->SetRegions(region);
	double spacingDouble[3] = { 1,1,1 };
	double origin[3] = { 0, 0, 0 };
	labelImage->SetOrigin(origin);
	labelImage->SetSpacing(spacingDouble);
	labelImage->Allocate();
	labelImage->FillBuffer(0);

	for (unsigned int z = 0; z < size[2]; z++)
	{
		for (unsigned int y = 0; y < size[1]; y++)
		{
			for (unsigned int x = 0; x< size[0]; x++)
			{
				ImageType::IndexType pixelIndex;
				pixelIndex[0] = x;
				pixelIndex[1] = y;
				pixelIndex[2] = z;

				if (image->GetPixel(pixelIndex) <= disThr){
					labelImage->SetPixel(pixelIndex, 0);
				}
				else if((z == 84 && x <= 64 && y <= 64 && x >= 2 && y >= 2)
					|| (z == 2 && x <= 64 && y <= 64 && x >= 2 && y >= 2)
					|| (x == 2 && z <= 84 && y <= 64 && z >= 2 && y >= 2)
					|| (x == 64 && z <= 84 && y <= 64 && z >= 6 && y >= 2)	//note here z >= 6 leaves some leak connection
					|| (y == 2 && z <= 84 && x <= 64 && z >= 2 && x >= 2)
					|| (y == 64 && z <= 84 && x <= 64 && z >= 2 && x >= 2)){

					labelImage->SetPixel(pixelIndex, 0);
				}
				else{
					labelImage->SetPixel(pixelIndex, 1);
				}
			}
		}
	}

	typedef itk::ConnectedComponentImageFilter <itk::Image< int, 3 >, itk::Image< int, 3 > >
		ConnectedComponentImageFilterType;
	ConnectedComponentImageFilterType::Pointer connected =
		ConnectedComponentImageFilterType::New();
	connected->SetInput(labelImage);
	connected->Update();
	labelImage = connected->GetOutput();

	ImageType::IndexType pixelIndex;
	pixelIndex[0] = 0;
	pixelIndex[1] = 0;
	pixelIndex[2] = 0;
	int exteriorLabel = labelImage->GetPixel(pixelIndex);


	for (unsigned int z = 0; z < size[2]; z++)
	{
		for (unsigned int y = 0; y < size[1]; y++)
		{
			for (unsigned int x = 0; x< size[0]; x++)
			{
				ImageType::IndexType pixelIndex;
				pixelIndex[0] = x;
				pixelIndex[1] = y;
				pixelIndex[2] = z;

				if ((labelImage->GetPixel(pixelIndex) == exteriorLabel)
					|| (z == 84 && x <= 64 && y <= 64 && x >= 2 && y >= 2)	//the boundary wall
					|| (z == 2 && x <= 64 && y <= 64 && x >= 2 && y >= 2)
					|| (x == 2 && z <= 84 && y <= 64 && z >= 2 && y >= 2)
					|| (x == 64 && z <= 84 && y <= 64 && z >= 2 && y >= 2)
					|| (y == 2 && z <= 84 && x <= 64 && z >= 2 && x >= 2)
					|| (y == 64 && z <= 84 && x <= 64 && z >= 2 && x >= 2)){

					if (image->GetPixel(pixelIndex) <= disThrExt){
						image->SetPixel(pixelIndex, 0);
					}
					else{
						image->SetPixel(pixelIndex, 1);
					}
				}
				else{
					if (image->GetPixel(pixelIndex) <= disThr){
						image->SetPixel(pixelIndex, 0);
					}
					else{
						image->SetPixel(pixelIndex, 1);
					}
				}				
			}
		}
	}
}

void processSurfaceData()
{
	//this process is implemented in vtkImplicitModeller class
	//however, the filter class is not well implemented with user defined origin/spacing configurations, and post-porcessing needs many other classes which are also not very reliable
	//therefore implement by coding here

	vtkSmartPointer<vtkPolyData> data;

	std::shared_ptr<DataMgr> dataMgr;
	dataMgr = std::make_shared<DataMgr>();
	dataPath = dataMgr->GetConfig("POLY_DATA_PATH");
	vtkSmartPointer<vtkXMLPolyDataReader> reader =
		vtkSmartPointer<vtkXMLPolyDataReader>::New();
	reader->SetFileName(dataPath.c_str());
	reader->Update();
	data = reader->GetOutput();

	float disThr;
	float3 shift;
	int3 dims;
	float3 spacing;
	string subfolder;
	PolyMesh::dataParameters(dataPath, dims, spacing, disThr, shift, subfolder);
	std::cout << "shifted "<<shift.x<<" "<<shift.y<<" "<<shift.z << std::endl;
	std::cout << "spacing " << spacing.x << " " << spacing.y << " " << spacing.z << std::endl;

	//vtkPolyDataShift(data, shift);
	vtkPolyDataShiftAndRespacing(data, shift, spacing);
	spacing = make_float3(1, 1, 1);

	vtkSmartPointer<vtkXMLPolyDataWriter> writer4 = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
	writer4->SetFileName("shifted.vtp");
	writer4->SetInputData(data);
	writer4->Write();

	typedef float PixelType;
	typedef itk::Image< PixelType, 3 > ImageType;
	ImageType::Pointer image = ImageType::New();
	itk::Index<3> start; start.Fill(0);
	itk::Size<3> size;
	size[0] = dims.x;
	size[1] = dims.y;
	size[2] = dims.z;
	ImageType::RegionType region(start, size);
	image->SetRegions(region);
	double spacingDouble[3] = { spacing.x, spacing.y, spacing.z };
	double origin[3] = { 0, 0, 0 };
	float3 originf3 = make_float3(origin[0], origin[1], origin[2]);
	image->SetOrigin(origin);
	image->SetSpacing(spacingDouble);
	image->Allocate();
	image->FillBuffer(0);

	// Make an empty data
	for (unsigned int z = 0; z < size[2]; z++)
	{
		for (unsigned int y = 0; y < size[1]; y++)
		{
			for (unsigned int x = 0; x< size[0]; x++)
			{
				ImageType::IndexType pixelIndex;
				pixelIndex[0] = x;
				pixelIndex[1] = y;
				pixelIndex[2] = z;

				image->SetPixel(pixelIndex, 10);
			}
		}
	}
	computeDistanceMap(image, data);
	/*
	int facecount = data->GetNumberOfCells();

	for (int i = 0; i < facecount; i++) {
		if (data->GetCell(i)->GetNumberOfPoints() != 3){
			std::cout << "readed poly data contains non-triangles. the current program cannot handle" << std::endl;
			exit(0);
		}

		int i1 = data->GetCell(i)->GetPointId(0);
		int i2 = data->GetCell(i)->GetPointId(1);
		int i3 = data->GetCell(i)->GetPointId(2);

		double coord[3];
		data->GetPoint(i1, coord);
		float3 p1 = make_float3(coord[0], coord[1], coord[2]);
		data->GetPoint(i2, coord);
		float3 p2 = make_float3(coord[0], coord[1], coord[2]);
		data->GetPoint(i3, coord);
		float3 p3 = make_float3(coord[0], coord[1], coord[2]);
		
		float3 v1 = p1 / spacing + originf3;
		float3 v2 = p2 / spacing + originf3;
		float3 v3 = p3 / spacing + originf3;

		float bbMargin = 3;
		int xstart = max(min(min(v1.x, v2.x), v3.x) - bbMargin, 0);
		int xend = min(ceil(max(max(v1.x, v2.x), v3.x) + bbMargin), size[0] - 1);
		int ystart = max(min(min(v1.y, v2.y), v3.y) - bbMargin, 0);
		int yend = min(ceil(max(max(v1.y, v2.y), v3.y) + bbMargin), size[1] - 1);
		int zstart = max(min(min(v1.z, v2.z), v3.z) - bbMargin, 0);
		int zend = min(ceil(max(max(v1.z, v2.z), v3.z) + bbMargin), size[2] - 1);

		for (unsigned int z = zstart; z <= zend; z++)
		{
			for (unsigned int y = ystart; y <= yend; y++)
			{
				for (unsigned int x = xstart; x <= xend; x++)
				{
					ImageType::IndexType pixelIndex;
					pixelIndex[0] = x;
					pixelIndex[1] = y;
					pixelIndex[2] = z;

					float cur = image->GetPixel(pixelIndex);
					float v = disToTri(make_float3(x, y, z) * spacing + originf3, p1, p2, p3, cur);
					if (v < cur){
						image->SetPixel(pixelIndex, v);
					}
				}
			}
		}
	}
	*/

	thresholdImage(image, disThr);

	ImageType::SizeType inputSize = image->GetLargestPossibleRegion().GetSize();
	std::cout << "Input size: " << inputSize << std::endl;

	typedef  itk::ImageFileWriter<ImageType> WriterType;
	std::cout << "Writing output... " << std::endl;
	WriterType::Pointer outputWriter = WriterType::New();
	outputWriter->SetFileName("cleanedChannel.mhd");
	outputWriter->SetInput(image);
	outputWriter->Update();
}


void processSurfaceMultiData()
{
	//similar to processSurfaceData(), but process multiple datasets into one cell volume, and shift and resample each of them
	//this function does not use the config.txt

	vtkSmartPointer<vtkPolyData> data[2];
	std::string paths[2];
	std::string folderpath = "D:/Data/moortgat/";
	paths[0] = folderpath + "sand60_067_xw2_iso0.0005.vtp";
	paths[1] = folderpath + "sand60_067_xw2_iso0.0012.vtp";

	//assume the following parameters is the same for all datasets in data[]
	float disThr;
	float3 shift;
	int3 dims;
	float3 spacing;
	string subfolder;
	PolyMesh::dataParameters(paths[0], dims, spacing, disThr, shift, subfolder);

	for (int i = 0; i < 2; i++){
		vtkSmartPointer<vtkXMLPolyDataReader> reader = vtkSmartPointer<vtkXMLPolyDataReader>::New();
		reader->SetFileName(paths[i].c_str());
		reader->Update();
		data[i] = reader->GetOutput();

		std::cout << "shifted " << shift.x << " " << shift.y << " " << shift.z << std::endl;
		std::cout << "spacing " << spacing.x << " " << spacing.y << " " << spacing.z << std::endl;

		//vtkPolyDataShift(data, shift);
		vtkPolyDataShiftAndRespacing(data[i], shift, spacing);

		vtkSmartPointer<vtkXMLPolyDataWriter> writer4 = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
		writer4->SetFileName((string("shifted") + to_string(i) + string(".vtp")).c_str());
		writer4->SetInputData(data[i]);
		writer4->Write();
	}

	spacing = make_float3(1, 1, 1);


	typedef float PixelType;
	typedef itk::Image< PixelType, 3 > ImageType;
	ImageType::Pointer image = ImageType::New();
	itk::Index<3> start; start.Fill(0);
	itk::Size<3> size;
	size[0] = dims.x;
	size[1] = dims.y;
	size[2] = dims.z;
	ImageType::RegionType region(start, size);
	image->SetRegions(region);
	double spacingDouble[3] = { spacing.x, spacing.y, spacing.z };
	double origin[3] = { 0, 0, 0 };
	float3 originf3 = make_float3(origin[0], origin[1], origin[2]);
	image->SetOrigin(origin);
	image->SetSpacing(spacingDouble);
	image->Allocate();
	image->FillBuffer(0);

	// Make an empty data
	for (unsigned int z = 0; z < size[2]; z++)
	{
		for (unsigned int y = 0; y < size[1]; y++)
		{
			for (unsigned int x = 0; x< size[0]; x++)
			{
				ImageType::IndexType pixelIndex;
				pixelIndex[0] = x;
				pixelIndex[1] = y;
				pixelIndex[2] = z;

				image->SetPixel(pixelIndex, 10);
			}
		}
	}

	for (int i = 0; i < 2; i++){
		computeDistanceMap(image, data[i]);
	}

	//thresholdImage(image, disThr);
	thresholdImageExteriorExtension(image, disThr, 2 * disThr);

	ImageType::SizeType inputSize = image->GetLargestPossibleRegion().GetSize();
	std::cout << "Input size: " << inputSize << std::endl;

	typedef  itk::ImageFileWriter<ImageType> WriterType;
	std::cout << "Writing output... " << std::endl;
	WriterType::Pointer outputWriter = WriterType::New();
	outputWriter->SetFileName("cleanedChannel.mhd");
	outputWriter->SetInput(image);
	outputWriter->Update();
}

void processParticleMeshData()
{
	std::shared_ptr<DataMgr> dataMgr;
	dataMgr = std::make_shared<DataMgr>();
	dataPath = dataMgr->GetConfig("POLY_DATA_PATH");

	double spacing[3] = { 1, 1, 1 };
	double origin[3] = { 0, 0, 0 };

	float3 shift = make_float3(5, 3, 0);
	//float3 shift = make_float3(0, 0, 0);

	int dim[3] = { 65 + shift.x, 225 + shift.y, 161 + shift.z };

	vtkSmartPointer<vtkPLYReader> reader =
		vtkSmartPointer<vtkPLYReader>::New();
	reader->SetFileName(dataPath.c_str());
	vtkSmartPointer<vtkPolyData> pd = reader->GetOutput();
	reader->Update();

	vtkSmartPointer<vtkPolyData> originalMesh;
	originalMesh = reader->GetOutput();

	vtkPolyDataShift(originalMesh, shift);

	//vtkSmartPointer<vtkPolyDataWriter> writer4 = vtkSmartPointer<vtkPolyDataWriter>::New();
	//writer4->SetFileName("shifted.vtk");
	//writer4->SetInputData(originalMesh);
	//writer4->Write();

	vtkSmartPointer<vtkImageData> whiteImage =
		vtkSmartPointer<vtkImageData>::New();
	whiteImage->SetSpacing(spacing);
	whiteImage->SetDimensions(dim);
	whiteImage->SetOrigin(origin);
#if VTK_MAJOR_VERSION <= 5
	whiteImage->SetScalarTypeToUnsignedChar();
	whiteImage->AllocateScalars();
#else
	whiteImage->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
#endif
	// fill the image with foreground voxels:
	unsigned char inval = 0;
	unsigned char outval = 1;
	vtkIdType count = whiteImage->GetNumberOfPoints();
	for (vtkIdType i = 0; i < count; ++i)
	{
		whiteImage->GetPointData()->GetScalars()->SetTuple1(i, inval);
	}


	// polygonal data --> image stencil:
	vtkSmartPointer<vtkPolyDataToImageStencil> pol2stenc =
		vtkSmartPointer<vtkPolyDataToImageStencil>::New();
#if VTK_MAJOR_VERSION <= 5
	pol2stenc->SetInput(pd);
#else
	//pol2stenc->SetInputData(pd);
	pol2stenc->SetInputData(originalMesh);
#endif
	pol2stenc->SetOutputOrigin(origin);
	pol2stenc->SetOutputSpacing(spacing);
	pol2stenc->SetOutputWholeExtent(whiteImage->GetExtent());
	pol2stenc->Update();


	// cut the corresponding white image and set the background:
	vtkSmartPointer<vtkImageStencil> imgstenc =
		vtkSmartPointer<vtkImageStencil>::New();
#if VTK_MAJOR_VERSION <= 5
	imgstenc->SetInput(whiteImage);
	imgstenc->SetStencil(pol2stenc->GetOutput());
#else
	imgstenc->SetInputData(whiteImage);
	imgstenc->SetStencilConnection(pol2stenc->GetOutputPort());
#endif
	imgstenc->ReverseStencilOff();
	imgstenc->SetBackgroundValue(outval);
	imgstenc->Update();


	vtkSmartPointer<vtkImageData> currentImage = imgstenc->GetOutput();

	vtkSmartPointer<vtkMetaImageWriter> writer =
		vtkSmartPointer<vtkMetaImageWriter>::New();
	writer->SetFileName("cleanedChannel.mhd");
	writer->SetRAWFileName("cleanedChannel.raw");
	writer->SetCompression(false);
#if VTK_MAJOR_VERSION <= 5
	writer->SetInput(imgstenc->GetOutput());
#else
	writer->SetInputData(currentImage);
#endif
	writer->Write();

	return;
}

int main(int argc, char **argv)
{
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);

	sdkStartTimer(&timer);

	//processVolumeData();
	//processSurfaceData();
	processSurfaceMultiData();

	//processParticleMeshData();

	sdkStopTimer(&timer);

	float timeCost = sdkGetAverageTimerValue(&timer) / 1000.f;
	std::cout << "time cost for computing the preprocessing: " << timeCost << std::endl;

	return 1;

}
