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

void processSurfaceData()
{
	vtkSmartPointer<vtkPolyData> inputPolyData;

	std::shared_ptr<DataMgr> dataMgr;
	dataMgr = std::make_shared<DataMgr>();
	dataPath = dataMgr->GetConfig("POLY_DATA_PATH");
	vtkSmartPointer<vtkXMLPolyDataReader> reader =
		vtkSmartPointer<vtkXMLPolyDataReader>::New();
	reader->SetFileName(dataPath.c_str());
	reader->Update();
	inputPolyData = reader->GetOutput();

	float disThr = 2;
	
	
	//float3 shift = make_float3(ceil(disThr) + 1, ceil(disThr) + 1, ceil(disThr) + 1);//+1 for more margin
	//vtkPolyDataShift(inputPolyData, shift);

	vtkSmartPointer<vtkImplicitModeller> implicitModeller =
		vtkSmartPointer<vtkImplicitModeller>::New();

	//double bounds[6];
	//inputPolyData->GetBounds(bounds);
	//double xrange = bounds[1] - bounds[0] + 1, yrange = bounds[3] - bounds[2] + 1, zrange = bounds[5] - bounds[4] + 1;
	//double minRange = min(min(xrange, yrange), zrange);
	//if (minRange < 50){
	//	double scale = 50 / minRange;
	//	xrange *= scale;		
	//	yrange *= scale;
	//	zrange *= scale;
	//}
	implicitModeller->SetSampleDimensions(68,68,68);
#if VTK_MAJOR_VERSION <= 5
	implicitModeller->SetInput(inputPolyData);
#else
	implicitModeller->SetInputData(inputPolyData);
#endif

	implicitModeller->AdjustBoundsOn();
	implicitModeller->SetAdjustDistance(.1); // Adjust by 10%

	implicitModeller->SetMaximumDistance(.2);


	implicitModeller->Update();

	vtkSmartPointer<vtkXMLImageDataWriter> writer =
		vtkSmartPointer<vtkXMLImageDataWriter>::New();
	writer->SetFileName("cleanedChannelpre.vti");
#if VTK_MAJOR_VERSION <= 5
	writer->SetInput(implicitModeller->GetOutput());
#else
	writer->SetInputData(implicitModeller->GetOutput());
#endif
	writer->Write();
	//vtkSmartPointer<vtkImageResample> imgResampler =
	//	vtkSmartPointer<vtkImageResample>::New();
	//imgResampler->SetInputData(implicitModeller->GetOutput());
	//double spacing[3] = { 1, 1, 1 };
	////imgResampler->SetOutputSpacing(spacing);
	//imgResampler->SetDimensionality(3);
	//imgResampler->SetOutputSpacing(1.0, 1.0, 1.0);
	//imgResampler->SetOutputOrigin(0.0, 0.0, 0.0);
	//imgResampler->InterpolateOn();
	//imgResampler->SetAxisMagnificationFactor(0, 1.2);
	//imgResampler->SetAxisMagnificationFactor(1, 1.2);
	//imgResampler->SetAxisMagnificationFactor(2, 1.2);
	//imgResampler->Update();
	//vtkSmartPointer<vtkImageData> img = imgResampler->GetOutput();
	vtkSmartPointer<vtkImageData> img = implicitModeller->GetOutput();

	int dims[3];
	img->GetDimensions(dims);
	std::cout << "output dims: " << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;
	////thresholding
	//for (int k = 0; k < dims[2]; k++){
	//	for (int j = 0; j < dims[1]; j++){
	//		for (int i = 0; i < dims[0]; i++){
	//			float v = *(static_cast<float*>(img->GetScalarPointer(i,j,k)));
	//			if (abs(v)>2){
	//				*(static_cast<float*>(img->GetScalarPointer(i, j, k))) = 1;
	//			}
	//			else{
	//				*(static_cast<float*>(img->GetScalarPointer(i, j, k))) = 0;
	//			}
	//		}
	//	}
	//}

	//for some unknown reason, vtkImageResample cannot work correctly. therefore use itk 
	ImageType::Pointer image = ImageType::New();
	itk::Index<3> start; start.Fill(0);
	itk::Size<3> size;
	size[0] = dims[0];
	size[1] = dims[1];
	size[2] = dims[2];
	ImageType::RegionType region(start, size);
	image->SetRegions(region);
	double spacing[3];
	img->GetSpacing(spacing);
	double origin[3];
	img->GetOrigin(origin);
	image->SetOrigin(origin);
	image->SetSpacing(spacing);
	image->Allocate();
	image->FillBuffer(0);

	// Make a white square
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
				float v = *(static_cast<float*>(img->GetScalarPointer(x, y, z)));
				if (abs(v)>disThr){
					image->SetPixel(pixelIndex, 1);
				}
				else{
					image->SetPixel(pixelIndex, 0);
				}
			}
		}
	}
	ImageType::SizeType inputSize = image->GetLargestPossibleRegion().GetSize();
	std::cout << "Input size: " << inputSize << std::endl;

	// Resize
	ImageType::SizeType outputSize;
	outputSize.Fill(68);
	ImageType::SpacingType outputSpacing;
	outputSpacing[0] = 1.0;
	outputSpacing[1] = 1.0;
	outputSpacing[2] = 1.0;
	double outputOrigin[3] = { 0, 0, 0 };


	typedef itk::IdentityTransform<double, 2> TransformType;
	typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
	ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();
	resample->SetInput(image);
	resample->SetSize(outputSize);
	resample->SetOutputSpacing(outputSpacing);
	resample->SetOutputOrigin(outputOrigin);
	//resample->SetTransform(TransformType::New());
	resample->UpdateLargestPossibleRegion();

	ImageType::Pointer output = resample->GetOutput();

	std::cout << "Output size: " << output->GetLargestPossibleRegion().GetSize() << std::endl;

	typedef  itk::ImageFileWriter<ImageType> WriterType;
	std::cout << "Writing output... " << std::endl;
	WriterType::Pointer outputWriter = WriterType::New();
	outputWriter->SetFileName("cleanedChannel.hdr");
	outputWriter->SetInput(output);
	outputWriter->Update();



}

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

	//locate to see if proj is inside the tri
	bool isInside = false;
	if (dot(cross(e12, -e31), cross(e12, proj - p1)) >= 0){
		float3 v12 = cross(n, e12);
		float d = dot(proj - p1, v12);
		if (d >= 0 && d <= dot(v12, -e31)){
			if (dot(cross(e23, -e12), cross(e23, proj - p2)) >= 0){
				float3 v23 = cross(n, e23);
				float d = dot(proj - p2, v23);
				if (d >= 0 && d <= dot(v23, -e12)){
					if (dot(cross(e31, -e23), cross(e31, proj - p3)) >= 0){
						float3 v31 = cross(n, e31);
						float d = dot(proj - p3, v31);
						if (d >= 0 && d <= dot(v31, -e23)){
							isInside = true;
						}
					}
				}
			}
		}
	}
	if (isInside){
		return abs(disToPlane);
	}
	float disInPlaneSqu = min(min(dot(proj - p1, proj - p1), dot(proj - p2, proj - p2)), dot(proj - p3, proj - p3));
	if (disInPlaneSqu > thr*thr){
		return thr + 1;
	}
	float d = dot(proj - p1, e12);
	if (d > 0 && d < dot(e12, e12)){
		float d2 = d / length(e12);
		disInPlaneSqu = min(disInPlaneSqu, dot(proj - p1, proj - p1) - d2*d2);
	}
	d = dot(proj - p2, e23);
	if (d > 0 && d < dot(e23, e23)){
		float d2 = d / length(e23);
		disInPlaneSqu = min(disInPlaneSqu, dot(proj - p2, proj - p2) - d2*d2);
	}
	d = dot(proj - p3, e31);
	if (d > 0 && d < dot(e31, e31)){
		float d2 = d / length(e31);
		disInPlaneSqu = min(disInPlaneSqu, dot(proj - p3, proj- p3) - d2*d2);
	}
	return sqrt(disInPlaneSqu + disToPlane*disToPlane);
}

void processSurfaceData3()
{
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
	
	vtkPolyDataShift(data, shift);

	vtkSmartPointer<vtkPolyDataWriter> writer4 = vtkSmartPointer<vtkPolyDataWriter>::New();
	writer4->SetFileName("shifted.vtk");
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
	image->SetOrigin(origin);
	image->SetSpacing(spacingDouble);
	image->Allocate();
	image->FillBuffer(0);

	// Make a white square
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

	int facecount = data->GetNumberOfCells();

	for (int i = 0; i < facecount; i++) {
		if (data->GetCell(i)->GetNumberOfPoints() != 3){
			std::cout << "readed PLY data contains non-triangles. the current program cannot handle" << std::endl;
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
		
		float bbMargin = 3;
		int xstart = max(min(min(p1.x, p2.x), p3.x) - bbMargin, 0);
		int xend = min(ceil(max(max(p1.x, p2.x), p3.x) + bbMargin), size[0] - 1);
		int ystart = max(min(min(p1.y, p2.y), p3.y) - bbMargin, 0);
		int yend = min(ceil(max(max(p1.y, p2.y), p3.y) + bbMargin), size[1] - 1);
		int zstart = max(min(min(p1.z, p2.z), p3.z) - bbMargin, 0);
		int zend = min(ceil(max(max(p1.z, p2.z), p3.z) + bbMargin), size[2] - 1);

		for (unsigned int z = zstart; z < zend; z++)
		{
			for (unsigned int y = ystart; y < yend; y++)
			{
				for (unsigned int x = xstart; x < xend; x++)
				{
					ImageType::IndexType pixelIndex;
					pixelIndex[0] = x;
					pixelIndex[1] = y;
					pixelIndex[2] = z;

					float cur = image->GetPixel(pixelIndex);
					float v = disToTri(make_float3(x,y,z), p1, p2, p3, cur);
					if (v < cur){
						image->SetPixel(pixelIndex, v);
					}
				}
			}
		}
	}
	
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
				
				if (image->GetPixel(pixelIndex)< disThr){
					image->SetPixel(pixelIndex, 0);
				}
				else{
					image->SetPixel(pixelIndex, 1);
				}
			}
		}
	}
	
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
	processSurfaceData3();
	//processParticleMeshData();

	sdkStopTimer(&timer);

	float timeCost = sdkGetAverageTimerValue(&timer) / 1000.f;
	std::cout << "time cost for computing the preprocessing: " << timeCost << std::endl;

	return 1;

}
