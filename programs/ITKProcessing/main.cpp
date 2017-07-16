#include <memory>
#include <string>
#include <iostream>

//#include <cuda_runtime.h>
//#include <helper_cuda.h>

#include "Volume.h"
#include "RawVolumeReader.h"
#include "DataMgr.h"
#include "VecReader.h"

#include "myDefineRayCasting.h"

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
	//implicitModeller->SetSampleDimensions(xrange, yrange, zrange);
#if VTK_MAJOR_VERSION <= 5
	implicitModeller->SetInput(inputPolyData);
#else
	implicitModeller->SetInputData(inputPolyData);
#endif
	implicitModeller->AdjustBoundsOn();
	implicitModeller->SetAdjustDistance(.1); // Adjust by 10%
	implicitModeller->SetMaximumDistance(.1);


	implicitModeller->Update();

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
			for (unsigned int x= 0; x< size[0]; x++)
			{
				ImageType::IndexType pixelIndex;
				pixelIndex[0] = x;
				pixelIndex[1] = y;
				pixelIndex[2] = z;
				float v = *(static_cast<float*>(img->GetScalarPointer(x,y,z)));
				if (abs(v)>2){
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
	double outputspacing[3] = { 0, 0, 0 };


	typedef itk::IdentityTransform<double, 2> TransformType;
	typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
	ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();
	resample->SetInput(image);
	resample->SetSize(outputSize);
	resample->SetOutputSpacing(outputSpacing);
	resample->SetOutputOrigin(outputspacing);
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

int main(int argc, char **argv)
{
	

	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);

	sdkStartTimer(&timer);

	//processVolumeData();
	processSurfaceData();
	
	sdkStopTimer(&timer);

	float timeCost = sdkGetAverageTimerValue(&timer) / 1000.f;
	std::cout << "time cost for computing the preprocessing: " << timeCost << std::endl;

	return 1;

}
