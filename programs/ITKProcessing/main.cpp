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


using namespace std;

const int PHASES_COUNT = 3; //phase 1 is for creating cell volume; phase 2 is for creating skeletion; phase 3 is for creating bilateral volume

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

int main(int argc, char **argv)
{
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);

	sdkStartTimer(&timer);


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
		return 1;
	}


	//compute skeletons
	ImageType::Pointer skelComponentedImg;
	std::cout << "finish computing skeletion volume..." << std::endl;
	skelComputing(connectedImg, initCellImg, dims, spacing, skelComponentedImg, maxComponentMark);

	std::vector<std::vector<float3>> viewArrays;
	//sample the skeleton to set the views
	findViews(skelComponentedImg, maxComponentMark, dims, spacing, viewArrays);

	if (PHASES_COUNT == 2){
		return 1;
	}


	//compute the bilateralVolume
	float* bilateralVolumeRes = 0;
	inputVolume->computeBilateralFiltering(bilateralVolumeRes, 2, 0.2);
	FILE * fp = fopen("bilat.raw", "wb");
	fwrite(bilateralVolumeRes, sizeof(float), inputVolume->size.x*inputVolume->size.y*inputVolume->size.z, fp);
	fclose(fp);
	delete[]bilateralVolumeRes;

	sdkStopTimer(&timer);

	float timeCost = sdkGetAverageTimerValue(&timer) / 1000.f;
	std::cout << "time cost for computing the preprocessing: " << timeCost << std::endl;

	return 1;

}
