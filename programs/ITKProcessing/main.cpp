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

void computeChannelVolume(std::shared_ptr<Volume> v, std::shared_ptr<Volume> channelV, std::shared_ptr<RayCastingParameters> rcp)
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
					channelV->values[ind] = 1;
				}
				else{
					channelV->values[ind] = 0;
				}
			}
		}
	}

	std::cout << "finish computing channel volume..." << std::endl;
	return;
}


void computeSkel(shared_ptr<Volume> channelVolume, shared_ptr<Volume> skelVolume, int3 dims, float3 spacing, int &maxComponentMark)
{
	//compute skeleton volume and itk skelComponented image from the beginning
	std::cout << "computing skeletion volume..." << std::endl;
	const unsigned int numberOfPixels = dims.x * dims.y * dims.z;
	PixelType * localBuffer = new PixelType[numberOfPixels];
	for (int i = 0; i < numberOfPixels; i++){
		if (channelVolume->values[i] > 0.5){
			localBuffer[i] = 1;
		}
		else{
			localBuffer[i] = 0;
		}
	}
	ImageType::Pointer skelComponentedImg;
	skelComputing(localBuffer, dims, spacing, skelVolume->values, skelComponentedImg, maxComponentMark);
	//skelVolume->initVolumeCuda();
	std::cout << "finish computing skeletion volume..." << std::endl;
	skelVolume->saveRawToFile("skel.raw");
	return;
}


int main(int argc, char **argv)
{
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);

	sdkStartTimer(&timer);

	int3 dims;
	float3 spacing;

	std::shared_ptr<DataMgr> dataMgr;
	dataMgr = std::make_shared<DataMgr>();
	const std::string dataPath = dataMgr->GetConfig("VOLUME_DATA_PATH");
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


	shared_ptr<Volume> channelVolume = std::make_shared<Volume>();
	channelVolume->setSize(inputVolume->size);
	channelVolume->dataOrigin = inputVolume->dataOrigin;
	channelVolume->spacing = inputVolume->spacing;
	computeChannelVolume(inputVolume, channelVolume, rcp);

	shared_ptr<Volume> skelVolume = std::make_shared<Volume>();
	skelVolume->setSize(inputVolume->size);
	skelVolume->dataOrigin = inputVolume->dataOrigin;
	skelVolume->spacing = inputVolume->spacing;
	
	
	int maxComponentMark; 
	
	computeSkel(channelVolume, skelVolume, dims, spacing, maxComponentMark);

	//then, read already computed skeleton volume and itk skelComponented image
	std::cout << "reading skeletion volume..." << std::endl;
	std::shared_ptr<RawVolumeReader> reader;
	reader = std::make_shared<RawVolumeReader>("skel.raw", dims, RawVolumeReader::dtFloat32);
	reader->OutputToVolumeByNormalizedValue(skelVolume);
	reader.reset();
	typedef itk::ImageFileReader<ImageType> ReaderType;
	ReaderType::Pointer readeritk = ReaderType::New();
	readeritk->SetFileName("skelComponented.hdr");
	readeritk->Update();
	ImageType::Pointer connectedImg = readeritk->GetOutput();
	


	std::vector<std::vector<float3>> viewArrays;
	//sample the skeleton to set the views
	findViews(connectedImg, maxComponentMark, dims, spacing, viewArrays);


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
