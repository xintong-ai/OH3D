#include <memory>
#include <string>
#include <iostream>

//#include <cuda_runtime.h>
//#include <helper_cuda.h>

#include "Volume.h"
#include "RawVolumeReader.h"
#include "DataMgr.h"
#include "VecReader.h"

#include "myDefine.h"

#include "imageProcessing.h"
#include <itkImage.h>

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
	//channelV->initVolumeCuda();
	std::cout << "finish computing channel volume..." << std::endl;
	return;
}


void computeSkel(shared_ptr<Volume> channelVolume, shared_ptr<Volume> skelVolume, int3 dims, float3 spacing)
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
	int maxComponentMark;
	skelComputing(localBuffer, dims, spacing, skelVolume->values, skelComponentedImg, maxComponentMark);
	//skelVolume->initVolumeCuda();
	std::cout << "finish computing skeletion volume..." << std::endl;
	skelVolume->saveRawToFile("skel.raw");


	return;
}


int main(int argc, char **argv)
{
	int3 dims;
	float3 spacing;

	std::shared_ptr<DataMgr> dataMgr;
	dataMgr = std::make_shared<DataMgr>();
	const std::string dataPath = dataMgr->GetConfig("VOLUME_DATA_PATH");

	std::shared_ptr<RayCastingParameters> rcp = std::make_shared<RayCastingParameters>();

	if (std::string(dataPath).find("MGHT2") != std::string::npos){
		dims = make_int3(320, 320, 256);
		spacing = make_float3(0.7, 0.7, 0.7);
	}
	else if (std::string(dataPath).find("MGHT1") != std::string::npos){
		dims = make_int3(256, 256, 176);
		spacing = make_float3(1.0, 1.0, 1.0);
		rcp = std::make_shared<RayCastingParameters>(1.0, 0.2, 0.7, 0.44, 0.29, 1.25, 512, 0.25f, 1.3, false);
	}
	else if (std::string(dataPath).find("nek128") != std::string::npos){
		dims = make_int3(128, 128, 128);
		spacing = make_float3(2, 2, 2); //to fit the streamline of nek256
	}
	else if (std::string(dataPath).find("nek256") != std::string::npos){
		dims = make_int3(256, 256, 256);
		spacing = make_float3(1, 1, 1);
	}
	else if (std::string(dataPath).find("cthead") != std::string::npos){
		dims = make_int3(208, 256, 225);
		spacing = make_float3(1, 1, 1);
	}
	else if (std::string(dataPath).find("brat") != std::string::npos){
		dims = make_int3(160, 216, 176);
		spacing = make_float3(1, 1, 1);
		rcp = std::make_shared<RayCastingParameters>(1.0, 0.2, 0.7, 0.44, 0.25, 1.25, 512, 0.25f, 1.3, false); //for brat
	}
	else if (std::string(dataPath).find("engine") != std::string::npos){
		dims = make_int3(149, 208, 110);
		spacing = make_float3(1, 1, 1);
		rcp = std::make_shared<RayCastingParameters>(0.8, 0.4, 1.2, 1.0, 0.05, 1.25, 512, 0.25f, 1.0, false);
	}
	else if (std::string(dataPath).find("knee") != std::string::npos){
		dims = make_int3(379, 229, 305);
		spacing = make_float3(1, 1, 1);
	}
	else if (std::string(dataPath).find("181") != std::string::npos){
		dims = make_int3(181, 217, 181);
		spacing = make_float3(1, 1, 1);
		rcp = std::make_shared<RayCastingParameters>(1.8, 1.0, 1.5, 1.0, 0.3, 2.6, 512, 0.25f, 1.0, false); //for 181
	}
	else{
		std::cout << "volume data name not recognized" << std::endl;
		exit(0);
	}

	shared_ptr<Volume> inputVolume = std::make_shared<Volume>(true);
	if (std::string(dataPath).find(".vec") != std::string::npos){
		std::shared_ptr<VecReader> reader;
		reader = std::make_shared<VecReader>(dataPath.c_str());
		reader->OutputToVolumeByNormalizedVecMag(inputVolume);
		//reader->OutputToVolumeByNormalizedVecDownSample(inputVolume,2);
		//reader->OutputToVolumeByNormalizedVecUpSample(inputVolume, 2);
		//reader->OutputToVolumeByNormalizedVecMagWithPadding(inputVolume,10);
		reader.reset();
	}
	else{
		std::shared_ptr<RawVolumeReader> reader;
		if (std::string(dataPath).find("engine") != std::string::npos || std::string(dataPath).find("knee") != std::string::npos || std::string(dataPath).find("181") != std::string::npos){
			reader = std::make_shared<RawVolumeReader>(dataPath.c_str(), dims, RawVolumeReader::dtUint8);
		}
		else{
			reader = std::make_shared<RawVolumeReader>(dataPath.c_str(), dims);
		}
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
	computeSkel(channelVolume, skelVolume, dims, spacing);


	
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
	int maxComponentMark = 55;


	std::vector<std::vector<float3>> viewArrays;
	//find the views for the tour
	findViews(connectedImg, maxComponentMark, dims, spacing, viewArrays);
	
	//std::cout << "views: " << endl << views.size() << endl << views[5].x << " " << views[5].y << " " << views[5].z<<endl;

	return 1;

}
