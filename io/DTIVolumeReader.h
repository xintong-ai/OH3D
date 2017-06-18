#ifndef DTI_VOLUME_READER_H
#define DTI_VOLUME_READER_H

#include "NrrdVolumeReader.h"
#include <vector>
#include <memory>

class Particle;

//when read Nrrdvolume, this reader will only process the tensor out of all the gradients at each grid
class DTIVolumeReader :public NrrdVolumeReader
{
public:
	DTIVolumeReader(const char* filename);
	~DTIVolumeReader(){
		if (nullptr != eigenvec)
			delete[] eigenvec;
		if (nullptr != eigenval)
			delete[] eigenval;
		if (nullptr != majorEigenvec)
			delete[] majorEigenvec;
		if (nullptr != fracAnis)
			delete[] fracAnis;
		if (nullptr != colors)
			delete[] colors;
	}

	float3* GetMajorComponent();
	float* GetFractionalAnisotropy();
	float3* GetColors();
	void GetSamples(std::vector<float4>& _pos, std::vector<float>& _val);
	void GetSamplesWithFeature(std::vector<float4>& _pos, std::vector<float>& _val, std::vector<char>& _feature);
	//float* GetEigenValue();
	void OutputToParticleData(std::shared_ptr<Particle> v);

private:
	void EigenAnalysis();

	float* eigenvec = nullptr;
	float* eigenval = nullptr;
	float3* majorEigenvec = nullptr;
	float* fracAnis = nullptr;
	float3* colors = nullptr;
};

#endif