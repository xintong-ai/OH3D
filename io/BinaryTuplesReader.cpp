#include "BinaryTuplesReader.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector_functions.h>

//for linux
#include <float.h>
#include <stdexcept>
#include <memory>


void BinaryTuplesReader::ReadArray(std::vector<float4> & curArray, std::vector<float> & curValArray, std::vector<char> & curFeatureArray, FILE *pFile){
	int numParticles;
	fread(&numParticles, sizeof(int), 1, pFile);
	int numComponentEachBinarySeg;
	fread(&numComponentEachBinarySeg, sizeof(int), 1, pFile);

	if (numComponentEachBinarySeg != 5){
		std::cout << "not implemented yet!!" << std::endl;
		exit(0);
	}

	float *coords = new float[numParticles * numComponentEachBinarySeg];
	fread(coords, sizeof(float), numParticles * numComponentEachBinarySeg, pFile);
	curArray.resize(numParticles);
	curValArray.resize(numParticles);
	curFeatureArray.resize(numParticles);
	for (int i = 0; i < numParticles; i++){
		curArray[i] = make_float4(coords[numComponentEachBinarySeg * i], coords[numComponentEachBinarySeg * i + 1], coords[numComponentEachBinarySeg * i + 2], 1.0);
		curValArray[i] = (char)coords[numComponentEachBinarySeg * i + 3];
		curFeatureArray[i] = (char)coords[numComponentEachBinarySeg * i + 4];
	}
	delete[] coords;
}

void BinaryTuplesReader::Load()
{
	FILE *pFile;
	pFile = fopen(datafilename.c_str(), "rb");
	if (pFile == NULL) { fputs("particle file error", stderr); exit(1); }

	fread(&numTupleArrays, sizeof(int), 1, pFile);
	posArrays.resize(numTupleArrays);
	valArrays.resize(numTupleArrays);
	featureArrays.resize(numTupleArrays);

	for (int i = 0; i < numTupleArrays; i++){
		ReadArray(posArrays[i], valArrays[i], featureArrays[i], pFile);
	}
}


void BinaryTuplesReader::OutputToParticleDataArrays(std::vector<std::shared_ptr<Particle>> & v)
{
	v.resize(posArrays.size());

	for (int i = 0; i < v.size(); i++){
		//v[i]->clear();
		v[i] = std::make_shared<Particle>();

		v[i]->pos = posArrays[i];
		v[i]->posOrig = posArrays[i];
		v[i]->val = valArrays[i];

		v[i]->numParticles = posArrays[i].size();
		v[i]->updateMaxMinValAndPos();

		if (feature.size() > 0){
			v[i]->setFeature(featureArrays[i]);
		}
	}
}