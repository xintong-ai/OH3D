#include "BinaryParticleReader.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector_functions.h>

//for linux
#include <float.h>
#include <stdexcept>
#include <memory>


void BinaryParticleReader::Load()
{
	FILE *pFile;
	pFile = fopen(datafilename.c_str(), "rb");
	if (pFile == NULL) { fputs("particle file error", stderr); exit(1); }
	int numParticles;
	fread(&numParticles, sizeof(int), 1, pFile);

	const int numComponentEachBinarySeg = 5;
	float *coords = new float[numParticles * numComponentEachBinarySeg];
	fread(coords, sizeof(float), numParticles * numComponentEachBinarySeg, pFile);


	pos.resize(numParticles);
	val.resize(numParticles);
	feature.resize(numParticles);
	for (int i = 0; i < numParticles; i++){
		pos[i] = make_float4(coords[numComponentEachBinarySeg * i], coords[numComponentEachBinarySeg * i + 1], coords[numComponentEachBinarySeg * i + 2], 1.0);
		val[i] = coords[numComponentEachBinarySeg * i + 3];
		feature[i] = (char)coords[numComponentEachBinarySeg * i + 4];
	}
	delete[] coords;
}


void BinaryParticleReader::OutputToParticleData(std::shared_ptr<Particle> v)
{
	v->clear();

	v->pos = pos;
	v->posOrig = pos;
	v->val = val;

	v->numParticles = pos.size();
	v->updateMaxMinValAndPos();
	
	if (feature.size() > 0){
		v->setFeature(feature);
	}
}