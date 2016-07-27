#include "UniverseParticleReader.h"
#include <vector>
#include <iostream>
#include <fstream>

void UniverseParticleReader::Load()
{
	std::ifstream infile;
	infile.open(datafilename.c_str(), std::ios::binary | std::ios::in);
	infile.read((char*)&num, sizeof(int)); // reads 7 bytes into a cell that is either 2 or 4 

	float p[3];
	for (int i = 0; i < num; i+=100) {

		infile.read((char*)&p, 3 * sizeof(float)); // reads 7 bytes into a cell that is either 2 or 4 
		pos.push_back(make_float4(p[0], p[1], p[2], 1.0f));
		val.push_back(1.0f);
		//pos.push_back(make_float4(ts->position[3 * i], ts->position[3 * i + 1], ts->position[3 * i + 2], 1.0f));
		//val.push_back(ts->concentration[i]);
	}
}
