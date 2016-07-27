#include "ParticleReader.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector_functions.h>

//for linux
#include <float.h>
#include <stdexcept>
// a simple class to encapsulate a single timestep;
// the constructor loads the timestep data from the specified file
// into three vectors
struct timestep
{
	unsigned int size;  // number of particles
	unsigned int step;  // simulation time index
	float        time;  // simulation time

	std::vector<float> position;      // array of (x,y,z) positions
	std::vector<float> velocity;      // array of (x,y,z) velocities
	std::vector<float> concentration; // array of concentrations

	timestep(const char* filename)
	{
		const size_t MAGIC_OFFSET = 4072;

		struct
		{
			std::int32_t _pad3;
			std::int32_t size;
			std::int32_t _pad1;
			std::int32_t step;
			std::int32_t _pad2;
			float        time;
		} header;

		std::ifstream in(filename, std::ios::binary);

		if (!in)
			throw std::runtime_error(
			"unable to read timestep from " +
			std::string(filename)
			);

		// read header
		in.seekg(MAGIC_OFFSET);
		in.read(reinterpret_cast<char*>(&header), sizeof(header));

		size = header.size;
		step = header.step;
		time = header.time;

		// allocate memory for data arrays
		concentration.resize(size);

		// read position array
		position.resize(size * 3);

		in.seekg(4, std::ios_base::cur);
		in.read(reinterpret_cast<char*>(&position[0]),
			position.size() * sizeof(float));

		// read velocity array
		velocity.resize(size * 3);

		in.seekg(4, std::ios_base::cur);
		in.read(reinterpret_cast<char*>(&velocity[0]),
			velocity.size() * sizeof(float));

		// read concentration array
		in.seekg(4, std::ios_base::cur);
		in.read(reinterpret_cast<char*>(&concentration[0]),
			concentration.size() * sizeof(float));

		if (!in)
			throw std::runtime_error(
			"unable to read timestep from " +
			std::string(filename)
			);

		std::cerr << "step " << step << ", "
			<< "time " << time << ", "
			<< size << " particles" << std::endl;
	}
};

void ParticleReader::GetValRange(float& vMin, float& vMax)
{
	vMax = -FLT_MAX;
	vMin = FLT_MAX;
	int n = pos.size();
	float v = 0;
	for (int i = 0; i < n; i++) {
		v = val[i];
		if (v > vMax)
			vMax = v;
		if (v < vMin)
			vMin = v;
	}
}


void ParticleReader::GetPosRange(float3& posMin, float3& posMax)
{
	posMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	posMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	int n = pos.size();
	float v = 0;
	for (int i = 0; i < n; i++) {
		v = pos[i].x;
		if (v > posMax.x)
			posMax.x = v;
		if (v < posMin.x)
			posMin.x = v;

		v = pos[i].y;
		if (v > posMax.y)
			posMax.y = v;
		if (v < posMin.y)
			posMin.y = v;

		v = pos[i].z;
		if (v > posMax.z)
			posMax.z = v;
		if (v < posMin.z)
			posMin.z = v;
	}
}


//float4* ParticleReader::GetPos()
//{
//	return (float4*)(&pos[0]);
//}

std::vector<float4> ParticleReader::GetPos()
{
	return pos;
}


int ParticleReader::GetNum()
{
	return pos.size();
}

//float* ParticleReader::GetVal()
//{
//	return &val[0];
//}

std::vector<float> ParticleReader::GetVal()
{
	return val;
}
