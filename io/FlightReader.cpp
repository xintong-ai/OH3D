#include "FlightReader.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector_functions.h>
#include <helper_math.h>
#define PI 3.1415926
//for linux
#include <float.h>
#include <stdexcept>
// a simple class to encapsulate a single timestep;
// the constructor loads the timestep data from the specified file
// into three vectors

#include <sstream>

class CSVRow
{
public:
	std::string const& operator[](std::size_t index) const
	{
		return m_data[index];
	}
	std::size_t size() const
	{
		return m_data.size();
	}
	void readNextRow(std::istream& str)
	{
		std::string         line;
		std::getline(str, line);

		std::stringstream   lineStream(line);
		std::string         cell;

		m_data.clear();
		while (std::getline(lineStream, cell, ','))
		{
			m_data.push_back(cell);
		}
	}
private:
	std::vector<std::string>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data)
{
	data.readNextRow(str);
	return str;
}

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

void FlightReader::Load(const char* filename)
{
	//ts = new timestep(datafilename.c_str());

	std::ifstream       file(filename);

	CSVRow              row;
	file >> row;
	float lat, lon, alt, x, y, z;
	int flt;
	float earthRadius = 6367; //km
	while (file >> row)
	{
		//std::cout << "4th Element(" << row[3] << ")\n";
		lat = std::stof(row[0]);
		lon = std::stof(row[1]);
		alt = std::stof(row[2]);
		flt = std::stoi(row[3]);
		x = (earthRadius + alt) * cos(lat * PI / 180.0) * cos(lon * PI / 180.0);
		y = (earthRadius + alt) * cos(lat * PI / 180.0) * sin(lon * PI / 180.0);
		z = (earthRadius + alt) * sin(lat * PI / 180.0);
		pos.push_back(make_float4(x, y, z, 1.0f));
		val.push_back(flt);
	}
	TranslateToCenter();
	//delete ts;
	//float a, b;
	//GetValRange(a, b);
	//float3 pMin, pMax;
	//GetDataRange(pMin, pMax);
}

void FlightReader::TranslateToCenter()
{
	float4 avg = make_float4(0, 0, 0, 0);
	for (auto& p : pos){
		avg += p;
	}
	avg /= pos.size();
	for (auto& p : pos){
		p.x = (p.x - avg.x) * 0.1;
		p.y = (p.y - avg.y) * 0.1;
		p.z = (p.z - avg.z) * 0.1;
	}
}


void FlightReader::GetValRange(float& vMin, float& vMax)
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


void FlightReader::GetPosRange(float3& posMin, float3& posMax)
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

std::vector<float4> FlightReader::GetPos()
{
	return pos;
}


int FlightReader::GetNum()
{
	return pos.size();
}

//float* ParticleReader::GetVal()
//{
//	return &val[0];
//}

std::vector<float> FlightReader::GetVal()
{
	return val;
}