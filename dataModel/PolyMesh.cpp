#include "PolyMesh.h"

PolyMesh::~PolyMesh(){
	if (vertexCoords) delete[]vertexCoords;
	if (vertexNorms) delete[]vertexNorms;
	if (indices) delete[]indices;
	if (vertexCoordsOri) delete[]vertexCoordsOri;
	if (vertexNormsOri) delete[]vertexNormsOri;
	if (indicesOri) delete[]indicesOri;
	if (faceValid) delete[]faceValid;
}

void PolyMesh::find_center_and_range()
{
	float x, y, z;
	min_x = 9999, max_x = -9999, min_y = 9999, max_y = -9999;
	min_z = 9999, max_z = -9999;

	x = y = z = 0;
	for (int i = 0; i < vertexcount; i++) {
		x += vertexCoords[3 * i];
		y += vertexCoords[3 * i + 1];
		z += vertexCoords[3 * i + 2];
		if (min_x > vertexCoords[3 * i]) min_x = vertexCoords[3 * i];
		if (max_x < vertexCoords[3 * i]) max_x = vertexCoords[3 * i];
		if (min_y > vertexCoords[3 * i + 1]) min_y = vertexCoords[3 * i + 1];
		if (max_y < vertexCoords[3 * i + 1]) max_y = vertexCoords[3 * i + 1];
		if (min_z > vertexCoords[3 * i + 2]) min_z = vertexCoords[3 * i + 2];
		if (max_z < vertexCoords[3 * i + 2]) max_z = vertexCoords[3 * i + 2];
	}
	cx = x / (float)vertexcount;
	cy = y / (float)vertexcount;
	cz = z / (float)vertexcount;

}

void PolyMesh::GetPosRange(float3& posMin, float3& posMax)
{
	posMin = make_float3(min_x, min_y, min_z);
	posMax = make_float3(max_x, max_y, max_z);
}

void PolyMesh::setVertexCoordsOri() //needed when the data is used for deformation
{
	if (vertexcount > 0){
		if (vertexCoordsOri) delete[]vertexCoordsOri;
		if (vertexNormsOri) delete[]vertexNormsOri;
		if (indicesOri) delete[]indicesOri;

		//let's hope double the storage will be large enough
		vertexcountOri = vertexcount;
		vertexCoordsOri = vertexCoords;
		vertexCoords = (float*)malloc(sizeof(float)* 3 * vertexcount * 2);
		memcpy(vertexCoords, vertexCoordsOri, sizeof(float)* 3 * vertexcount);

		vertexNormsOri = vertexNorms;
		vertexNorms = (float*)malloc(sizeof(float)* 3 * vertexcount * 2);
		memcpy(vertexNorms, vertexNormsOri, sizeof(float)* 3 * vertexcount);

		facecountOri = facecount;
		indicesOri = indices;
		indices = (unsigned int*)malloc(sizeof(unsigned int)* 3 * facecount * 2);
		memcpy(indices, indicesOri, sizeof(unsigned int)* 3 * facecount);

		faceValid = new bool[facecount];
		for (int i = 0; i < facecount; i++){
			faceValid[i] = true;
		}
	}
	readyForDeform = true;
}



void PolyMesh::reset()
{
	if (vertexcountOri > 0){
		vertexcount = vertexcountOri;
		memcpy(vertexCoords, vertexCoordsOri, sizeof(float)* 3 * vertexcount);
		memcpy(vertexNorms, vertexNormsOri, sizeof(float)* 3 * vertexcount);
		facecount = facecountOri;
		memcpy(indices, indicesOri, sizeof(unsigned int)* 3 * facecount);
		std::cout << "poly data successfully reset " << std::endl;
	}
}

bool PolyMesh::inRange(float3 v)
{
	return v.x >= min_x && v.x < max_x && v.y >= min_y && v.y < max_y && v.z >= min_z && v.z < max_z;
}