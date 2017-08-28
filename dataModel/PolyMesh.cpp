#include "PolyMesh.h"
#include "Particle.h"
#include "BinaryTuplesReader.h"
#include <helper_math.h>

using namespace std;

PolyMesh::~PolyMesh(){
	if (vertexCoords) delete[]vertexCoords;
	if (vertexNorms) delete[]vertexNorms;
	if (indices) delete[]indices;
	if (vertexCoordsOri) delete[]vertexCoordsOri;
	if (vertexNormsOri) delete[]vertexNormsOri;
	if (indicesOri) delete[]indicesOri;
	if (faceValid) delete[]faceValid;

	if (vertexDeviateVals) delete[]vertexDeviateVals;

	if (vertexColorVals) delete[]vertexColorVals;
}


void PolyMesh::setVertexDeviateVals()
{
	if (vertexDeviateVals) delete[]vertexDeviateVals;
	vertexDeviateVals = (float*)malloc(sizeof(float)* vertexcount * 2); //times 2 to prepare for newly added vertices
	memset((void*)(vertexDeviateVals), 0, sizeof(float)* vertexcount * 2);
}


void PolyMesh::setVertexColorVals(float v)
{
	if (v < 0 || v > 1){
		std::cout << "vertexColorVals not implemented!!" << std::endl;
		exit(0);
	}
	if (vertexColorVals) delete[]vertexColorVals;
	vertexColorVals = (float*)malloc(sizeof(float)* vertexcount * 2); //times 2 to prepare for newly added vertices
	for (int i = 0; i < vertexcount; i++) {
		vertexColorVals[i] = v;
	}
	memset((void*)(vertexColorVals + vertexcount), 0, sizeof(float)* vertexcount);//the rest will always be set to 0 regardless of v
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
		
		if (vertexDeviateVals){
			memset((void*)(vertexDeviateVals), 0, sizeof(float)* vertexcount * 2);
		}
	}
}

bool PolyMesh::inRange(float3 v)
{
	return v.x >= min_x && v.x < max_x && v.y >= min_y && v.y < max_y && v.z >= min_z && v.z < max_z;
}

void PolyMesh::setAssisParticle(const char* fname)
{
	std::shared_ptr<BinaryTuplesReader> reader3 = std::make_shared<BinaryTuplesReader>(fname);
	particle = std::make_shared<Particle>();
	reader3->OutputToParticleData(particle);

	int nRegion = particle->numParticles;
	for (int i = 0; i < nRegion; i++){
		float3 c = make_float3(particle->pos[i]);
		int start = particle->valTuple[i * particle->tupleCount +2];
		int end = particle->valTuple[i * particle->tupleCount + 3];
		for (int j = start; j <= end; j++){
			vertexCoords[3 * j] -= c.x;
			vertexCoords[3 * j + 1] -= c.y;
			vertexCoords[3 * j + 2] -= c.z;
		}
	}

}

void PolyMesh::doShift(float3 shift)
{
	min_x += shift.x;
	max_x += shift.x;
	min_y += shift.y;
	max_y += shift.y;
	min_z += shift.z;
	max_z += shift.z;

	cx += shift.x;
	cy += shift.y;
	cz += shift.z;

	if (particle == 0){
		for (int i = 0; i < vertexcount; i++) {
			vertexCoords[3 * i] += shift.x;
			vertexCoords[3 * i + 1] += shift.y;
			vertexCoords[3 * i + 2] += shift.z;
		}
	}
	else{
		for (int i = 0; i < particle->numParticles; i++) {
			particle->pos[i].x += shift.x;
			particle->pos[i].y += shift.y;
			particle->pos[i].z += shift.z;

			particle->posOrig[i].x += shift.x;
			particle->posOrig[i].y += shift.y;
			particle->posOrig[i].z += shift.z;
		}
		particle->posMin += shift;
		particle->posMax += shift;
	}
}

void PolyMesh::createByCombiningPolyMeshes(std::vector<std::shared_ptr<PolyMesh>> polyMeshes)
{
	int nMeshes = polyMeshes.size();
	bool useVertexColor = true;
	vertexcount = 0, facecount = 0;
	for (int i = 0; i < nMeshes; i++){
		vertexcount += polyMeshes[i]->vertexcount;
		facecount += polyMeshes[i]->facecount;
		useVertexColor = (useVertexColor && (polyMeshes[i]->vertexColorVals != 0));
	}

	vertexCoords = new float[3 * vertexcount];
	vertexNorms = new float[3 * vertexcount];

	int vid = 0;
	for (int i = 0; i < nMeshes; i++){
		for (int j = 0; j < polyMeshes[i]->vertexcount; j++) {	
			vertexCoords[3 * vid] = polyMeshes[i]->vertexCoords[3 * j];
			vertexCoords[3 * vid + 1] = polyMeshes[i]->vertexCoords[3 * j + 1];
			vertexCoords[3 * vid + 2] = polyMeshes[i]->vertexCoords[3 * j + 2];
			vertexNorms[3 * vid] = polyMeshes[i]->vertexNorms[3 * j];
			vertexNorms[3 * vid + 1] = polyMeshes[i]->vertexNorms[3 * j + 1];
			vertexNorms[3 * vid + 2] = polyMeshes[i]->vertexNorms[3 * j + 2];
			vid++;
		}
	}

	indices = new unsigned[3 * facecount];

	int fid = 0;
	int offset = 0;
	for (int i = 0; i < nMeshes; i++){
		for (int j = 0; j < polyMeshes[i]->facecount; j++) {	
			indices[3 * fid] = polyMeshes[i]->indices[3 * j] + offset;
			indices[3 * fid + 1] = polyMeshes[i]->indices[3 * j + 1] + offset;
			indices[3 * fid + 2] = polyMeshes[i]->indices[3 * j + 2] + offset;
			fid++;
		}
		offset += polyMeshes[i]->vertexcount;
	}
	
	if (useVertexColor){
		if (vertexColorVals) delete[]vertexColorVals;
		vertexColorVals = (float*)malloc(sizeof(float)* vertexcount * 2); //times 2 to prepare for newly added vertices
		memset((void*)(vertexColorVals + vertexcount), 0, sizeof(float)* vertexcount);//the rest will always be set to 0 regardless of v
		
		int vid = 0;
		for (int i = 0; i < nMeshes; i++) {
			for (int j = 0; j < polyMeshes[i]->vertexcount; j++) {
				vertexColorVals[vid] = polyMeshes[i]->vertexColorVals[j];
				vid++;
			}
		}
	}

	find_center_and_range();
}