#include "PositionBasedDeformProcessorForTV.h"
#include "TransformFunc.h"
#include "MatrixManager.h"

#include "Volume.h"
#include "Particle.h"
#include "PolyMesh.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

#include "PositionBasedDeformProcessor.h"
//#include "PolyRenderable.h"


////// !!!!!!!!!!!!!!!!!!!!!! the reset here actually do not work for real reset  !!!!!!!!!!!!!!!
void PositionBasedDeformProcessorForTV::turnActive()
{
	isActive = true;
	if (curT > -1){
		resetPolyMeshes();
	}
	curT = 0;
}

void PositionBasedDeformProcessorForTV::resetPolyMeshes()
{
	for (int i = 0; i < polyMeshes.size(); i++){
		//polyMeshes[i]->reset();
		polyMeshes[i]->particle->posOrig = polyMeshesOri[i]->particle->posOrig;
		polyMeshes[i]->particle->pos = polyMeshes[i]->particle->posOrig;
	}
}

void PositionBasedDeformProcessorForTV::saveOriginalCopyOfMeshes()
{
	for (int i = 0; i < polyMeshes.size(); i++){
		std::shared_ptr<PolyMesh> polyMesh = std::make_shared<PolyMesh>();
		polyMesh->particle = std::make_shared<Particle>();
		polyMesh->particle->numParticles = polyMeshes[i]->particle->numParticles;
		polyMesh->particle->posOrig = polyMeshes[i]->particle->posOrig;
		polyMeshesOri.push_back(polyMesh);
	}
}

bool PositionBasedDeformProcessorForTV::process(float* modelview, float* projection, int winWidth, int winHeight)
{
	if (!isActive){
		return false;
	}


	//sdkStopTimer(&timer);
	//float timePassed = sdkGetAverageTimerValue(&timer) / 1000.f;
	if (curT % (numInter + 1) == 0){
		int meshid = curT / (numInter + 1);
		
		//polyMesh->justChanged = true;
		//polyMesh->newPoly = polyMeshes[meshid];
		
		polyMesh = polyMeshes[meshid];
		//channelVolume = channelVolumes[meshid];
		
		justChangedForRenderer = true;
		
		//polyRenderable->polyMesh = polyMesh;//NOTE!! the polymesh stored in other modules are not always combined with the pointer "polyMesh" here!!!
		//polyRenderable->dataChange();

		positionBasedDeformProcessor->updateChannelWithTranformOfTVData(channelVolumes[meshid]);
		positionBasedDeformProcessor->updateParticleData(polyMesh->particle);
		//

	}
	else{
		int meshid1 = curT / (numInter + 1), meshid2 = meshid1 + 1;
		float ratio = 1.0 * (curT % (numInter + 1)) / (numInter + 1);

		//polyMesh = polyMeshes[meshid];//should not need to change polyMesh in this case
		int n = polyMesh->particle->numParticles;
		int tupleCount = polyMesh->particle->tupleCount; //should be 10
		for (int i = 0; i < n; i++){
			int m = cellMaps[meshid1][i];
			int label = polyMesh->particle->valTuple[i*tupleCount + 7];
			if (m > -1){
				polyMesh->particle->posOrig[i] = polyMeshesOri[meshid1]->particle->posOrig[i] * (1 - ratio) + polyMeshesOri[meshid2]->particle->posOrig[m] * ratio;
				polyMesh->particle->pos[i] = polyMesh->particle->posOrig[i];
				regionMoveVecs[label] = make_float3(polyMesh->particle->pos[i] - polyMeshesOri[meshid1]->particle->posOrig[i]);
			}
			else{
				//can think of someway to remove it from being shown
				polyMesh->particle->posOrig[i] = make_float4(-10000, -10000, -10000, 1);
				polyMesh->particle->pos[i] = polyMesh->particle->posOrig[i];

				regionMoveVecs[label] = make_float3(10001, 0, 0);//as a marker to denote that this region does not exist in the next time step
			}
		}
		
		positionBasedDeformProcessor->updateChannelWithTranformOfTVData_Intermediate(channelVolumes[meshid1], regionMoveVecs);
		positionBasedDeformProcessor->updateParticleData(polyMesh->particle); //actually pointer address of the particle is not changed, but the content is changed

	}
	//std::cout << "at " << curT << ", polyMesh->particle->pos[0]: " << polyMesh->particle->pos[0].x << " " << polyMesh->particle->pos[0].y << " " << polyMesh->particle->pos[0].z << std::endl;
	std::cout << "used data from " << curT << std::endl;

	curT++;
	if (curT > ((timeEnd - timeStart) * (numInter + 1))){
		isActive = false;
	}
	return true;
}