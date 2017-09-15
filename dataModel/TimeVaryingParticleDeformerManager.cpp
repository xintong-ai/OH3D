#include "TimeVaryingParticleDeformerManager.h"
#include "TransformFunc.h"
#include "MatrixManager.h"

#include "Particle.h"
#include "PolyMesh.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

#include "PositionBasedDeformProcessor.h"

void TimeVaryingParticleDeformerManager::turnActive()
{
	isActive = true;
	if (curT > -1){
		resetPolyMeshes();
	}
	curT = 0;

	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
}

////// not actual full reset, just reset the parts that might be changed
void TimeVaryingParticleDeformerManager::resetPolyMeshes()
{
	for (int i = 0; i < polyMeshes.size(); i++){
		//polyMeshes[i]->reset();
		polyMeshes[i]->particle->posOrig = polyMeshesOri[i]->particle->posOrig;
		polyMeshes[i]->particle->pos = polyMeshes[i]->particle->posOrig;
	}
}

////// not all infor is recorded, but just record necessary parts
void TimeVaryingParticleDeformerManager::saveOriginalCopyOfMeshes()
{
	for (int i = 0; i < polyMeshes.size(); i++){
		std::shared_ptr<PolyMesh> curPoly = std::make_shared<PolyMesh>();
		curPoly->particle = std::make_shared<Particle>();
		curPoly->particle->numParticles = polyMeshes[i]->particle->numParticles;
		curPoly->particle->posOrig = polyMeshes[i]->particle->posOrig;
		polyMeshesOri.push_back(curPoly);
	}
}

bool TimeVaryingParticleDeformerManager::process(float* modelview, float* projection, int winWidth, int winHeight)
{
	if (!isActive){
		return false;
	}

	if (paused){
		return false;
	}

	float timePassed = timer->getTime();
	int timeStepPassed = (int)(timePassed / durationEachTimeStep);



	if (timeStepPassed != curT){// curT % (numInter + 1) == 0){	//in this case, not only the particles stored in polyMesh will change, but the vertices of polyMesh will also change
		//therefore, this branch must be done at least once when entering a new time step, to update the vertices

		int meshid = timeStepPassed;// curT / (numInter + 1);
		curT = timeStepPassed;

		if (curT > timeEnd - timeStart){
			std::cout << "slow machine behavior in time varying manager!!" << curT + timeStart << std::endl;
			isActive = false;
			positionBasedDeformProcessor->tv = false;
			sdkResetTimer(&timer);
			return true;//on very slow machine, or during debugging
		}
		std::cout << "used data from " << curT + timeStart << std::endl;


		polyMesh->copyFrom(polyMeshes[meshid], true);
		polyMesh->verticesJustChanged = true;

		if (curT > 0)
		{
			std::vector<float4> lastPos;
			positionBasedDeformProcessor->getLastPos(lastPos);

			int n = polyMesh->particle->numParticles;
			std::vector<float4> newLastPos(n, make_float4(-1000, -1000, -1000, 1000));

			int lastT = curT - 1;
			int ll = cellMaps[lastT].size();
			for (int i = 0; i < ll; i++){
				int m = cellMaps[lastT][i];
				if (m > -1){
					newLastPos[m] = lastPos[i];
				}
			}
			positionBasedDeformProcessor->newLastPos(newLastPos);
		}

	}
	else{ //in this case, only the particles stored in polyMesh will change, and the vertices of polyMesh will NOT change

		if (curT >= timeEnd - timeStart){
			std::cout << "UNEXPECTED behavior in time varying manager!!" << curT + timeStart << std::endl;
			isActive = false;
			positionBasedDeformProcessor->tv = false;
			sdkResetTimer(&timer);
			return true;
		}

		int meshid1 = curT, meshid2 = meshid1 + 1;
		float ratio = 1.0 * (timePassed / durationEachTimeStep - timeStepPassed * 1.0);

		int n = polyMesh->particle->numParticles;
		int tupleCount = polyMesh->particle->tupleCount; //should be 10
		for (int i = 0; i < n; i++){
			int m = cellMaps[meshid1][i];
			if (m > -1){
				polyMesh->particle->posOrig[i] = polyMeshesOri[meshid1]->particle->posOrig[i] * (1 - ratio) + polyMeshesOri[meshid2]->particle->posOrig[m] * ratio;
				polyMesh->particle->pos[i] = polyMesh->particle->posOrig[i];
			}
			else{
				//can think of someway to remove it from being shown
				polyMesh->particle->posOrig[i] = make_float4(-10000, -10000, -10000, 1);
				polyMesh->particle->pos[i] = polyMesh->particle->posOrig[i];
			}
		}
	}

	positionBasedDeformProcessor->particleDataUpdated();


	//std::cout << "at " << curT << ", polyMesh->particle->pos[0]: " << polyMesh->particle->pos[0].x << " " << polyMesh->particle->pos[0].y << " " << polyMesh->particle->pos[0].z << std::endl;


	if (curT >= timeEnd - timeStart){
		isActive = false;
		positionBasedDeformProcessor->tv = false;
		sdkResetTimer(&timer);
	}
	return true;
}

