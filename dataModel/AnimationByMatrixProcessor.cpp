#include <iostream>
#include <helper_math.h>
#include "AnimationByMatrixProcessor.h"
#include "GLMatrixManager.h"

void AnimationByMatrixProcessor::startAnimation()
{
	if (views.size() > 1){
		isActive = true;
		start = std::clock();
	}
	else{
		std::cout << "views not set correctly! animation cannot start" << std::endl;
	}
}

bool AnimationByMatrixProcessor::process(float modelview[16], float projection[16], int winW, int winH)
{
	if (!isActive)
		return false;

	double past = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	if (past >= totalDuration){
		//end animation
		isActive = false;
		std::cout << "animiation ends" << std::endl;
	}

	int n = views.size();
	double p = past / totalDuration * (n - 1);
	int n1 = floor(p), n2 = n1 + 1;
	if (n2 < n){
		float3 view = views[n1] * (n2 - p) + views[n2] * (p - n1);
		//std::cout << view.x << " " << view.y << " " << view.z << std::endl;
		matrixMgr->moveEyeInLocalByModeMat(view);
	}
	return false;
}

