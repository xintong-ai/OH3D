#ifndef ANIMATIONBYMATRIXPROCESSOR_H
#define ANIMATIONBYMATRIXPROCESSOR_H
#include <memory>
#include <ctime>
#include <vector>
#include <vector_types.h>

#include "Processor.h"

class GLMatrixManager;

class AnimationByMatrixProcessor :public Processor
{
public:
	AnimationByMatrixProcessor(std::shared_ptr<GLMatrixManager> _v){
		matrixMgr = _v;
	};
	~AnimationByMatrixProcessor(){};

	void startAnimation();
	void setViews(std::vector<float3> _views){
		views = _views;
	};
private:
	std::shared_ptr<GLMatrixManager> matrixMgr;

	std::clock_t start;
	double totalDuration = 5;

	std::vector<float3> views;

public:
	bool process(float modelview[16], float projection[16], int winW, int winH) override;
	
};
#endif