#ifndef IMAGECOMPUTER_H
#define IMAGECOMPUTER_H

#include <QtWidgets>
#include <memory>
#include <vector_types.h>

class GLMatrixManager;
struct RayCastingParameters;

class ImageComputer
{

public:
	ImageComputer(std::shared_ptr<GLMatrixManager> m);
	~ImageComputer();
	
	int counterL = 0, counterR = 0;

	void compute(int3 volumeSize, float3 spacing, std::shared_ptr<RayCastingParameters>);
	std::vector<float3> viewpoints;
	int N = 30;// better be the times of 30 due to 30fps
	//67 74 - 137 107
private:
	void saveImage(uint *output, int winWidth, int winHeight);
	std::shared_ptr<GLMatrixManager> matrixMgr;

	uint *d_output, *output;
	
	const int IMGWIDTH = 4096;
	const int IMGHEIGHT = IMGWIDTH / 2;

	int winWidth = 4096, winHeight = winWidth;

};


#endif

