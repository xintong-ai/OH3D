#ifndef SCREENMARKER_H
#define SCREENMARKER_H

#include <stdlib.h>
#include <string>
#include <cuda_runtime.h>

class ScreenMarker{
public:
	bool justChanged = false;

	ScreenMarker(){};
	~ScreenMarker(){
		if (isPixelSelected != 0)
			delete[] isPixelSelected;
		if (dev_isPixelSelected != 0)
			delete[] dev_isPixelSelected;
	}
	
	void initMaskPixel(int w, int h){
		if (isPixelSelected != 0)
			delete[] isPixelSelected;
		if (dev_isPixelSelected != 0)
			cudaFree(dev_isPixelSelected);
		
		isPixelSelected = new char[w*h];
		memset(isPixelSelected, 0, w*h);

		cudaMalloc(&dev_isPixelSelected, sizeof(char)*w*h);
		cudaMemset(dev_isPixelSelected, 0, w*h);
	}
	
	char *isPixelSelected = 0;
	char *dev_isPixelSelected = 0;

};

#endif