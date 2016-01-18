#include "Renderable.h"
#include "DataMgr.h"
#include "glwidget.h"

void Renderable::AllocOutImage() {
    if(h_output != NULL)
        delete [] h_output;
	int2 winSize = actor->GetWindowSize();
	h_output = new uint[winSize.x * winSize.y];
}

Renderable::~Renderable() {
    if(h_output != NULL)
        delete [] h_output;
}

void Renderable::resize(int width, int height) {
    //winWidth = width;
    //winHeight = height;
    //AllocOutImage();
}

void Renderable::draw(float modelview[16], float projection[16])
{
}

//void Renderable::GetDataDim(int &nx, int &ny, int &nz) {
//    dataMgr->GetDataDim(nx, ny, nz);
//}
