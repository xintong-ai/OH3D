#ifndef GLYPH_RENDERABLE_H
#define GLYPH_RENDERABLE_H

#include "Renderable.h"
class Displace;

class GlyphRenderable: public Renderable
{
	//float3 dataMin, dataMax;
protected:
	float4* pos = nullptr;
	int num;
	Displace* displace;
	void ComputeDisplace();
	
	GlyphRenderable(float4* _pos, int _num);// { pos = _pos; num = _num; displace.LoadOrig(spherePos, sphereCnt); }
	//float3 DataCenter();// { return (dataMin + dataMax) * 0.5; }
public:
	//void SetVolRange(float3 _dataMin, float3 _dataMax) { dataMin = _dataMin; dataMax = _dataMax; }
	bool MouseWheel(int x, int y, int delta)  override;
};
#endif