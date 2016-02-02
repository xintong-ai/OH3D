#ifndef GLYPH_RENDERABLE_H
#define GLYPH_RENDERABLE_H

#include "Renderable.h"
#include <memory>
//#include <Displace.h>
class Displace;

class GlyphRenderable: public Renderable
{
	Q_OBJECT
		//float3 dataMin, dataMax;
protected:
	std::vector<float4> pos;// = nullptr;
	std::shared_ptr<Displace> displace;
	std::vector<float> glyphSizeScale;
	float glyphSizeAdjust = 0.5;

	void ComputeDisplace();
	//GlyphRenderable();
	GlyphRenderable(std::vector<float4>& _pos);//, float* _glyphSize = 0);// { pos = _pos; num = _num; displace.LoadOrig(spherePos, sphereCnt); }
	//float3 DataCenter();// { return (dataMin + dataMax) * 0.5; }
public:
	void RecomputeTarget();
	void DisplacePoints(std::vector<float2>& pts);
	//void SetVolRange(float3 _dataMin, float3 _dataMax) { dataMin = _dataMin; dataMax = _dataMax; }
public slots:
	//void SlotFocusSizeChanged(int v);// { displace - (10 - v) * 0.1; }
	//void SlotSideSizeChanged(int v);// { displace - (10 - v) * 0.1; }
	void SlotGlyphSizeAdjustChanged(int v);// { displace - (10 - v) * 0.1; }
	
};
#endif