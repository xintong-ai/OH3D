#include "GlyphRenderable.h"
#include <Displace.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>
#include <glwidget.h>
#include <Lens.h>
#include <LensRenderable.h>


void GlyphRenderable::ComputeDisplace()
{
	int2 winSize = actor->GetWindowSize();
	displace->Compute(&matrix_mv.v[0].x, &matrix_pj.v[0].x, winSize.x, winSize.y,
		((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses(), &pos[0], &glyphSizeScale[0], &glyphBright[0]);
}


GlyphRenderable::GlyphRenderable(std::vector<float4>& _pos)
{ 
	pos = _pos; 
	displace = std::make_shared<Displace>();
	displace->LoadOrig(&pos[0], pos.size());
	glyphSizeScale.assign(pos.size(), 1.0f);
	glyphBright.assign(pos.size(), 1.0f);
}

GlyphRenderable::~GlyphRenderable()
{ 
	if (nullptr != glProg){
		delete glProg;
		glProg = nullptr;
	}
}


void GlyphRenderable::RecomputeTarget()
{
	displace->RecomputeTarget();
}

void GlyphRenderable::DisplacePoints(std::vector<float2>& pts)
{
	int2 winSize = actor->GetWindowSize();
	displace->DisplacePoints(pts,
		((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses(), &matrix_mv.v[0].x, &matrix_pj.v[0].x, winSize.x, winSize.y);
}

void GlyphRenderable::SlotGlyphSizeAdjustChanged(int v)
{
	glyphSizeAdjust = v * 0.1;
	actor->UpdateGL();
}

void GlyphRenderable::mouseMove(int x, int y, int modifier)
{
}

void GlyphRenderable::resize(int width, int height)
{
	if (INTERACT_MODE::TRANSFORMATION == actor->GetInteractMode()) {
		RecomputeTarget();
	}
}

float3 GlyphRenderable::findClosetGlyph(float3 aim)
{
	return displace->findClosetGlyph(aim, snappedGlyphId);
}



