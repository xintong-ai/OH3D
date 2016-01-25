#ifndef SQ_RENDERABLE_H
#define SQ_RENDERABLE_H

#include "GlyphRenderable.h"
#include <memory>

class SQRenderable :public GlyphRenderable
{

public:
	SQRenderable(std::vector<float4> pos, std::vector < float > val);
};

#endif //SQ_RENDERABLE_H