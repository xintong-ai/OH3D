#ifndef INFOGUIDE_RENDERABLE_H
#define INFOGUIDE_RENDERABLE_H
#include "Renderable.h"
#include <memory>
#include <vector>
#include <helper_math.h>

class ViewpointEvaluator;
class GLMatrixManager;
class InfoGuideRenderable :public Renderable
{
	Q_OBJECT

	std::shared_ptr<ViewpointEvaluator> ve;
	std::shared_ptr<GLMatrixManager> matrixMgr;

	void drawLocalGuide(int2 winSize);
	void drawGlobalGuide(float modelview[16], float projection[16], int2 winSize);
	std::vector<float2> globalGuideArrow;
	std::vector<std::vector<float2>> turnArrowParts; //GL_Polygon only for concave shapes

public:
	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	
	InfoGuideRenderable(std::shared_ptr<ViewpointEvaluator> _ve, std::shared_ptr<GLMatrixManager> m){
		ve = _ve;
		matrixMgr = m;
		
		//float2 shift = make_float2(0.5, 0.5);
		float2 shift = make_float2(-0.5, 0);

		globalGuideArrow.push_back(make_float2(0.5, 1) + shift);
		globalGuideArrow.push_back(make_float2(0, 0.6) + shift);
		globalGuideArrow.push_back(make_float2(0.3, 0.6) + shift);
		globalGuideArrow.push_back(make_float2(0.3, 0) + shift);
		globalGuideArrow.push_back(make_float2(0.7, 0) + shift);
		globalGuideArrow.push_back(make_float2(0.7, 0.6) + shift);
		globalGuideArrow.push_back(make_float2(1, 0.6) + shift);

		
		//std::vector<float2> turnArrow;
		//turnArrow.clear();
		//turnArrow.push_back(make_float2(1, 0));
		//turnArrow.push_back(make_float2(1, 0.5));
		//turnArrow.push_back(make_float2(0.7, 1));
		//turnArrow.push_back(make_float2(0.5, 1));
		//turnArrow.push_back(make_float2(0.5, 0.8));
		//turnArrow.push_back(make_float2(0.8, 0.8));
		//turnArrow.push_back(make_float2(0.8, 1));
		//turnArrowParts.push_back(turnArrow);


		//turnArrow.clear();
		//turnArrow.push_back(make_float2(0.5, 1));
		//turnArrow.push_back(make_float2(0.4, 1));
		//turnArrow.push_back(make_float2(0.1, 0.3));
		//turnArrow.push_back(make_float2(0.3, 0.3));
		//turnArrow.push_back(make_float2(0.3, 0.7));
		//turnArrow.push_back(make_float2(0.5, 0.8));
		//turnArrowParts.push_back(turnArrow);

		std::vector<float2> turnArrow;
		turnArrow.clear();
		turnArrow.push_back(make_float2(0.9, 0));
		turnArrow.push_back(make_float2(1, 0.2));
		turnArrow.push_back(make_float2(1, 0.6));
		turnArrow.push_back(make_float2(0.8, 1));
		turnArrow.push_back(make_float2(0.4, 1));
		turnArrow.push_back(make_float2(0.2, 0.6));
		turnArrow.push_back(make_float2(0.2, 0.3));
		turnArrowParts.push_back(turnArrow);

		turnArrow.clear();
		turnArrow.push_back(make_float2(0, 0.3));
		turnArrow.push_back(make_float2(0.2, 0));
		turnArrow.push_back(make_float2(0.4, 0.3));
		turnArrowParts.push_back(turnArrow);
	};

	~InfoGuideRenderable()
	{
	};

	bool globalGuideOn = false; //currently only used for label count
};
#endif