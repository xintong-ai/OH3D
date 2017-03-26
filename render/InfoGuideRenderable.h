#ifndef INFOGUIDE_RENDERABLE_H
#define INFOGUIDE_RENDERABLE_H
#include "Renderable.h"
#include "GLArrow.h"
#include <memory>
#include <vector>
#include <ctime>
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

	float3 storedEyeLocal = make_float3(0, 0, 0), storedViewDir = make_float3(0, 0, 0), storedUpVector = make_float3(0, 0, 0);
	GLArrow glArrow;

	const float maxTransparency = 0.6;
	float transp = 0.6;
	std::clock_t startTime;
	float durationFix = 2, durationDecrease = 1;

	bool globalGuideOn = false;

public:
	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	
	InfoGuideRenderable(std::shared_ptr<ViewpointEvaluator> _ve, std::shared_ptr<GLMatrixManager> m){
		ve = _ve;
		matrixMgr = m;
		
		std::vector<float2> turnArrow;
		turnArrow.clear();
		float sample = 10;
		for (int i = 0; i <sample; i++){
			float theta = 90 * (1.0*i) / (sample - 1)/180*3.1415926;
			float r1 = 0.75 + 0.2*(1.0*i) / (sample - 1), r2 = 1;
			turnArrow.push_back(make_float2(1 + cos(theta)*r1, sin(theta)*r1)/2);
			turnArrow.push_back(make_float2(1 + cos(theta)*r2, sin(theta)*r2)/2);
		}
		for (int i = sample-1; i>=3; i--){
			float theta = 90 * (1.0*i) / (sample - 1) / 180 * 3.1415926;
			float r1 = 0.70 + 0.25*(1.0*i) / (sample - 1), r2 = 0.95 + 0.05*(1.0*i) / (sample - 1);
			turnArrow.push_back(make_float2(1 - cos(theta)*r1, sin(theta)*r1) / 2);
			turnArrow.push_back(make_float2(1 - cos(theta)*r2, sin(theta)*r2) / 2);
		}
		turnArrowParts.push_back(turnArrow);

		float2 cur1 = turnArrow[turnArrow.size() - 2], cur2 = turnArrow[turnArrow.size() - 1];
		turnArrow.clear();
		turnArrow.push_back(cur1);
		turnArrow.push_back(cur2);
		turnArrow.push_back(make_float2(cur2.x, cur1.y));
		turnArrow.push_back(make_float2(0, cur1.y));
		turnArrow.push_back(make_float2((cur2.x + cur1.x)/2, 0));
		turnArrow.push_back(make_float2(cur2.x + cur1.x, cur1.y));
		turnArrowParts.push_back(turnArrow);
	};

	~InfoGuideRenderable()
	{
	};

	void changeWhetherGlobalGuideMode(bool b){
		globalGuideOn = b;
		startTime = std::clock();
		transp = maxTransparency;
	};

	bool isAlwaysLocalGuide = false;

};
#endif