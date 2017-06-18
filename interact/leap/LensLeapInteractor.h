#ifndef LENSLEAPINTERACTOR_H
#define LENSLEAPINTERACTOR_H
#include <vector>
#include "LeapInteractor.h"

class Lens;
class Particle;
class LensLeapInteractor :public LeapInteractor
{
private:


	//used for Leap
	void ChangeLensCenterbyLeap(Lens *l, float3 p);
	void ChangeLensCenterbyTransferredLeap(Lens *l, float3 p);
	float3 prevPos, prevPos2, prevPointOfLens;
	float preForce;

	//should be replaced to connect to renderer
	bool highlightingCuboidFrame = false;
	bool highlightingCenter = false;
	bool highlightingMajorSide = false;
	bool highlightingMinorSide = false;


	inline bool outOfDomain(float3 p, float3 posMin, float3 posMax)
	{
		float3 difmin = p - posMin, difmax = p - posMax;

		return min(difmin.x, min(difmin.y, difmin.z))<0 || max(difmax.x, max(difmax.y, difmax.z))>0;
	}

	std::vector<Lens*> *lenses = 0;
	std::shared_ptr<Particle> leapFingerIndicators;

	float3 GetTransferredLeapPos(float3 p);

public:
	LensLeapInteractor(){};
	~LensLeapInteractor(){};

	void SetLenses(std::vector<Lens*> *_lenses){ lenses = _lenses; }
	void SetFingerIndicator(std::shared_ptr<Particle> p){ leapFingerIndicators = p; }


	bool SlotRightHandChanged(float3 thumpLeap, float3 indexLeap, float3 middleLeap, float3 ringLeap, float &f) override;
	void SlotTwoHandChanged(float3 l, float3 r) override;

	//bool changeLensWhenRotateData = true; //view dependant or not
	bool isSnapToGlyph = false;
	bool isSnapToFeature = false;



};
#endif