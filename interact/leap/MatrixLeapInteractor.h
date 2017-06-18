#ifndef MATRIXLEAPINTERACTOR_H
#define MATRIXLEAPINTERACTOR_H
#include <vector>
#include "LeapInteractor.h"

enum HAND_STATE {
	idle,
	movingLeft
};

class GLMatrixManager;
class MatrixLeapInteractor :public LeapInteractor
	//note that this class is a leap implementation of ImmersiveInteractor. It is not similar with RegularInteractor
{
private:

	////used for Leap
	//void ChangeLensCenterbyLeap(Lens *l, float3 p);
	//void ChangeLensCenterbyTransferredLeap(Lens *l, float3 p);
	//float3 GetTransferredLeapPos(float3 p);
	//float3 prevPos, prevPos2, prevPointOfLens;
	//float preForce;

	////should be replaced to connect to renderer
	//bool highlightingCuboidFrame = false;
	//bool highlightingCenter = false;
	//bool highlightingMajorSide = false;
	//bool highlightingMinorSide = false;


	float3 lastPos;

	inline bool outOfDomain(float3 p, float3 posMin, float3 posMax)
	{
		float3 difmin = p - posMin, difmax = p - posMax;

		return min(difmin.x, min(difmin.y, difmin.z))<0 || max(difmax.x, max(difmax.y, difmax.z))>0;
	}

	HAND_STATE handState = idle;
	
	float3 targetUpVecInLocal = make_float3(0, 0, 1);	//note! the vector make_float3(0,0,1) may also be used in ImmersiveInteractor class

	std::shared_ptr<GLMatrixManager> matrixMgr;

	int frameCounter = 0;

	//the two functions are copied from ImmersiveInteractor.h
	void moveViewHorizontally(int d);
	void moveViewVertically(int d);
	void moveViewForwardBackward(int d);

public:
	MatrixLeapInteractor(std::shared_ptr<GLMatrixManager> m){ matrixMgr = m; };
	~MatrixLeapInteractor(){};

//	void SetLenses(std::vector<Lens*> *_lenses){ lenses = _lenses; }
	//void SetFingerIndicator(std::shared_ptr<Particle> p){ leapFingerIndicators = p; }

	bool SlotLeftHandChanged(float3 thumpLeap, float3 indexLeap, float3 middleLeap, float3 ringLeap) override;
	bool SlotRightHandChanged(float3 thumpLeap, float3 indexLeap, float3 middleLeap, float3 ringLeap, float &f) override;

};
#endif