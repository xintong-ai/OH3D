#ifndef LEAPINTERACTOR_H
#define LEAPINTERACTOR_H
#include <memory>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>

class GLWidget;

class LeapInteractor
{
public:
	LeapInteractor(){};
	~LeapInteractor(){};


	void SetActor(GLWidget* _actor) {
		actor = _actor;
	}


	virtual bool SlotOneHandChanged(float3 thumpLeap, float3 indexLeap, float3 middleLeap, float3 ringLeap, float &f){ return false; };

	virtual bool SlotOneHandChanged(float3 thumpLeap, float3 indexLeap, float3 middleLeap, float3 ringLeap, float4 &markerPos, float &valRight, float &f){ return false; };
	virtual void SlotTwoHandChanged(float3 l, float3 r){};

protected:
	GLWidget* actor;

//	int2 lastPt = make_int2(0, 0);


};
#endif