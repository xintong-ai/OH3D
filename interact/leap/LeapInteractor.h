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


	virtual bool SlotRightHandChanged(float3 thumpLeap, float3 indexLeap, float3 middleLeap, float3 ringLeap, float &force){ return false; }; //force is used for lens force change. this is not a good design and will be modified later

	virtual bool SlotLeftHandChanged(float3 thumpLeap, float3 indexLeap, float3 middleLeap, float3 ringLeap){ return false; }; 

	virtual void SlotTwoHandChanged(float3 l, float3 r){};



protected:
	GLWidget* actor;

	inline float3 GetNormalizedLeapPos(float3 p)//leap motion coord normalized into [0,1], with translation based on empirical values. see https://developer.leapmotion.com/documentation/csharp/devguide/Leap_Coordinate_Mapping.html
	{
		float3 leapPos;
		leapPos.x = clamp((p.x + 117.5) / 235.0, 0.0f, 1.0f);
		leapPos.y = clamp((p.y - 82.5) / 235.0, 0.0f, 1.0f);
		leapPos.z = clamp((p.z + 73.5f) / 147.0f, 0.0f, 1.0f);
		return leapPos;
	}

};
#endif