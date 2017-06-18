#ifndef LABELVOLUMEPROCESSOR_H
#define LABELVOLUMEPROCESSOR_H
#include <memory>
#include "Processor.h"

class ScreenMarker;
class VolumeCUDA;
class RayCastingParameters;

class LabelVolumeProcessor :public Processor
{
public:
	LabelVolumeProcessor(std::shared_ptr<VolumeCUDA> _v){
		labelVolume = _v;
	};
	~LabelVolumeProcessor(){};
	std::shared_ptr<RayCastingParameters> rcp;
private:
	std::shared_ptr<ScreenMarker> sm;
	std::shared_ptr<VolumeCUDA> labelVolume;
public:
	void setScreenMarker(std::shared_ptr<ScreenMarker> _sm){ sm = _sm; }

	bool process(float modelview[16], float projection[16], int winW, int winH) override;

	void resize(int width, int height)override;

};
#endif