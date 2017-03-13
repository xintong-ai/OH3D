#include "LeapListener.h"
#include "LeapInteraction.h"
#include <qvector3d.h>
#include <QVector3D>

inline QVector3D Leap2QVector(Leap::Vector v)
{
	return QVector3D(v.x, v.y, v.z);
}

inline float3 Leap2float3(Leap::Vector v)
{
	return make_float3(v.x, v.y, v.z);
}

inline QVector<QVector3D> Leap2QVector(std::vector<Leap::Vector> v)
{
	QVector<QVector3D> ret;
	for(int i = 0; i < v.size(); i++)	
		ret.push_back(Leap2QVector(v[i]));
	return ret;
}

inline QVector<QVector<QVector3D>> Leap2QVector(std::vector<std::vector<Leap::Vector>> v)
{
	QVector<QVector<QVector3D>> ret;
	for(int i = 0; i < v.size(); i++)	
		ret.push_back(Leap2QVector(v[i]));
	return ret;
}

void LeapListener::onFrame(const Leap::Controller & ctl)
{

//	if(timer->elapsed() > 16.7)
	if (timer->elapsed() > 50)

	{
		Leap::Frame f = ctl.frame();
		setObjectName(QString::number(f.id()));


		Leap::Vector thumbTipLeft, thumbTipRight, indexTipLeft, indexTipRight, indexDir;
		Leap::Hand rightMostHand = f.hands().rightmost();
		Leap::Hand leftMostHand = f.hands().leftmost();
		//std::cout << "num of hands:" << f.hands().count() << std::endl;
		if (1 == f.hands().count())
		{
			Leap::Vector middleTipRight, ringTipRight;
			GetFingers(rightMostHand, thumbTipRight, indexTipRight, middleTipRight, ringTipRight);
			//emit UpdateHandsNew(Leap2QVector(thumbTipRight), Leap2QVector(indexTipRight), QVector3D(0, 0, 0), QVector3D(0, 0, 0), Leap2QVector(middleTipRight), Leap2QVector(ringTipRight), 1);
			
			float force = 800;
			
			float3 rightThumbTip = Leap2float3(thumbTipRight);
			float3 rightIndexTip = Leap2float3(indexTipRight);
			float3 rightMiddleTip = Leap2float3(middleTipRight);
			float3 rightRingTip = Leap2float3(ringTipRight);

			float3 leftThumbTip = Leap2float3(thumbTipLeft);
			float3 leftIndexTip = Leap2float3(indexTipRight);
			float3 leftMiddleTip = Leap2float3(middleTipRight);
			float3 leftRingTip = Leap2float3(ringTipRight);

			for (auto interactor : interactors){
				interactor.second->SlotRightHandChanged(rightThumbTip, rightIndexTip, rightMiddleTip, rightRingTip, force);
				interactor.second->SlotLeftHandChanged(leftThumbTip, leftIndexTip, leftMiddleTip, leftRingTip);
			}
		}


		else if (2 == f.hands().count()){
			GetFingers(leftMostHand, thumbTipLeft, indexTipLeft, indexDir);
			GetFingers(rightMostHand, thumbTipRight, indexTipRight, indexDir);
		}

		timer->restart();
	}
}

void LeapListener::AddLeapInteractor(const char* name, void* r)
{
	interactors[name] = (LeapInteractor*)r;
}