#include "LeapListener.h"
#include "LeapInteraction.h"
#include <qvector3d.h>

inline QVector3D Leap2QVector(Leap::Vector v)
{
	return QVector3D(v.x, v.y, v.z);
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

	if(timer->elapsed() > 16.7)
	{
		Leap::Frame f = ctl.frame();
		setObjectName(QString::number(f.id()));
		// emits objectNameChanged(QString)
		//emit translate2(SimpleTranslate(f));


		//Leap::Vector center, xDir, yDir, zDir;
		//GetSpace(f, center, xDir, yDir, zDir);
		//emit UpdateCamera(Leap2QVector(center), Leap2QVector(xDir), Leap2QVector(yDir), Leap2QVector(zDir));

		//Leap::Vector toolTip, toolDir;
		//GetTool(f, toolTip, toolDir);
		//emit UpdatePlane(Leap2QVector(toolTip), Leap2QVector(toolDir));

		//Leap::Vector point1, point2;
		//GetTwoPoints(f, point1, point2);
		//emit UpdateLine(Leap2QVector(point1), Leap2QVector(point2));

		Leap::Vector thumbTip, indexTipLeft, indexTipRight, indexDir;
		Leap::Hand rightMostHand = f.hands().rightmost();
		Leap::Hand leftMostHand = f.hands().leftmost();
		//std::cout << "num of hands:" << f.hands().count() << std::endl;
		if (1 == f.hands().count())
		{
			GetFingers(rightMostHand, thumbTip, indexTipRight, indexDir);
			emit UpdateHands(Leap2QVector(indexTipLeft), Leap2QVector(indexTipRight), 1);
		}
		else if (2 == f.hands().count()){
			//if (){//leftMostHand.isLeft()){
			GetFingers(leftMostHand, thumbTip, indexTipLeft, indexDir);
			GetFingers(rightMostHand, thumbTip, indexTipRight, indexDir);
			//emit UpdateTwoHands(Leap2QVector(thumbTip), Leap2QVector(indexTipLeft), Leap2QVector(indexDir));
			emit UpdateHands(Leap2QVector(indexTipLeft), Leap2QVector(indexTipRight), 2);
		}
		//	else{
		//		//emit UpdateRightHand(Leap2QVector(thumbTip), Leap2QVector(indexTipRight), Leap2QVector(indexDir));
		//	}
		//	//if ((thumbTip - indexTip).magnitude() < 15){
		//	//}
		//}

		//std::vector<std::vector<Leap::Vector>> fingers;
		//std::vector<Leap::Vector> palm ;
		//float sphereRadius;
		//if(leftMostHand.isLeft())
		//{
		//	GetSkeletonHand(leftMostHand, fingers, palm, sphereRadius);
		//	//		QVector<QVector<QVector3D>> tmp = Leap2QVector(fingers);
		//	emit UpdateSkeletonHand(Leap2QVector(fingers), Leap2QVector(palm), sphereRadius);
		//}
		//int gesture = GetGesture(f);
		////std::cout<<gesture<<std::endl;
		//emit UpdateGesture(gesture);

		timer->restart();
	}
}
