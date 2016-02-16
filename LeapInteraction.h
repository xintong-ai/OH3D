#ifndef LEAP_INTERACTION_H
#define LEAP_INTERACTION_H
#include "leap.h"

//using namespace Leap;

inline float Clamp(float v)
{
	v = v > 1.0f ? 1.0f : v;
	v = v < 0.0f ? 0.0f : v;
	return v;
}

inline Leap::Vector Clamp(Leap::Vector v)
{
	return Leap::Vector(Clamp(v.x), Clamp(v.y), Clamp(v.z));
}

inline Leap::Vector NormlizePoint(Leap::Vector p)
{
	return Clamp(Leap::Vector((p.x + 50) * 0.01 , (p.y - 150) * 0.01, (p.z + 50) * 0.01));
}

inline void GetAbsoluteRectangle(Leap::Frame frame, Leap::Vector &origin, Leap::Vector &point1, Leap::Vector &point2)
{
	//Leap::Hand leftHand = frame.hands().leftmost();
	Leap::Hand hand = frame.hands().leftmost();
	//Leap::Vector dirLeft = leftHand.direction().normalized();
	//Leap::Vector zDir = leftHand.palmNormal().normalized();
	//Leap::Vector yDir = zDir.cross(dirLeft).normalized();
	//Leap::Vector xDir = yDir.cross(zDir);
	//
	//Leap::Vector rightNormal = rightHand.palmNormal().normalized();
	//planeNormal.x = rightNormal.dot(dirLeft);
	//planeNormal.y = rightNormal.dot(yDir);
	//planeNormal.z = rightNormal.dot(palmNormalLeft);
	//planeNormal = planeNormal.normalized();

	Leap::Vector center = hand.palmPosition();
	point1 = hand.fingers().fingerType(Leap::Finger::Type::TYPE_MIDDLE).frontmost().tipPosition();
	//point2 = hand.fingers().fingerType(Leap::Finger::Type::TYPE_THUMB).frontmost().tipPosition();

	Leap::Vector normal = hand.palmNormal().normalized();
	Leap::Vector vec1 = point1 - center;
	Leap::Vector dir1 = vec1.normalized();
	Leap::Vector dir2 = normal.cross(vec1).normalized();
	origin = center - dir1 * 40 - dir2 * 40;
	point1 = center + dir1 * 40 - dir2 * 40;
	point2 = center - dir1 * 40 + dir2 * 40;


	//point1 = (point1 - origin).normalized() * 10 + origin;
	//point2 = (point2 - origin).normalized() * 10 + origin;
	origin = NormlizePoint(origin);
	point1 = NormlizePoint(point1);
	point2 = NormlizePoint(point2);

	//cout<<"**origin:"<<origin.x<<","<<origin.y<<","<<origin.z<<endl;
	//cout<<"**point1:"<<point1.x<<","<<point1.y<<","<<point1.z<<endl;
	//cout<<"**point2:"<<point2.x<<","<<point2.y<<","<<point2.z<<endl;


	//cout<<"**origin:"<<origin.x<<","<<origin.y<<","<<origin.z<<endl;
	//cout<<"**point1:"<<point1.x<<","<<point1.y<<","<<point1.z<<endl;
	//cout<<"**point2:"<<point2.x<<","<<point2.y<<","<<point2.z<<endl;
}


//double origin[3], double xdir[3], double ydir[3], double zdir[3])
inline void GetSpace(Leap::Frame frame, Leap::Vector &origin, Leap::Vector &xDir, Leap::Vector &yDir, Leap::Vector &zDir)
{
	Leap::Hand hand = frame.hands().leftmost();
	Leap::Vector dirLeft = hand.direction().normalized();
	xDir = hand.palmNormal().normalized();
	zDir = dirLeft.cross(xDir).normalized();
	yDir = zDir.cross(xDir);

	origin = frame.hands().leftmost().fingers().fingerType(Leap::Finger::Type::TYPE_MIDDLE).frontmost().bone(Leap::Bone::Type::TYPE_PROXIMAL).prevJoint();//hand.palmPosition();
	//origin = hand.palmPosition();

	//cout<<"**origin:"<<origin.x<<","<<origin.y<<","<<origin.z<<endl;
	//cout<<"**point1:"<<point1.x<<","<<point1.y<<","<<point1.z<<endl;
	//cout<<"**point2:"<<point2.x<<","<<point2.y<<","<<point2.z<<endl;
}


inline Leap::Vector RelativePalm3DLoc(Leap::Frame frame, Leap::Vector p)
{
	Leap::Vector o, xDir, yDir, zDir;
	GetSpace(frame, o, xDir, yDir, zDir);

	Leap::Hand leftHand = frame.hands().leftmost();
	Leap::Vector palmCenter = leftHand.palmPosition();
	float spaceSide = leftHand.palmWidth() * 2;
	//Leap::Vector dir = leftHand.direction().normalized();
	//Leap::Finger middleFinger = leftHand.fingers().fingerType(Leap::Finger::Type::TYPE_MIDDLE).frontmost();
	//Leap::Vector middleFingerTip = middleFinger.stabilizedTipPosition();
	//Leap::Vector palmNormal = leftHand.palmNormal();
	//Leap::Vector yDir = palmNormal.cross(dir).normalized();
	//cout<<"xDir:"<<xDir.x<<","<<xDir.y<<","<<xDir.z<<endl;
	//cout<<"yDir:"<<yDir.x<<","<<yDir.y<<","<<yDir.z<<endl;
	//cout<<"zDir:"<<zDir.x<<","<<zDir.y<<","<<zDir.z<<endl;

	Leap::Vector origin = palmCenter + xDir * spaceSide * 0.5 - zDir * spaceSide * 0.8 - yDir * spaceSide * 0.5;

	//	Leap::Vector pointerTip = rightHand.fingers().fingerType(Leap::Finger::Type::TYPE_INDEX).frontmost().tipPosition();
	//Leap::Vector vecPalmCenter2Tip = p - palmCenter;
	//float dist2Palm = vecPalmCenter2Tip.dot(xDir);
	//http://stackoverflow.com/questions/9605556/how-to-project-a-3d-point-to-a-3d-plane
	//Leap::Vector projTip = vecPalmCenter2Tip - dist2Palm * xDir + palmCenter;

	//Leap::Vector vecOrigin2ProjTip = projTip - origin;
	Leap::Vector ret;
	if(spaceSide < FLT_EPSILON)
		return Leap::Vector(0.0, 0.0, 0.0);
	Leap::Vector vecOrigin2Tip = p - origin;
	ret.x = vecOrigin2Tip.dot(xDir) / spaceSide;
	ret.y = vecOrigin2Tip.dot(yDir) / spaceSide;
	ret.z = vecOrigin2Tip.dot(zDir) / spaceSide;

	//cout<<"ret:"<<ret.x<<","<<ret.y<<","<<ret.z<<endl;
	return ret;
}


inline void GetTool(Leap::Frame frame, Leap::Vector &origin, Leap::Vector &dir)
{
	// Get tools
	const Leap::ToolList tools = frame.tools();
	//if(tools.isEmpty())
	//	return;
	const Leap::Tool tool = tools.leftmost();
	//std::cout << std::string(2, ' ') <<  "Tool, id: " << tool.id()
	//	<< ", position: " << tool.tipPosition()
	//	<< ", direction: " << tool.direction() << std::endl;

	Leap::Finger pointingFinger = frame.hands().rightmost().fingers().fingerType(Leap::Finger::Type::TYPE_INDEX).frontmost();
	origin = Clamp(RelativePalm3DLoc(frame, pointingFinger.tipPosition()));


	dir = pointingFinger.direction().normalized();
}

inline int GetGesture(Leap::Frame frame)
{
	const Leap::GestureList gestures = frame.gestures();
	int ret = 0;
	for (int g = 0; g < gestures.count(); ++g) {
		Leap::Gesture gesture = gestures[g];

		switch (gesture.type()) {
		case Leap::Gesture::TYPE_CIRCLE:
			{
				//CircleGesture circle = gesture;
				//std::string clockwiseness;

				//if (circle.pointable().direction().angleTo(circle.normal()) <= PI/2) {
				//  clockwiseness = "clockwise";
				//} else {
				//  clockwiseness = "counterclockwise";
				//}

				//// Calculate angle swept since last frame
				//float sweptAngle = 0;
				//if (circle.state() != Gesture::STATE_START) {
				//  CircleGesture previousUpdate = CircleGesture(controller.frame(1).gesture(circle.id()));
				//  sweptAngle = (circle.progress() - previousUpdate.progress()) * 2 * PI;
				//}
				//std::cout << std::string(2, ' ')
				//          << "Circle id: " << gesture.id()
				//          << ", state: " << stateNames[gesture.state()]
				//          << ", progress: " << circle.progress()
				//          << ", radius: " << circle.radius()
				//          << ", angle " << sweptAngle * RAD_TO_DEG
				//          <<  ", " << clockwiseness << std::endl;
				break;
			}
		case Leap::Gesture::TYPE_SWIPE:
			{
				//SwipeGesture swipe = gesture;
				//std::cout << std::string(2, ' ')
				//  << "Swipe id: " << gesture.id()
				//  << ", state: " << stateNames[gesture.state()]
				//  << ", direction: " << swipe.direction()
				//  << ", speed: " << swipe.speed() << std::endl;
				break;
			}
		case Leap::Gesture::TYPE_KEY_TAP:
			{
				//KeyTapGesture tap = gesture;
				//std::cout << std::string(2, ' ')
				//	<< "Key Tap id: " << gesture.id()
				//	<< ", state: " << stateNames[gesture.state()]
				//<< ", position: " << tap.position()
				//	<< ", direction: " << tap.direction()<< std::endl;'
				ret = 1;
				break;
			}
		case Leap::Gesture::TYPE_SCREEN_TAP:
			{
				//ScreenTapGesture screentap = gesture;
				//std::cout << std::string(2, ' ')
				//  << "Screen Tap id: " << gesture.id()
				//  << ", state: " << stateNames[gesture.state()]
				//  << ", position: " << screentap.position()
				//  << ", direction: " << screentap.direction()<< std::endl;
				break;
			}
		default:
			//std::cout << std::string(2, ' ')  << "Unknown gesture type." << std::endl;
			break;
		}
	}
	return ret;
}

inline void GetTwoPoints(Leap::Frame frame, Leap::Vector &point1, Leap::Vector &point2)
{
	Leap::Vector thumbTip = frame.hands().rightmost().fingers().fingerType(Leap::Finger::Type::TYPE_THUMB).frontmost().tipPosition();
	Leap::Vector indexTip = frame.hands().rightmost().fingers().fingerType(Leap::Finger::Type::TYPE_INDEX).frontmost().tipPosition();
	point1 = Clamp(RelativePalm3DLoc(frame, thumbTip));
	point2 = Clamp(RelativePalm3DLoc(frame, indexTip));
}

inline void GetFingers(Leap::Hand hand, Leap::Vector &thumbTip, Leap::Vector &indexTip, Leap::Vector &indexDir)
{
	thumbTip = hand.fingers().fingerType(Leap::Finger::Type::TYPE_THUMB).frontmost().tipPosition();
	indexTip = hand.fingers().fingerType(Leap::Finger::Type::TYPE_INDEX).frontmost().tipPosition();
	indexDir = hand.fingers().fingerType(Leap::Finger::Type::TYPE_INDEX).frontmost().direction().normalized();
}

inline void GetSkeletonHand(Leap::Hand hand, std::vector<std::vector<Leap::Vector>> &fingerJoints, 
	std::vector<Leap::Vector> &palm, float &sphereRadius)
{
	//static const float kfJointRadiusScale = 0.75f;
	//static const float kfBoneRadiusScale = 0.5f;
	//static const float kfPalmRadiusScale = 1.15f;

	//LeapUtilGL::GLAttribScope colorScope( GL_CURRENT_BIT );

	const Leap::Vector vPalm = hand.palmPosition();
	const Leap::Vector vPalmDir = hand.direction();
	const Leap::Vector vPalmNormal = hand.palmNormal();
	const Leap::Vector vPalmSide = vPalmDir.cross(vPalmNormal).normalized();

	const float fThumbDist = hand.fingers()[Leap::Finger::TYPE_THUMB].bone(Leap::Bone::TYPE_METACARPAL).prevJoint().distanceTo(hand.palmPosition());
	const Leap::Vector vWrist = vPalm - fThumbDist*(vPalmDir*0.90f + (hand.isLeft() ? -1.0f : 1.0f)*vPalmSide*0.38f);

	Leap::FingerList fingers = hand.fingers();

	float fRadius = 0.0f;
	Leap::Vector vCurBoxBase;
	Leap::Vector vLastBoxBase = vWrist;

	for (int i = 0, ei = fingers.count(); i < ei; i++) {
		const Leap::Finger& finger = fingers[i];
		fRadius = finger.width() * 0.5f;

		std::vector<Leap::Vector> oneFinger;
		// draw individual fingers
		for (int j = Leap::Bone::TYPE_METACARPAL; j <= Leap::Bone::TYPE_DISTAL; j++) {
			const Leap::Bone& bone = finger.bone(static_cast<Leap::Bone::Type>(j));

			// don't draw metacarpal, a box around the metacarpals is draw instead.
			if (j == Leap::Bone::TYPE_METACARPAL) {
				// cache the top of the metacarpal for the next step in metacarpal box
				vCurBoxBase = bone.nextJoint();
			} else {
				//glColor4fv(vBoneColor);
				//drawCylinder(kStyle_Solid, bone.prevJoint(), bone.nextJoint(), kfBoneRadiusScale*fRadius);
				//glColor4fv(vJointColor);
				//drawSphere(kStyle_Solid, bone.nextJoint(), kfJointRadiusScale*fRadius);
			}
			oneFinger.push_back(bone.nextJoint());
		}
		fingerJoints.push_back(oneFinger);

		// draw segment of metacarpal box
		//glColor4fv(vBoneColor);
		//drawCylinder(kStyle_Solid, vCurBoxBase, vLastBoxBase, kfBoneRadiusScale*fRadius);
		//glColor4fv(vJointColor);
		//drawSphere(kStyle_Solid, vCurBoxBase, kfJointRadiusScale*fRadius);
		vLastBoxBase = vCurBoxBase;

		palm.push_back(vCurBoxBase);
	}

	// close the metacarpal box
	fRadius = fingers[Leap::Finger::TYPE_THUMB].width() * 0.5f;
	vCurBoxBase = vWrist;
	//glColor4fv(vBoneColor);
	//drawCylinder(kStyle_Solid, vCurBoxBase, vLastBoxBase, kfBoneRadiusScale*fRadius);
	//glColor4fv(vJointColor);
	//drawSphere(kStyle_Solid, vCurBoxBase, kfJointRadiusScale*fRadius);
	palm.push_back(vCurBoxBase);
	// draw palm position
	//glColor4fv(vJointColor);
	//drawSphere(kStyle_Solid, vPalm, kfPalmRadiusScale*fRadius);
	sphereRadius = hand.sphereRadius();
}


#endif