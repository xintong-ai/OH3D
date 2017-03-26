#include "InfoGuideRenderable.h"
#include "GLMatrixManager.h"
#include "ViewpointEvaluator.h"
#include "GLWidget.h"


#include <iostream>
#include <algorithm>
using namespace std;

QVector3D vectorTransform(QVector3D v, QMatrix3x3 m)
{
	QVector3D transformed = QVector3D();
	transformed.setX(((m.operator()(0, 0))*v.x()) + ((m.operator()(0, 1))*v.y()) + ((m.operator()(0, 2))*v.z()));
	transformed.setY(((m.operator()(1, 0))*v.x()) + ((m.operator()(1, 1))*v.y()) + ((m.operator()(1, 2))*v.z()));
	transformed.setZ(((m.operator()(2, 0))*v.x()) + ((m.operator()(2, 1))*v.y()) + ((m.operator()(2, 2))*v.z()));
	return transformed;
}

void InfoGuideRenderable::init()
{
}


void InfoGuideRenderable::draw(float modelview[16], float projection[16])
{
	RecordMatrix(modelview, projection);

	if (!visible)
		return;

	float3 eyeInLocal = matrixMgr->getEyeInLocal();
	float3 viewDir = matrixMgr->getViewVecInLocal();
	float3 upDir = matrixMgr->getUpInLocal();

	int2 winSize = actor->GetWindowSize();

	const float locationChangeThr = 1;
	const float vecChangeThr = 0.99;
	const float disThr = 3;
	if (globalGuideOn && length(matrixMgr->getEyeInLocal() - ve->optimalEyeInLocal)>disThr){  //should do global guide
		if (length(eyeInLocal - storedEyeLocal) > locationChangeThr || abs(dot(viewDir, storedViewDir)) < vecChangeThr || abs(dot(upDir, storedUpVector)) < vecChangeThr){

			transp = maxTransparency;
			startTime = std::clock();
			storedEyeLocal = eyeInLocal;
			storedViewDir = viewDir;
			storedUpVector = upDir;
		}
		else{
			double past = (std::clock() - startTime) / (double)CLOCKS_PER_SEC;
			if (past>durationFix){
				if (past < durationFix + durationDecrease)
				{
					transp = (durationFix + durationDecrease - past) / durationDecrease*maxTransparency;
				}
				else{
					transp = 0;
				}
			}
		}
		drawGlobalGuide(modelview, projection, winSize);

	}
	else{// should do local guide

		if (isAlwaysLocalGuide){ //for fps test
			ve->computeCubeEntropy(matrixMgr->getEyeInLocal(), matrixMgr->getViewVecInLocal(), matrixMgr->getUpInLocal(), ve->currentMethod);

			if (length(eyeInLocal - storedEyeLocal) > locationChangeThr || abs(dot(viewDir, storedViewDir)) < vecChangeThr || abs(dot(upDir, storedUpVector)) < vecChangeThr){

				//cout << "cube info " << ve->cubeInfo[0] << " " << ve->cubeInfo[1] << " " << ve->cubeInfo[2] << " " << ve->cubeInfo[3] << " " << ve->cubeInfo[4] << " " << ve->cubeInfo[5] << endl;

				transp = maxTransparency;
				startTime = std::clock();
				storedEyeLocal = eyeInLocal;
				storedViewDir = viewDir;
				storedUpVector = upDir;
			}
			else{
				double past = (std::clock() - startTime) / (double)CLOCKS_PER_SEC;
				if (past > durationFix){
					if (past < durationFix + durationDecrease)
					{
						transp = (durationFix + durationDecrease - past) / durationDecrease*maxTransparency;
					}
					else{
						transp = 0.1; //always draw
					}
				}
			}
		}
		else{
			if (length(eyeInLocal - storedEyeLocal) > locationChangeThr || abs(dot(viewDir, storedViewDir)) < vecChangeThr || abs(dot(upDir, storedUpVector)) < vecChangeThr){
				ve->computeCubeEntropy(matrixMgr->getEyeInLocal(), matrixMgr->getViewVecInLocal(), matrixMgr->getUpInLocal(), ve->currentMethod);

				//cout << "cube info " << ve->cubeInfo[0] << " " << ve->cubeInfo[1] << " " << ve->cubeInfo[2] << " " << ve->cubeInfo[3] << " " << ve->cubeInfo[4] << " " << ve->cubeInfo[5] << endl;

				transp = maxTransparency;
				startTime = std::clock();
				storedEyeLocal = eyeInLocal;
				storedViewDir = viewDir;
				storedUpVector = upDir;
			}
			else{
				double past = (std::clock() - startTime) / (double)CLOCKS_PER_SEC;
				if (past > durationFix){
					if (past < durationFix + durationDecrease)
					{
						transp = (durationFix + durationDecrease - past) / durationDecrease*maxTransparency;
					}
					else{
						transp = 0;
					}
				}
			}
		}
		drawLocalGuide(winSize);

	}
}


void InfoGuideRenderable::drawLocalGuide(int2 winSize)
{
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, winSize.x - 1, 0.0, winSize.y - 1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT);
	glLineWidth(4);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


	//glColor4f(0.89f, 0.39f, 0.26f, transp);
	glColor4f(0.29f, 0.39f, 0.86f, transp);


	int maxInd = std::max_element(ve->cubeInfo.begin(), ve->cubeInfo.begin() + ve->cubeInfo.size()) - ve->cubeInfo.begin();
	float maxInfo = ve->cubeInfo[maxInd];

	float thrRatio = 1.05;
	const float tipRatio = 0.1;
	const float sideRatio = 0.2;
	if (maxInfo > thrRatio*ve->cubeInfo[0]){
		if (maxInd == 1){
			glBegin(GL_TRIANGLE_STRIP);
			float2 move = make_float2(winSize.x*0.3, winSize.y*0.75);
			float2 scale = make_float2(winSize.x*0.4, winSize.y*0.3); //y range of turnArrowParts only covers [0,0.5]
			for (int j = 0; j < turnArrowParts[0].size(); j++){
				glVertex2f(turnArrowParts[0][j].x * scale.x + move.x, turnArrowParts[0][j].y *scale.y + move.y);
			}
			glEnd();
			glBegin(GL_TRIANGLES);
			for (int j = 0; j < turnArrowParts[1].size(); j++){
				glVertex2f(turnArrowParts[1][j].x * scale.x + move.x, turnArrowParts[1][j].y *scale.y + move.y);
			}
			glEnd();
		}
		else if(maxInd == 2){
			glBegin(GL_TRIANGLES);
			glVertex2f(winSize.x*tipRatio, winSize.y*0.5);
			glVertex2f(winSize.x*tipRatio * 2, winSize.y*sideRatio);
			glVertex2f(winSize.x*tipRatio * 2, winSize.y*(1 - sideRatio));
			glEnd();
		}
		else if (maxInd == 3){
			glBegin(GL_TRIANGLES);
			glVertex2f(winSize.x*(1 - tipRatio), winSize.y*0.5);
			glVertex2f(winSize.x*(1 - tipRatio * 2), winSize.y*sideRatio);
			glVertex2f(winSize.x*(1 - tipRatio * 2), winSize.y*(1 - sideRatio));
			glEnd();
		}
		else if (maxInd == 4){
			glBegin(GL_TRIANGLES);
			glVertex2f(winSize.x*0.5, winSize.y*(1 - tipRatio));
			glVertex2f(winSize.x*sideRatio, winSize.y*(1 - 2 * tipRatio));
			glVertex2f(winSize.x*(1 - sideRatio), winSize.y*(1 - 2 * tipRatio));
			glEnd();
		}
		else if (maxInd == 5){
			glBegin(GL_TRIANGLES);
			glVertex2f(winSize.x*0.5, winSize.y*tipRatio);
			glVertex2f(winSize.x*sideRatio, winSize.y * 2 * tipRatio);
			glVertex2f(winSize.x*(1 - sideRatio), winSize.y * 2 * tipRatio);
			glEnd();
		}
	}

	glDisable(GL_BLEND);

	glPopAttrib();

	//restore the original 3D coordinate system
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}


void InfoGuideRenderable::drawGlobalGuide(float modelview[16], float projection[16], int2 winSize)
{
	//glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT);
	glLineWidth(14);

//	glDisable(GL_DEPTH_TEST);
	glDepthFunc(GL_ALWAYS);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	glColor4f(0.89f, 0.89f, 0.26f, transp);

	float3 vecToOpti = normalize(ve->optimalEyeInLocal - matrixMgr->getEyeInLocal());
	float3 upVec = matrixMgr->getUpInLocal();
	
	float3 viewVec = matrixMgr->getViewVecInLocal();
	float z = dot(vecToOpti, viewVec);
	
	float3 startpos = matrixMgr->getEyeInLocal() + viewVec * 5 - upVec*3;

	float arrowLength = 3;

	if (z > 0){
		QMatrix4x4 rotateMat;
		float3 orientation = glArrow.orientation;

		float sinTheta = length(cross(orientation, vecToOpti));
		float cosTheta = dot(orientation, vecToOpti);
		if (sinTheta<0.00001)
		{
			rotateMat.setToIdentity();
		}
		else
		{
			float3 axis = normalize(cross(orientation, vecToOpti));
			float theta;
			if (cosTheta>0)
				theta = asin(sinTheta) * 180 / 3.1415926;
			else 
				theta = 180 - asin(sinTheta) * 180 / 3.1415926;
			rotateMat.rotate(theta, QVector3D(axis.x, axis.y, axis.z));
		}

		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glLoadMatrixf(projection);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glLoadMatrixf(modelview);


		QVector3D initPoint = rotateMat*QVector3D(glArrow.grids[0].x, glArrow.grids[0].y, glArrow.grids[0].z * arrowLength);

		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); glLineWidth(1);
		glBegin(GL_TRIANGLES);
		for (int i = 0; i < glArrow.grids.size(); i++){
			if (i % 18 < 9)
				glColor4f(0.79f, 0.89f, 0.26f, transp);
			else if (i % 18 < 15){
				glColor4f(0.89f, 0.79f, 0.06f, transp);
			}
			else{
				glColor4f(0.99f, 0.99f, 0.06f, transp);
			}
			QVector3D newp = rotateMat*QVector3D(glArrow.grids[i].x, glArrow.grids[i].y, glArrow.grids[i].z*arrowLength) - initPoint;
			float3 p = startpos + make_float3(newp.x(), newp.y(), newp.z());
			glVertex3f(p.x, p.y, p.z);
		}		
		glEnd();
		//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
	}
	else{
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(0.0, winSize.x - 1, 0.0, winSize.y - 1, -1, 1);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();

		glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glBegin(GL_TRIANGLE_STRIP);
		float2 move = make_float2(winSize.x*0.3, winSize.y*0.1);
		float2 scale = make_float2(winSize.x*0.4, winSize.y*0.3); //y range of turnArrowParts only covers [0,0.5]
		for (int j = 0; j < turnArrowParts[0].size(); j++){
			glVertex2f(turnArrowParts[0][j].x * scale.x + move.x, turnArrowParts[0][j].y *scale.y + move.y);
		}
		glEnd();
		glBegin(GL_TRIANGLES);
		for (int j = 0; j < turnArrowParts[1].size(); j++){
			glVertex2f(turnArrowParts[1][j].x * scale.x + move.x, turnArrowParts[1][j].y *scale.y + move.y);
		}
		glEnd();
		
		glDisable(GL_BLEND);

		glPopAttrib();

		//restore the original 3D coordinate system
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
		
	}
	glDisable(GL_BLEND);
	glPopAttrib();
	//glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);


}
