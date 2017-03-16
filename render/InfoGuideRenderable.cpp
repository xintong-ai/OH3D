#include "InfoGuideRenderable.h"
#include "GLMatrixManager.h"
#include "ViewpointEvaluator.h"
#include "GLWidget.h"


#include <iostream>
#include <algorithm>
using namespace std;

void InfoGuideRenderable::init()
{
}


void InfoGuideRenderable::draw(float modelview[16], float projection[16])
{
	RecordMatrix(modelview, projection);

	if (!visible)
		return;

	if (matrixMgr->justChanged){// !!! note !!! when using VR device, justChanged is not enough to detect matrix change
		if (globalGuideOn){
			ve->computeCubeEntropy(matrixMgr->getEyeInLocal(), matrixMgr->getViewVecInLocal(), matrixMgr->getUpInLocal());
		}
		else{	
			ve->computeCubeEntropy(matrixMgr->getEyeInLocal(), matrixMgr->getViewVecInLocal(), matrixMgr->getUpInLocal(), VPMethod::Tao09Detail);
		}
		matrixMgr->justChanged = false;

		//cout << "cube info " << ve->cubeInfo[0] << " " << ve->cubeInfo[1] << " " << ve->cubeInfo[2] << " " << ve->cubeInfo[3] << " " << ve->cubeInfo[4] << " " << ve->cubeInfo[5] << endl;
	}

	int2 winSize = actor->GetWindowSize();

	if (globalGuideOn){
	//if (ve->useLabelCount){
		float maxInfo = *std::max_element(ve->cubeInfo.begin(), ve->cubeInfo.begin() + ve->cubeInfo.size());

		float infoThr = 0.001;
		if (maxInfo > infoThr){
			drawLocalGuide(winSize);
		}
		else{
			if (ve->optimalEyeValid){
				drawGlobalGuide(modelview, projection, winSize);
			}
		}
	}
	else{
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


	glColor4f(0.89f, 0.39f, 0.26f, 0.5f);

	//float maxInfo = *std::max_element(ve->cubeInfo.begin(), ve->cubeInfo.begin() + ve->cubeInfo.size());
	int maxInd = std::max_element(ve->cubeInfo.begin(), ve->cubeInfo.begin() + ve->cubeInfo.size()) - ve->cubeInfo.begin();
	float maxInfo = ve->cubeInfo[maxInd];

	float thrRatio = 1.1;
	const float tipRatio = 0.1;
	const float sideRatio = 0.2;
	if (maxInfo > thrRatio*ve->cubeInfo[0]){
		if (maxInd == 1){

			glLineWidth(40);

			glBegin(GL_LINE_STRIP);
			for (int j = 0; j < turnArrowParts[0].size(); j++){
				glVertex2f(turnArrowParts[0][j].x * 120 + winSize.x*0.4, turnArrowParts[0][j].y * 120 + winSize.y*0.7);
			}
			glEnd();

			glLineWidth(4);

			glBegin(GL_TRIANGLES);
			for (int j = 0; j < turnArrowParts[1].size(); j++){
				glVertex2f(turnArrowParts[1][j].x * 120 + winSize.x*0.4, turnArrowParts[1][j].y * 120 + winSize.y*0.7);
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
	
	glColor4f(0.89f, 0.89f, 0.26f, 0.5f);

	float3 vecToOpti = normalize(ve->optimalEyeInLocal - matrixMgr->getEyeInLocal());
	float3 upVec = matrixMgr->getUpInLocal();
	
	float3 viewVec = matrixMgr->getViewVecInLocal();
	float3 sideVec = cross(viewVec, upVec);
	float x = dot(vecToOpti, sideVec), y = dot(vecToOpti, upVec), z = dot(vecToOpti, viewVec);

	float3 imagineXVec;
	if (dot(vecToOpti, upVec) < 0.9){
		imagineXVec = normalize(cross(vecToOpti, upVec));
	}
	else{
		imagineXVec = sideVec;
	}
	
	float3 startpos = matrixMgr->getEyeInLocal() + viewVec * 5;

	float arrowLength =2;

	//vecToOpti = arrowLength * normalize(vecToOpti);

	if (z > 0){
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glLoadMatrixf(projection);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glLoadMatrixf(modelview);

		//float centerxRatio = 0.5, centeryRatio = 0.25;

		//cout <<endl;
		glBegin(GL_POLYGON);
		for (int i = 0; i < globalGuideArrow.size(); i++){
			float3 p = startpos + globalGuideArrow[i].x*imagineXVec* arrowLength + globalGuideArrow[i].y*vecToOpti*arrowLength;
			glVertex3f(p.x, p.y, p.z);
			//cout << "p " << p.x << " " << p.y << " " << p.z << endl;
		}
		glEnd();


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


		glColor4f(0.89f, 0.89f, 0.26f, 0.5f);

		//for (int i = 0; i < turnArrowParts.size(); i++){
		//	glBegin(GL_POLYGON);
		//	for (int j = 0; j < turnArrowParts[i].size(); j++){
		//		glVertex2f(turnArrowParts[i][j].x * 120 + winSize.x*0.4, turnArrowParts[i][j].y * 120 + winSize.y*0.7);
		//	}
		//	glEnd();	
		//}

		glLineWidth(40);

		glBegin(GL_LINE_STRIP);
		for (int j = 0; j < turnArrowParts[0].size(); j++){
			glVertex2f(turnArrowParts[0][j].x * 120 + winSize.x*0.4, turnArrowParts[0][j].y * 120 + winSize.y*0.7);
		}
		glEnd();

		glLineWidth(4);

		glBegin(GL_TRIANGLES);
		for (int j = 0; j < turnArrowParts[1].size(); j++){
			glVertex2f(turnArrowParts[1][j].x * 120 + winSize.x*0.4, turnArrowParts[1][j].y * 120 + winSize.y*0.7);
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
