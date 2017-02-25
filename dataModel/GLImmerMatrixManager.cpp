#include <GLImmerMatrixManager.h>
#include <mouse/trackball.h>
#include <mouse/Rotation.h>
#include <qmatrix4x4.h>
#include <helper_math.h>
#include <fstream>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

float3 inline qvec3ToFloat3(QVector3D q){ return make_float3(q.x(), q.y(), q.z()); }
QVector3D inline float3ToQvec3(float3 f){ return QVector3D(f.x, f.y, f.z); }

GLImmerMatrixManager::GLImmerMatrixManager()
{
	viewMat.setToIdentity(); //identity view matrix is close to what VR device provided. it is equivalent to seeing from (0,0,0) to the direction of (0,0,-1)
	originalViewVecInLocal = QVector3D(0.0f, 0.0f, -1.0f); //corresponding to the identity view matrix
	originalUpVecInLocal = QVector3D(0.0f, 1.0f, 0.0f); //corresponding to the identity view matrix

	projAngle = 96.73;
	zNear = 0.1;
	zFar = 100;

	trackball = new Trackball();
	rot = new Rotation();

	

	//eyeInLocal = QVector3D(64, 109, 107); //eyeInWorld is decided by view matrix. identity view matrix actually gives (0,0,0) as eyeInWorld
	//eyeInLocal = QVector3D(100, -100, 100);
	eyeInLocal = make_float3(64, 109, 107);

	viewVecInLocal = make_float3(0.0f, 1.0f, 0.0f);

	rotMat.setToIdentity();
	float ang = acos(QVector3D::dotProduct(float3ToQvec3(viewVecInLocal), originalViewVecInLocal)) * 180.0 / M_PI;
	rotMat.rotate(ang, QVector3D::crossProduct(float3ToQvec3(viewVecInLocal), originalViewVecInLocal));
	//we want to replace originalViewVecInLocal by viewVecInLocal, which means in the coordinate, the vector whose coordinate is viewVecInLocal will be rotated to the position of originalViewVecInLocal

	//whenever rotMat is changed, viewVecInLocal and upVecInLocal may also be changed
	upVecInLocal =qvec3ToFloat3(rotMat.inverted()*originalUpVecInLocal);  //similarly, we want to find the vector whose coordinate  will be rotated to the position of originalUpVecInLocal

	//std::cout << "the new upVecInLocal: " << upVecInLocal.x() << " " << upVecInLocal.y() << " " << upVecInLocal.z() << std::endl;
	
	resetModelMat();
}

void GLImmerMatrixManager::resetModelMat()
{
	modelMat.setToIdentity();
	modelMat = modelMat * rotMat;
	modelMat.translate(-QVector3D(eyeInLocal.x, eyeInLocal.y, eyeInLocal.z));
}


void GLImmerMatrixManager::SetVol(float3 posMin, float3 posMax)
{
	dataMin = posMin;
	dataMax = posMax;
}


void GLImmerMatrixManager::GetModelViewMatrix(float mv[16])
{
	QMatrix4x4 m = viewMat*modelMat;
	m = m.transposed();
	m.copyDataTo(mv);
}

void GLImmerMatrixManager::GetProjection(float p[16], float width, float height)
{
	QMatrix4x4 m;
	m.setToIdentity();
	m.perspective(projAngle, (float)width / height, zNear, zFar);
	m = m.transposed();
	m.copyDataTo(p);
}

void GLImmerMatrixManager::moveWheel(float v)
{
	//QVector3D moveVecInLocal = QVector3D::crossProduct(upVecInLocal, QVector3D::crossProduct(float3ToQvec3(viewVecInLocal), upVecInLocal));
	//std::cout << moveVecInLocal.x() << " " << moveVecInLocal.y() << " " << moveVecInLocal.z() << std::endl;
	eyeInLocal = eyeInLocal + getMoveVecFromViewAndUp() * v / 200.0;
	resetModelMat();
}

void GLImmerMatrixManager::Rotate(float fromX, float fromY, float toX, float toY)
{
	*rot = trackball->rotate(fromX, fromY,
		toX, toY);
	float m[16];
	rot->matrix(m);
	
	QMatrix4x4 qm = QMatrix4x4(m).transposed();
	//rotMat = qm.inverted() * rotMat;
	QMatrix4x4 temprotMat = qm.inverted() * rotMat;
	
	upVecInLocal; //when rotated, assume the view vec is changed, but we keep the up direction
	viewVecInLocal = normalize(qvec3ToFloat3(temprotMat.inverted()*originalViewVecInLocal));
	
	if (dot(viewVecInLocal, upVecInLocal) > 0.95)
		return;

	//rotMat = temprotMat;
	//resetModelMat();
	//return;


	//std::cout << "the new viewVecInLocal: " << viewVecInLocal.x() << " " << viewVecInLocal.y() << " " << viewVecInLocal.z() << std::endl;

	QMatrix4x4 hypoModel; //hypothetatic model matrix, if not using immersive mode
	hypoModel.translate(-QVector3D(eyeInLocal.x, eyeInLocal.y, eyeInLocal.z));
	QMatrix4x4 hypoView; //hypothetatic view matrix, if not using immersive mode
	hypoView.lookAt(QVector3D(0, 0, 0), float3ToQvec3(make_float3(0,0,0)+viewVecInLocal), float3ToQvec3(upVecInLocal)); //here actually should use viewVecInWorld and upVecInWorld, but since we know hypoModel is just translation, so the vector will not change



	QMatrix4x4 mv = hypoView*hypoModel;
	modelMat = viewMat.inverted()*mv;

	rotMat = modelMat*hypoModel.inverted();
	viewVecInLocal = qvec3ToFloat3(rotMat.inverted()*originalViewVecInLocal);//further adjust


//	std::cout << "the new upVecInLocal: " << upVecInLocal.x << " " << upVecInLocal.y << " " << upVecInLocal.z << std::endl;

	int y = 0;
	return;
}