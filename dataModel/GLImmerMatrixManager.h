#ifndef GL_IMMER_MATRIX_MANAGER_H
#define GL_IMMER_MATRIX_MANAGER_H

#include "GLMatrixManager.h"
class Trackball;
class Rotation;
class GLImmerMatrixManager :public GLMatrixManager
{
	float projAngle;
	
	//QVector3D upVecInLocal = QVector3D(0.0f, 0.0f, 1.0f);
	//QVector3D viewVecInLocal;
	//QVector3D eyeInLocal;

	QMatrix4x4 modelMat; //all processings will be directly or indirectly applied on modelMat. viewMat and projMat should remain untouched, since they will be set by VR devices
	void resetModelMat();
public:
	GLImmerMatrixManager();
	void SetVol(float3 posMin, float3 posMax) override;

	void GetModelViewMatrix(float mv[16]) override;
	void GetProjection(float ret[16], float width, float height) override;
	void Rotate(float fromX, float fromY, float toX, float toY) override;
	void moveWheel(float v) override;

};
#endif //GL_MATRIX_MANAGER_H