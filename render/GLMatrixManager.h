#ifndef GL_MATRIX_MANAGER_H
#define GL_MATRIX_MANAGER_H
#include <vector_types.h>
#include <vector_functions.h>
#include <QVector3D>
#include <QMatrix4x4>
class Trackball;
class Rotation;
class GLMatrixManager{
	Trackball *trackball;
	Rotation *rot;
	//transformation states
	QVector3D transVec;
	QVector3D eyeVecInWorld;
	QMatrix4x4 rotMat;
	QMatrix4x4 viewMat;
	float transScale = 1;
	float currentTransScale = 1;

	float3 dataMin = make_float3(0, 0, 0);
	float3 dataMax = make_float3(10, 10, 10);

	bool vrMode = false;
	bool immersiveMode = true;

public:
	GLMatrixManager(bool _vrMode = false);
	void Rotate(float fromX, float fromY, float toX, float toY);
	void Translate(float x, float y);
	void Scale(float v);
	void GetModelView(float mv[16]);
	void GetModelView(float mv[16], float _viewMat[16]);
	void GetModelMatrix(float ret[16]);
	void GetProjection(float ret[16], float width, float height);

	void SetVol(float3 posMin, float3 posMax);
	void GetVol(float3 &posMin, float3 &posMax){ posMin = dataMin; posMax = dataMax; }
	void SetTransVec(float x, float y, float z){ transVec[0] = x; transVec[1] = y; transVec[2] = z; }
	void SetScale(float v){ transScale = v; }
	void SetCurrentScale(float v){ currentTransScale = v; }
	void FinishedScale();
	float3 DataCenter();
	void SetViewMat(QMatrix4x4 _m){ viewMat = _m; }

	void SaveState(const char* filename);
	void LoadState(const char* filename);
};
#endif //GL_MATRIX_MANAGER_H