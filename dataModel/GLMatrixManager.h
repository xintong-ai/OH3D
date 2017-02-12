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
	QVector3D eyeInWorld; 
	QVector3D upVecInWorld = QVector3D(0.0f, 0.0f, 1.0f);
	QVector3D cofLocal; //center of focus in local coordinate //always assume the cof in worldcoordinate is (0,0,0)
	QMatrix4x4 rotMat;
	QMatrix4x4 viewMat;
	float volScale = 1;
	float transScale = 1;
	float currentTransScale = 1;

	float3 dataMin = make_float3(0, 0, 0);
	float3 dataMax = make_float3(10, 10, 10);

	bool vrMode = false;
	bool immersiveMode = false;

public:
	GLMatrixManager(bool _vrMode = false);
	void SetImmersiveMode();
	void SetNonImmersiveMode();
	
	
	//setting functions
	void SetTransVec(float x, float y, float z){ transVec[0] = x; transVec[1] = y; transVec[2] = z; }
	void SetScale(float v){ transScale = v; }
	void SetCurrentScale(float v){ currentTransScale = v; }
	void FinishedScale();
	void Scale(float v);
	void SetViewMat(QMatrix4x4 _m){ viewMat = _m; }
	void applyPreRotMat(QMatrix4x4 r){ rotMat = r * rotMat; };
	void TranslateInWorldSpace(float x, float y){
		transVec[0] += x;
		transVec[1] += y;
	}
	void Rotate(float fromX, float fromY, float toX, float toY);//historical methods. may deprecate
	void setCofLocal(QVector3D _cofLocal){ cofLocal = _cofLocal; }
	void setEyeAndUpInWorld(QVector3D _eyeVecInWorld, QVector3D _upVecInWorld){
		eyeInWorld = _eyeVecInWorld; 
		upVecInWorld = _upVecInWorld;
		viewMat.setToIdentity(); 
		viewMat.lookAt(eyeInWorld, QVector3D(0.0f, 0.0f, 0.0f), upVecInWorld);
	}

	void moveEyeInLocalTo(float3 _eyeInLocal){
		moveEyeInLocalTo(QVector3D(_eyeInLocal.x, _eyeInLocal.y, _eyeInLocal.z));
	}
	void moveEyeInLocalTo(QVector3D _eyeInLocal){
		QMatrix4x4 m;
		GetModelMatrix(m);
		QVector3D eyeInLocal = m.inverted().map(eyeInWorld);
		cofLocal = cofLocal + _eyeInLocal - eyeInLocal;
	}

	//getting functions
	Trackball * getTrackBall(){ return trackball; };
	QVector3D getEyeVecInWorld(){ return eyeInWorld; };
	QVector3D getUpVecInWorld(){ return upVecInWorld; };
	QVector3D getCofLocal(){ return cofLocal; };

	void GetModelViewMatrix(float mv[16]);
	void GetModelViewMatrix(float mv[16], float _viewMat[16]);
	void GetModelViewMatrix(QMatrix4x4 &mv);
	void GetModelMatrix(float ret[16]);
	void GetModelMatrix(QMatrix4x4 &m);
	void GetProjection(float ret[16], float width, float height);
	void GetViewMatrix(QMatrix4x4 &v){
		v = viewMat;
	};



	float3 DataCenter();
	void SetVol(float3 posMin, float3 posMax);
	void GetVol(float3 &posMin, float3 &posMax){ posMin = dataMin; posMax = dataMax; }
	
	void SaveState(const char* filename);
	void LoadState(const char* filename);
};
#endif //GL_MATRIX_MANAGER_H