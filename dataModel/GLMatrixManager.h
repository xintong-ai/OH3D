#ifndef GL_MATRIX_MANAGER_H
#define GL_MATRIX_MANAGER_H
#include <vector_types.h>
#include <vector_functions.h>
#include <QVector3D>
#include <QMatrix4x4>
#include "MatrixManager.h"

class Trackball;
class Rotation;
class GLMatrixManager: public MatrixManager{
protected:
	Trackball *trackball;
	Rotation *rot;
	//transformation states
	QVector3D transVec;
	QVector3D eyeInWorld; 
	QVector3D upVecInWorld = QVector3D(0.0f, 0.0f, 1.0f);
	QVector3D cofLocal; //center of focus in local coordinate //always assume the cof in worldcoordinate is (0,0,0)
	QMatrix4x4 rotMat;
	QMatrix4x4 viewMat;
	QVector3D originalViewVecInLocal, originalUpVecInLocal; //fixed viewMat is equivalent to using these two vectors
	QMatrix4x4 projMat;
	float volScale = 1;
	float transScale = 1;
	float currentTransScale = 1;

	float3 dataMin = make_float3(0, 0, 0);
	float3 dataMax = make_float3(10, 10, 10);

	bool immersiveMode = false;

public:
	GLMatrixManager();
	void SetImmersiveMode();
	void SetRegularMode();
	
	float zNear = 0.1;
	float zFar = 100;
	
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
	virtual void Rotate(float fromX, float fromY, float toX, float toY);//historical methods. may deprecate
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

	void GetModelMatrix(QMatrix4x4 &m);
	void GetModelMatrix(float ret[16]);
	void GetViewMatrix(QMatrix4x4 &v){
		v = viewMat;
	}; 
	void GetModelViewMatrix(QMatrix4x4 &mv);
	virtual void GetModelViewMatrix(float mv[16]);
	void GetModelViewMatrix(float mv[16], float _viewMat[16]);
	void GetProjection(QMatrix4x4 &p, float width, float height);
	virtual void GetProjection(float ret[16], float width, float height);
	//when asking the projection matrix with the width and height, compute the matrix using the given width and height, and modify the stored projMat
	void GetProjection(QMatrix4x4 &p){
		p = projMat;
	};
	//when asking the projection matrix without the width and height, give the stored projMat 


 //while we have set our own mvp matrices in regular mode, the vr device can also provided _view and _projection matrix. We should apply the matrices provided by the device to achieve better stereo vision. to achieve this, we compute a fake model matrix fakeModel, will have the same effect with regular mode when applying together with the device _view and _projection matrix

	QMatrix4x4 fakeModel;
	void computeFakeModel(QMatrix4x4 vrViewMat, QMatrix4x4 vrProjectMat)
	{
		QMatrix4x4 oriMV, oriP;
		GetModelViewMatrix(oriMV);
		oriP = projMat;
		fakeModel = (vrProjectMat*vrViewMat).inverted()*oriP*oriMV;
	}
	//returns the mv matrix when applying fakeModel
	void GetModelViewMatrixVR(float mv[16], float _viewMat[16]);


	float3 DataCenter();
	virtual void SetVol(float3 posMin, float3 posMax);
	void GetVol(float3 &posMin, float3 &posMax){ posMin = dataMin; posMax = dataMax; }
	
	void SaveState(const char* filename);
	void LoadState(const char* filename);


	virtual void moveWheel(float v){
		QMatrix4x4 m;
		GetModelMatrix(m);

		QVector3D dir = eyeInWorld.normalized()*v / -1000.0;
		QVector3D newCofInWorld = dir;

		setCofLocal(m.inverted().map(newCofInWorld));
	};
};
#endif //GL_MATRIX_MANAGER_H