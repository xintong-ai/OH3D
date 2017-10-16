#ifndef GL_MATRIX_MANAGER_H
#define GL_MATRIX_MANAGER_H
#include <vector_types.h>
#include <vector_functions.h>
#include <QVector3D>
#include <QMatrix4x4>
#include "MatrixManager.h"


class GLMatrixManager: public MatrixManager{
protected:
	//for mode matrix
	float scaleEff = 1; //better use less. hard to control 
	QMatrix4x4 rotMat;
	QVector3D transVec;  //prviously we applied transVec in the end. now we apply transVec at the beginning
	void UpdateModelMatrixFromDetail();
	QMatrix4x4 modeMat;

	//for view matrix
	QVector3D eyeInWorld;
	QVector3D upVecInWorld;
	QVector3D viewVecInWorld;
	////derivable quantities, all defined in MatrixManager
	//float3 eyeInLocal; 
	//float3 viewVecInLocal;
	//float3 upVecInLocal;
	void UpdateViewMatrixFromDetail();
	QMatrix4x4 viewMat;

	//for projection 
	//float zNear = 0.1;
	//float zFar = 100;
	float projAngle = 60;
	float width = 1, height = 1;
	void UpdateProjMatrixFromDetail(float width, float height);
	void UpdateProjMatrixFromDetail();

	QMatrix4x4 projMat;

	//data related
	float3 dataMin = make_float3(0, 0, 0);
	float3 dataMax = make_float3(10, 10, 10);
	float2 clipDepthRangeOfVol = make_float2(0, 0);

	//utility funcs
	float3 inline qvec3ToFloat3(QVector3D q){ return make_float3(q.x(), q.y(), q.z()); }
	QVector3D inline float3ToQvec3(float3 f){ return QVector3D(f.x, f.y, f.z); }

	void updateDepthRange();

public:
	//bool justChanged = false; //currently just cover the change of model and view mat, not prj mat

	GLMatrixManager(float3 posMin = make_float3(0, 0, 0), float3 posMax = make_float3(0, 0, 0));
	void setDefaultForImmersiveMode();
	void setWinSize(float w, float h);

	float zNear = 0.1;
	float zFar = 1000;
	
	//setting functions
	void Scale(float v);
	void SetViewMat(QMatrix4x4 _m){ 
		viewMat = _m; 
		QMatrix4x4 invViewMat = viewMat.inverted();
		eyeInWorld = invViewMat*QVector3D(0, 0, 0);
		viewVecInWorld = QVector3D(invViewMat*QVector4D(0, 0, -1, 0));
		upVecInWorld = QVector3D(invViewMat*QVector4D(0, 1, 0, 0));
		updateDepthRange();
	}
	void applyPreRotMat(QMatrix4x4 r){ 
		rotMat = r * rotMat; 
		UpdateModelMatrixFromDetail();
	};
	void setRotMat(QMatrix4x4 r){
		rotMat = r;
		UpdateModelMatrixFromDetail();
	}; 
	void setTransVec(QVector3D _t){
		QVector3D oldTransVec = transVec;
		transVec = _t;
		recentMove =  - qvec3ToFloat3(transVec - oldTransVec); //opposite direction!
		UpdateModelMatrixFromDetail();
	};

	QVector3D getTransVec(){ return transVec; }
	QVector3D getEyeVecInWorld(){ return eyeInWorld; };
	QVector3D getUpVecInWorld(){ return upVecInWorld; };
	QVector3D getViewVecInWorld(){ return viewVecInWorld; };

	float3 getViewVecInLocal() override
	{
		return normalize(qvec3ToFloat3(QVector3D(modeMat.inverted()*QVector4D(viewVecInWorld, 0)))); //viewVecInWorld is a vector, so add 0 to make it a vector4
	}
	float3 getEyeInLocal() override
	{
		return qvec3ToFloat3(modeMat.inverted()*eyeInWorld);
	};
	float3 getUpInLocal() override
	{
		return normalize(qvec3ToFloat3(QVector3D(modeMat.inverted()*QVector4D(upVecInWorld, 0)))); //viewVecInWorld is a vector, so add 0 to make it a vector4;
	};
	float3 getHorizontalMoveVec(float3 refUpInLocal) override
	{
		float3 viewVecInLocal = getViewVecInLocal();
		return normalize(cross(refUpInLocal, cross(viewVecInLocal, refUpInLocal)));
	}

	void setEyeInWorld(QVector3D _eyeVecInWorld)
	{
		eyeInWorld = _eyeVecInWorld;
		UpdateViewMatrixFromDetail();
	}
	void setViewAndUpInWorld(QVector3D _viewVecInWorld, QVector3D _upVecInWorld)
	{
		viewVecInWorld = _viewVecInWorld;
		upVecInWorld = _upVecInWorld;
		UpdateViewMatrixFromDetail();
	}

	void moveEyeInLocalByModeMat(float3 newEyeInLocal){	
		//moving eyeinlocal can be done in 2 ways, by changing modemat or by changing eyeinworld. need to specific
		float3 oldeye = getEyeInLocal(); //for immersive mode, this should be the same with -transVec, but may not the same for other modes
		setTransVec(transVec - float3ToQvec3(newEyeInLocal - oldeye));
		UpdateModelMatrixFromDetail();
	}

	void GetRotMatrix(QMatrix4x4 &m){ m = rotMat; };
	
	void GetModelMatrix(QMatrix4x4 &m){ m = modeMat; };
	void GetModelViewMatrix(QMatrix4x4 &mv){ mv = viewMat*modeMat; };
	void GetModelViewMatrix(float mv[16]);
	void GetProjection(float ret[16]);
	//when asking the projection matrix with the width and height, compute the matrix using the given width and height, and modify the stored projMat

	float3 DataCenter();
	void GetVol(float3 &posMin, float3 &posMax){ posMin = dataMin; posMax = dataMax; }
	float GetVolScale()//used to roughly estimate how big the volume is
	{ 
		float3 t = dataMax - dataMin;
		return t.x + t.y + t.z - min(min(t.x, t.y), t.z) - max(max(t.x, t.y), t.z);
	}
	void GetClipDepthRangeOfVol(float2 & depthRange);
	void SaveState(const char* filename);
	void LoadState(const char* filename);


	bool toRotateLeft = false, toRotateRight = false;//to receive change requirement, then the GLMatrixManger object will be changed in certain processor
};
#endif //GL_MATRIX_MANAGER_H