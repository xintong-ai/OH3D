#ifndef RENDERABLE_H
#define RENDERABLE_H
#include "QMatrix4x4"
#include <iostream>

#include <vector_functions.h>

#include <defines.h>

class GLWidget;
#define USE_PBO 0
#define DRAW_PIXELS 0
class StopWatchInterface;
class VRWidget;

inline void GetInvPVM(float modelview[16], float projection[16], float invPVM[16])
{
    QMatrix4x4 q_modelview(modelview);
    q_modelview = q_modelview.transposed();

    QMatrix4x4 q_projection(projection);
    q_projection = q_projection.transposed();

    QMatrix4x4 q_invProjMulView = (q_projection * q_modelview).inverted();

    q_invProjMulView.copyDataTo(invPVM);
}

inline void GetNormalMatrix(float modelview[16], float NormalMatrix[9])
{
    QMatrix4x4 q_modelview(modelview);
    q_modelview = q_modelview.transposed();

    q_modelview.normalMatrix().copyDataTo(NormalMatrix);
}

class Renderable: public QObject
{
	/****timing****/
	StopWatchInterface *timer = 0;
	int fpsCount = 0;        // FPS count for averaging
	int fpsLimit = 128;        // FPS limit for sampling

public:
	Renderable();
    ~Renderable();

	virtual void init(){}

    virtual void resize(int width, int height);

    virtual void draw(float modelview[16], float projection[16]);

	virtual void animate() {}
	virtual void mousePress(int x, int y, int modifier) {}
	virtual void mouseRelease(int x, int y, int modifier) {}
	virtual void mouseMove(int x, int y, int modifier) {}
	virtual void PinchScaleFactorChanged(float x, float y, float totalScaleFactor) {}

	void RecordMatrix(float* modelview, float* projection){
		memcpy(&matrix_mv.v[0].x, modelview, sizeof(float4) * 4);
		memcpy(&matrix_pj.v[0].x, projection, sizeof(float4) * 4);
	}

    //void SetDataMgr(DataMgr* ptr) {dataMgr = (DataMgr*)ptr;}

    uint* GetHostImage(){ return h_output;}

    void SetDeviceImage(uint* p){ d_output = p;}

    void SetPBO(uint v) {pbo = v;}

    //void GetDataDim(int &nx, int &ny, int &nz);

    //void SetWindowSize(int w, int h) {winWidth = w; winHeight = h;}

    //void SaveMatrices(float* mv, float* pj) {
    //    ModelviewMatrix = QMatrix4x4(mv);
    //    ModelviewMatrix = ModelviewMatrix.transposed();

    //    ProjectionMatrix = QMatrix4x4(pj);
    //    ProjectionMatrix = ProjectionMatrix.transposed();
    //}

	virtual bool MouseWheel(int x, int y, int modifier, int delta){ return false; };// {cameraChanged = true; }

	void SetAllRenderable(std::map<std::string, Renderable*>* _allrenderer) {
		allrenderer = _allrenderer;
	}

	void SetActor(GLWidget* _actor) {
		actor = _actor;
	}

	virtual void SetVRActor(VRWidget*){};

	void SetVisibility(bool b) { visible = b; }
	//void SetDrawScreenSpace(bool b) { drawScreenSpace = b; }

	void ReportDrawTime();

	void DrawBegin();

	void DrawEnd(const char* rendererName);

protected:
    //DataMgr *dataMgr;
    //int winWidth, winHeight;
    uint pbo;
    uint* d_output = NULL;
    uint* h_output = NULL;

	matrix4x4 matrix_mv;
	matrix4x4 matrix_pj;

    bool cameraChanged = false;

	std::map<std::string, Renderable*>* allrenderer;

	GLWidget* actor;

	bool visible = true;

	//bool drawScreenSpace = false;

private:
    void AllocOutImage();
};
#endif //RENDERABLE_H
