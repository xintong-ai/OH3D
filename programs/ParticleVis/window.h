#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <QVector3D>
#include <memory>
#include "CMakeConfig.h"
#include <vector_types.h>

class DataMgr;
class DeformGLWidget;
class MarchingCubes;
class QPushButton;
class QSlider;
class Renderable;
class QCheckBox;
class QLabel;
class GlyphRenderable;
class QRadioButton;
class QTimer;
class LensRenderable;
class DataMgr;
class GLMatrixManager;
class MeshRenderable;
class MeshDeformProcessor;
class Particle;
class ScreenLensDisplaceProcessor;
class PhysicalParticleDeformProcessor;
class Lens;

//#define USE_NEW_LEAP

#ifdef USE_LEAP
class LeapListener;
namespace Leap{
	class Controller;
}
#endif

#ifdef USE_NEW_LEAP
class LeapListener;
class ArrowRenderable; //used to draw leap finger indicators
namespace Leap{
	class Controller;
}
#endif

#ifdef USE_OSVR
class VRWidget;
class VRGlyphRenderable;
#endif

class Window : public QWidget
{
	Q_OBJECT	//without this line, the slot does not work
public:
    Window();
    ~Window();
	void init();

private:
	std::shared_ptr<DeformGLWidget> openGL;
	QTimer *aTimer;
	const int nScale = 20;
	QPushButton* addLensBtn;
	QPushButton* addLineLensBtn;

	std::shared_ptr<QPushButton> delLensBtn;
	std::shared_ptr<QRadioButton> radioDeformScreen;
	std::shared_ptr<QRadioButton> radioDeformObject;

	std::shared_ptr<QPushButton> saveStateBtn;
	std::shared_ptr<QPushButton> loadStateBtn;

	QSlider *deformForceSlider;

	QCheckBox* usingGlyphSnappingCheck;
	QCheckBox* usingGlyphPickingCheck;
	QCheckBox* freezingFeatureCheck;
	QCheckBox* usingFeatureSnappingCheck;
	QCheckBox* usingFeaturePickingCheck;

	std::shared_ptr<GlyphRenderable> glyphRenderable;
	std::shared_ptr<LensRenderable> lensRenderable;
	std::shared_ptr<MeshRenderable> meshRenderable;
	std::shared_ptr<DataMgr> dataMgr;
	std::shared_ptr<GLMatrixManager> matrixMgr;
	QPushButton *addCurveLensBtn;
	std::shared_ptr<MeshDeformProcessor> meshDeformer; 
	std::shared_ptr<ScreenLensDisplaceProcessor> screenLensDisplaceProcessor;
	std::shared_ptr<PhysicalParticleDeformProcessor> physicalParticleDeformer;
	std::shared_ptr<Particle> inputParticle;
	std::vector<Lens*> lenses; //can change Lens* to shared pointer, to avoid manually deleting

	QLabel *deformForceLabel;
	QLabel *meshResLabel;
	float deformForceConstant = 3;
	int meshResolution = 20;

#ifdef USE_OSVR
	std::shared_ptr<VRWidget> vrWidget;
	std::shared_ptr<VRGlyphRenderable> vrGlyphRenderable;

	//for test
	std::shared_ptr<VRGlyphRenderable> vrGlyphRenderable2;
	std::shared_ptr<LensRenderable> lensRenderable2;

#endif
#ifdef USE_LEAP
	LeapListener* listener;
	Leap::Controller* controller;
#endif

#ifdef USE_NEW_LEAP
	LeapListener* listener;
	Leap::Controller* controller; 
	std::shared_ptr<ArrowRenderable> arrowRenderable;
	std::shared_ptr<Particle> leapFingerIndicators;
	std::vector<float3> leapFingerIndicatorVecs;
#endif

private slots:
	void AddLens();
	void AddLineLens();
	void AddCurveLens(); 
	void SlotDelLens();
	void SlotToggleGrid(bool b); 
	void SlotToggleBackFace(bool b);
	void SlotToggleUdbe(bool b);
	void SlotToggleCbChangeLensWhenRotateData(bool b);
	void SlotToggleCbDrawInsicionOnCenterFace(bool b);
	void SlotToggleUsingGlyphSnapping(bool b);
	void SlotTogglePickingGlyph(bool b);
	void SlotToggleGlyphPickingFinished();
	void SlotDeformModeChanged(bool clicked);
	void SlotSaveState();
	void SlotLoadState();
	void deformForceSliderValueChanged(int);

	void SlotToggleFreezingFeature(bool b);
	void SlotToggleUsingFeatureSnapping(bool b);
	void SlotTogglePickingFeature(bool b);
	void SlotToggleFeaturePickingFinished();

	void SlotRbUniformChanged(bool);
	void SlotRbDensityChanged(bool);
	void SlotRbTransferChanged(bool);
	void SlotRbGradientChanged(bool);

	void SlotAddMeshRes();
	void SlotMinusMeshRes();

#ifdef USE_LEAP
	void SlotUpdateHands(QVector3D leftIndexTip, QVector3D rightIndexTip, int numHands);
#endif
#ifdef USE_NEW_LEAP
	void SlotUpdateHands(QVector3D rightThumbTip, QVector3D rightIndexTip, QVector3D leftThumbTip, QVector3D leftIndexTip, QVector3D rightMiddleTip, QVector3D rightRingTip, int numHands);
#endif
};

#endif
