////// NOTE!!!!!!!!!!
////// LEAP and OSVR of this project has not been well set



#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <memory>
#include "CMakeConfig.h"

class DataMgr;
class DeformGLWidget;
class MarchingCubes;
class QPushButton;
class QSlider;
class Renderable;
class QCheckBox;
class QLabel;
class Cubemap;
class GlyphRenderable;
class QRadioButton;
class QTimer;
class LensRenderable;
class LeapListener;
class DataMgr;
class GLMatrixManager;
class PolyRenderable;
class QListWidget;
class Particle;
class Lens;
class PolyMesh;
class MeshDeformProcessor;
class ScreenLensDisplaceProcessor;
class PhysicalParticleDeformProcessor;
class RegularInteractor;
class LensInteractor;
class MeshRenderable;
#ifdef USE_OSVR
class VRWidget;
class VRGlyphRenderable;
#endif

namespace Leap{
	class Controller;
}
class Window : public QWidget
{
	Q_OBJECT	//without this line, the slot does not work
public:
    Window();
    ~Window();
	void init();

private:
    std::shared_ptr<DeformGLWidget> openGL;
	std::shared_ptr<Particle> inputParticle;
	Cubemap* cubemap;
	QTimer *aTimer;
	const int nScale = 20;
	QPushButton* addLensBtn;
	QPushButton* addLineLensBtn;

	std::shared_ptr<QPushButton> delLensBtn;
	std::shared_ptr<QRadioButton> radioDeformScreen;
	std::shared_ptr<QRadioButton> radioDeformObject;

	std::shared_ptr<QPushButton> saveStateBtn;
	std::shared_ptr<QPushButton> loadStateBtn;

	QCheckBox* usingGlyphSnappingCheck;
	QCheckBox* usingGlyphPickingCheck;
	QCheckBox* freezingFeatureCheck;
	QCheckBox* usingFeatureSnappingCheck;
	QCheckBox* usingFeaturePickingCheck;

	std::shared_ptr<GlyphRenderable> glyphRenderable;
	std::shared_ptr<LensRenderable> lensRenderable;
	std::shared_ptr<MeshRenderable> meshRenderable;

#ifdef USE_OSVR
	std::shared_ptr<VRWidget> vrWidget;
	std::shared_ptr<VRGlyphRenderable> vrGlyphRenderable;
#endif
	std::shared_ptr<DataMgr> dataMgr;
	std::shared_ptr<GLMatrixManager> matrixMgr;

	LeapListener* listener;
	Leap::Controller* controller;

	QPushButton *addCurveLensBtn;

	std::shared_ptr<RegularInteractor> rInteractor;
	std::shared_ptr<LensInteractor> lensInteractor;

	PolyRenderable * polyFeature0, *polyFeature1, *polyFeature2;
	QListWidget *featuresLw = NULL;
	std::shared_ptr<PolyMesh>  polyMeshFeature0, polyMeshFeature1, polyMeshFeature2;

	std::vector<Lens*> lenses; //can change Lens* to shared pointer, to avoid manually deleting

	std::shared_ptr<MeshDeformProcessor> meshDeformer;
	std::shared_ptr<ScreenLensDisplaceProcessor> screenLensDisplaceProcessor;
	std::shared_ptr<PhysicalParticleDeformProcessor> physicalParticleDeformer;
	float deformForceConstant = 3;
	int meshResolution = 20;

private slots:
	void AddLens();
	void AddLineLens();
	void AddCurveLens();

	//void animate();
	void SlotToggleGrid(bool b);
	//void UpdateRightHand(QVector3D thumbTip, QVector3D indexTip, QVector3D indexDir);
#ifdef USE_LEAP
	void SlotUpdateHands(QVector3D leftIndexTip, QVector3D rightIndexTip, int numHands);
#endif
	void SlotToggleUsingGlyphSnapping(bool b);
	void SlotTogglePickingGlyph(bool b);
	void SlotToggleFreezingFeature(bool b);
	void SlotToggleUsingFeatureSnapping(bool b);
	void SlotTogglePickingFeature(bool b);

	void SlotToggleGlyphPickingFinished();
	void SlotToggleFeaturePickingFinished();

	void SlotDeformModeChanged(bool clicked);

	void SlotSaveState();
	void SlotLoadState();

	void SlotFeaturesLwRowChanged(int);
	void SlotDelLens();

};

#endif
