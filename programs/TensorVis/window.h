#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <memory>
#include "CMakeConfig.h"

class DataMgr;
class GLWidget;
class MarchingCubes;
class QPushButton;
class QSlider;
class Renderable;
class TexPlaneRenderable;
class QCheckBox;
class QLabel;
class Cubemap;
//class GlyphRenderable;
class GlyphRenderable;
class QRadioButton;
class QTimer;
class LensRenderable;
class GridRenderable;
class LeapListener;
class DataMgr;
class GLMatrixManager;
class ModelGridRenderable;
class ModelGrid;
class PolyRenderable;
class QListWidget;

//#define USE_PARTICLE

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
    std::unique_ptr<GLWidget> openGL;
	Cubemap* cubemap;
	QTimer *aTimer;
	const int nScale = 20;
	QPushButton* addLensBtn;
	QPushButton* addLineLensBtn;

	std::unique_ptr<QPushButton> delLensBtn;
	std::unique_ptr<QRadioButton> radioDeformScreen;
	std::unique_ptr<QRadioButton> radioDeformObject;

	std::unique_ptr<QPushButton> saveStateBtn;
	std::unique_ptr<QPushButton> loadStateBtn;

	QCheckBox* usingGlyphSnappingCheck;
	QCheckBox* usingGlyphPickingCheck;
	QCheckBox* freezingFeatureCheck;
	QCheckBox* usingFeatureSnappingCheck;
	QCheckBox* usingFeaturePickingCheck;

	std::unique_ptr<GlyphRenderable> glyphRenderable;
	std::unique_ptr<LensRenderable> lensRenderable;
	std::unique_ptr<GridRenderable> gridRenderable;
#ifdef USE_OSVR
	std::unique_ptr<VRWidget> vrWidget;
	std::unique_ptr<VRGlyphRenderable> vrGlyphRenderable;
#endif
	std::unique_ptr<ModelGridRenderable> modelGridRenderable;
	std::unique_ptr<DataMgr> dataMgr;
	std::shared_ptr<GLMatrixManager> matrixMgr;

	LeapListener* listener;
	Leap::Controller* controller;

	QPushButton *addCurveBLensBtn;

	std::shared_ptr<ModelGrid> modelGrid;

	PolyRenderable * polyFeature0, *polyFeature1, *polyFeature2;
	QListWidget *featuresLw = NULL;
private slots:
	void AddLens();
	void AddLineLens();
	void AddCurveBLens();

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

};

#endif
