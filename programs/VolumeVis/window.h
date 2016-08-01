#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <QVector3D>
#include <memory>
#include "CMakeConfig.h"

class DataMgr;
class GLWidget;
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
class GridRenderable;
class DataMgr;
class GLMatrixManager;
class ModelGridRenderable;
class ModelGrid;
class Volume;
class ModelVolumeDeformer;
class VolumeRenderableCUDA;

#ifdef USE_LEAP
class LeapListener;
namespace Leap{
	class Controller;
}
#endif

#ifdef USE_NEW_LEAP
class LeapListener;
namespace Leap{
	class Controller;
}
#endif


#ifdef USE_OSVR
class VRWidget;
class VRVolumeRenderableCUDA;
#endif

class Window : public QWidget
{
	Q_OBJECT	//without this line, the slot does not work
public:
    Window();
    ~Window();
	void init();

private:
    std::shared_ptr<GLWidget> openGL;
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

	std::shared_ptr<VolumeRenderableCUDA> volumeRenderable;
	std::shared_ptr<GlyphRenderable> glyphRenderable;
	std::shared_ptr<LensRenderable> lensRenderable;
	std::shared_ptr<GridRenderable> gridRenderable;
	std::shared_ptr<ModelGridRenderable> modelGridRenderable;
	std::shared_ptr<DataMgr> dataMgr;
	std::shared_ptr<GLMatrixManager> matrixMgr;
	QPushButton *addCurveBLensBtn;
	std::shared_ptr<ModelGrid> modelGrid;
	std::shared_ptr<Volume> inputVolume;
	std::shared_ptr<ModelVolumeDeformer> modelVolumeDeformer;

	QLabel *deformForceLabel;
	QLabel *laLabel, *ldLabel, *lsLabel;
	QLabel *transFuncP1Label, *transFuncP2Label, *brLabel, *dsLabel;

#ifdef USE_OSVR
	std::shared_ptr<VRWidget> vrWidget;
	std::shared_ptr<VRVolumeRenderableCUDA> vrGlyphRenderable;
#endif
#ifdef USE_LEAP
	LeapListener* listener;
	Leap::Controller* controller;
#endif
#ifdef USE_NEW_LEAP
	LeapListener* listener;
	Leap::Controller* controller;
#endif
private slots:
	void AddLens();
	void AddLineLens();
	void AddCurveBLens(); 
	void SlotToggleGrid(bool b);
	void SlotToggleUdbe(bool b);
	void SlotToggleUsingGlyphSnapping(bool b);
	void SlotTogglePickingGlyph(bool b);
	void SlotToggleGlyphPickingFinished();
	void SlotDeformModeChanged(bool clicked);
	void SlotSaveState();
	void SlotLoadState();

	void deformForceSliderValueChanged(int);
	void transFuncP1LabelSliderValueChanged(int);
	void transFuncP2LabelSliderValueChanged(int); 
	void brSliderValueChanged(int v);
	void dsSliderValueChanged(int v);
	void laSliderValueChanged(int);
	void ldSliderValueChanged(int);
	void lsSliderValueChanged(int);



#ifdef USE_LEAP
	void SlotUpdateHands(QVector3D leftIndexTip, QVector3D rightIndexTip, int numHands);
#endif
#ifdef USE_NEW_LEAP
	void SlotUpdateHands(QVector3D leftIndexTip, QVector3D rightIndexTip, int numHands);
#endif
};

#endif
