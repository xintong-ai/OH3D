#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <QVector3D>
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
class QRadioButton;
class QTimer;
class LensRenderable;
class DataMgr;
class GLMatrixManager;
class ModelGridRenderable;
class MeshDeformProcessor;
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
	
	std::shared_ptr<VolumeRenderableCUDA> volumeRenderable;
	std::shared_ptr<LensRenderable> lensRenderable;
	std::shared_ptr<ModelGridRenderable> modelGridRenderable;
	std::shared_ptr<DataMgr> dataMgr;
	std::shared_ptr<GLMatrixManager> matrixMgr;
	QPushButton *addCurveLensBtn;
	std::shared_ptr<MeshDeformProcessor> modelGrid;
	std::shared_ptr<Volume> inputVolume;
	std::shared_ptr<ModelVolumeDeformer> modelVolumeDeformer;

	QLabel *laLabel, *ldLabel, *lsLabel;
	QLabel *transFuncP1Label, *transFuncP2Label, *brLabel, *dsLabel;
	QLabel *deformForceLabel;
	QLabel *meshResLabel;

	float deformForceConstant = 100;
	int meshResolution = 20;

#ifdef USE_OSVR
	std::shared_ptr<VRWidget> vrWidget;
	std::shared_ptr<VRVolumeRenderableCUDA> vrVolumeRenderable;
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
	void AddCurveLens(); 
	void SlotToggleGrid(bool b);
	void SlotToggleBackFace(bool b);
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

	void SlotRbUniformChanged(bool);
	void SlotRbDensityChanged(bool);
	void SlotRbTransferChanged(bool);
	void SlotRbGradientChanged(bool);
	
	void SlotDelLens();

	void SlotAddMeshRes();
	void SlotMinusMeshRes();
	void SlotToggleCbDrawInsicionOnCenterFace(bool b);

#ifdef USE_LEAP
	void SlotUpdateHands(QVector3D leftIndexTip, QVector3D rightIndexTip, int numHands);
#endif
#ifdef USE_NEW_LEAP
	void SlotUpdateHands(QVector3D leftIndexTip, QVector3D rightIndexTip, int numHands);
#endif
};

#endif
