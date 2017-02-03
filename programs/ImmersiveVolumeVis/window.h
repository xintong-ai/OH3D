#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <QVector3D>
#include <memory>
#include "CMakeConfig.h"
#include "myDefine.h"

class DataMgr;
class GLWidget;
class MarchingCubes;
class QPushButton;
class QSlider;
class Renderable;
class QCheckBox;
class QLabel;
class QRadioButton;
class QTimer;
class GLMatrixManager;
class Volume;
class VolumeCUDA;
class VolumeRenderableImmerCUDA;
class VolumeRenderableCUDA;
class ImmersiveInteractor;
class QLineEdit;
class ScreenBrushInteractor;
class RegularInteractor;
class LabelVolumeProcessor;
class ViewpointEvaluator;

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

	RayCastingParameters rcp;
	std::shared_ptr<Volume> inputVolume;
	std::shared_ptr<Volume> channelVolume;

	std::shared_ptr<VolumeCUDA> labelVol;
	bool useLabel;

	std::shared_ptr<GLWidget> openGL;
	
	std::shared_ptr<ImmersiveInteractor> immersiveInteractor;
	std::shared_ptr<ScreenBrushInteractor> sbInteractor;
		
	std::shared_ptr<LabelVolumeProcessor> lvProcessor;

	std::shared_ptr<VolumeRenderableImmerCUDA> volumeRenderable;

	std::shared_ptr<GLMatrixManager> matrixMgr;

	std::shared_ptr<QPushButton> saveStateBtn;
	std::shared_ptr<QPushButton> loadStateBtn;
	QLabel *laLabel, *ldLabel, *lsLabel;
	QLabel *transFuncP1Label, *transFuncP2Label, *brLabel, *dsLabel;
	QLabel *deformForceLabel;
	QLabel *meshResLabel;
	QLineEdit *eyePosLineEdit;

	std::shared_ptr<QRadioButton> oriVolumeRb;
	std::shared_ptr<QRadioButton> channelVolumeRb;
	
	std::shared_ptr<QRadioButton> immerRb;
	std::shared_ptr<QRadioButton> nonImmerRb;

	std::shared_ptr<ViewpointEvaluator> ve;

	QTimer *aTimer;

	//for miniature
	std::shared_ptr<GLWidget> openGLMini;
	std::shared_ptr<VolumeRenderableCUDA> volumeRenderableMini;
	std::shared_ptr<GLMatrixManager> matrixMgrMini;
	std::shared_ptr<RegularInteractor> regularInteractor;
	std::shared_ptr<RegularInteractor> regularInteractorMini;

#ifdef USE_OSVR
	std::shared_ptr<VRWidget> vrWidget;
	std::shared_ptr<VRVolumeRenderableCUDA> vrVolumeRenderable;
#endif

private slots:
	
	void SlotSaveState();
	void SlotLoadState();
	void applyEyePos();

	void transFuncP1LabelSliderValueChanged(int);
	void transFuncP2LabelSliderValueChanged(int); 
	void brSliderValueChanged(int v);
	void dsSliderValueChanged(int v);
	void laSliderValueChanged(int);
	void ldSliderValueChanged(int);
	void lsSliderValueChanged(int);

	void isBrushingClicked();
	void moveToOptimalBtnClicked();

	void SlotOriVolumeRb(bool);
	void SlotChannelVolumeRb(bool);

	void SlotImmerRb(bool);
	void SlotNonImmerRb(bool);
};

#endif
