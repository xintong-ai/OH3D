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
class ImmersiveInteractor;
class QLineEdit;
class ScreenBrushInteractor;
class LabelVolumeProcessor;

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
	void setLabel(std::shared_ptr<VolumeCUDA> labelVol);
	RayCastingParameters rcp;
	std::shared_ptr<Volume> inputVolume;
	std::shared_ptr<VolumeCUDA> labelVol;
	std::shared_ptr<VolumeRenderableImmerCUDA> volumeRenderable;
	bool useLabel;

private:
	std::shared_ptr<GLWidget> openGL;
	QTimer *aTimer;

	std::shared_ptr<QPushButton> saveStateBtn;
	std::shared_ptr<QPushButton> loadStateBtn;
	
	std::shared_ptr<ImmersiveInteractor> immersiveInteractor;
	std::shared_ptr<ScreenBrushInteractor> sbInteractor;
		
	std::shared_ptr<LabelVolumeProcessor> lvProcessor;

	
	std::shared_ptr<GLMatrixManager> matrixMgr;


	QLabel *laLabel, *ldLabel, *lsLabel;
	QLabel *transFuncP1Label, *transFuncP2Label, *brLabel, *dsLabel;
	QLabel *deformForceLabel;
	QLabel *meshResLabel;

	QLineEdit *eyePosLineEdit;

	


#ifdef USE_OSVR
	std::shared_ptr<VRWidget> vrWidget;
	std::shared_ptr<VRVolumeRenderableCUDA> vrVolumeRenderable;
#endif

private slots:
	
	void isBrushingClicked();
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


};

#endif
