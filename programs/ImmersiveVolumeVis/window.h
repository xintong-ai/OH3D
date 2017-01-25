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
class GLMatrixManager;
class Volume;
class VolumeCUDA;
class VolumeRenderableCUDA;
class ImmersiveInteractor;
class QLineEdit;

#ifdef USE_OSVR
class VRWidget;
class VRVolumeRenderableCUDA;
#endif

class Window : public QWidget
{
	Q_OBJECT	//without this line, the slot does not work
public:
	Window(std::shared_ptr<Volume> v, std::shared_ptr<VolumeCUDA>);
    ~Window();
	void init();
	void setLabel(std::shared_ptr<VolumeCUDA> labelVol);

private:
	std::shared_ptr<DeformGLWidget> openGL;
	QTimer *aTimer;

	std::shared_ptr<QPushButton> saveStateBtn;
	std::shared_ptr<QPushButton> loadStateBtn;
	
	std::shared_ptr<ImmersiveInteractor> immersiveInteractor;
	std::shared_ptr<GLMatrixManager> matrixMgr;
	std::shared_ptr<Volume> inputVolume;

	QLabel *laLabel, *ldLabel, *lsLabel;
	QLabel *transFuncP1Label, *transFuncP2Label, *brLabel, *dsLabel;
	QLabel *deformForceLabel;
	QLabel *meshResLabel;

	QLineEdit *eyePosxLineEdit;
	QLineEdit *eyePosyLineEdit;
	QLineEdit *eyePoszLineEdit;
	QLineEdit *eyePosLineEdit;

	std::shared_ptr<VolumeRenderableCUDA> volumeRenderable;
	std::shared_ptr<VolumeCUDA> labelVol;

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


};

#endif
