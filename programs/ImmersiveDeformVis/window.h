#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <QVector3D>
#include <memory>
#include "CMakeConfig.h"
#include <vector_types.h>

#include "GLWidgetQtDrawing.h"

class DataMgr;
class GLWidget;
class QPushButton;
class QSlider;
class QCheckBox;
class QLabel;
class QRadioButton;
class QTimer;
class GLMatrixManager;
class QLineEdit;

class Volume;
class VolumeCUDA;
class VolumeRenderableImmerCUDA;
class VolumeRenderableCUDA;
class ImmersiveInteractor;
class ScreenBrushInteractor;
class RegularInteractor;
class LabelVolumeProcessor;
class ViewpointEvaluator;
class AnimationByMatrixProcessor;
class PositionBasedDeformProcessor;
class SphereRenderable;
class MatrixMgrRenderable;
class InfoGuideRenderable;
class DeformFrameRenderable;
class GlyphRenderable;
struct RayCastingParameters;
class PolyMesh;
class PolyRenderable;
class TraceRenderable;

#ifdef USE_LEAP
class LeapListener;
namespace Leap{
	class Controller;
}
class MatrixLeapInteractor;
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
	int3 dims;
	float3 spacing;
	std::vector<float3> views;

	std::shared_ptr<RayCastingParameters> rcp;
	std::shared_ptr<RayCastingParameters> rcpForChannelSkel;

	std::shared_ptr<Volume> inputVolume;
	std::shared_ptr<Volume> channelVolume = 0; //only render when need to test

	std::shared_ptr<PositionBasedDeformProcessor> positionBasedDeformProcessor = 0;

	//bool useLabel;
	bool hasLabelFromFile = false;


	std::shared_ptr<GLWidget> openGL;
	std::shared_ptr<GLMatrixManager> matrixMgr;

	std::shared_ptr<ImmersiveInteractor> immersiveInteractor;
	std::shared_ptr<RegularInteractor> regularInteractor;

	std::shared_ptr<LabelVolumeProcessor> lvProcessor;
	std::shared_ptr<AnimationByMatrixProcessor> animationByMatrixProcessor;


	std::shared_ptr<GLMatrixManager> matrixMgrExocentric;


	//for miniature
	std::shared_ptr<GLWidget> openGLMini;
	std::shared_ptr<VolumeRenderableCUDA> volumeRenderableMini;
	std::shared_ptr<RegularInteractor> regularInteractorMini;
	std::shared_ptr<MatrixMgrRenderable> matrixMgrRenderableMini;
	std::shared_ptr<GlyphRenderable> glyphRenderable;



	//for main view
	std::shared_ptr<VolumeRenderableImmerCUDA> volumeRenderable;
	std::shared_ptr<MatrixMgrRenderable> matrixMgrRenderable;
	std::shared_ptr<InfoGuideRenderable> infoGuideRenderable;
	std::shared_ptr<DeformFrameRenderable> deformFrameRenderable;
	//for 2d view
	Helper helper;

#ifdef USE_LEAP
	LeapListener* listener;
	Leap::Controller* controller; 
	std::shared_ptr<MatrixLeapInteractor> matrixMgrLeapInteractor;
#endif

#ifdef USE_OSVR
	std::shared_ptr<VRWidget> vrWidget;
	std::shared_ptr<VRVolumeRenderableCUDA> vrVolumeRenderable;
#endif

private:
	std::shared_ptr<QPushButton> saveStateBtn;
	std::shared_ptr<QPushButton> loadStateBtn;
	QLabel *laLabel, *ldLabel, *lsLabel;
	QLabel *transFuncP1Label, *transFuncP2Label, *transFuncP1SecondLabel, *transFuncP2SecondLabel, *brLabel, *dsLabel;
	QLabel *deformForceLabel;
	QLabel *meshResLabel;
	QLineEdit *eyePosLineEdit;

	std::shared_ptr<QRadioButton> oriVolumeRb;
	std::shared_ptr<QRadioButton> channelVolumeRb;
	std::shared_ptr<QRadioButton> skelVolumeRb;
	std::shared_ptr<QRadioButton> surfaceRb;

	std::shared_ptr<QRadioButton> immerRb;
	std::shared_ptr<QRadioButton> nonImmerRb;

private slots:
	
	void SlotSaveState();
	void SlotLoadState();
	void applyEyePos();

	void usePreIntCBClicked(bool);
	void useSplineInterpolationCBClicked(bool);

	void transFuncP1LabelSliderValueChanged(int);
	void transFuncP2LabelSliderValueChanged(int); 
	void transFuncP1SecondLabelSliderValueChanged(int);
	void transFuncP2SecondLabelSliderValueChanged(int); 
	void brSliderValueChanged(int v);
	void dsSliderValueChanged(int v);
	void laSliderValueChanged(int);
	void ldSliderValueChanged(int);
	void lsSliderValueChanged(int);

	void isDeformEnabledClicked(bool b);

	void SlotOriVolumeRb(bool);
	void SlotChannelVolumeRb(bool);
	void SlotSkelVolumeRb(bool);
	void SlotSurfaceRb(bool);

	void SlotImmerRb(bool);
	void SlotNonImmerRb(bool);

	void zSliderValueChanged(int v);

	void doTourBtnClicked();
	void saveScreenBtnClicked();
};

#endif
