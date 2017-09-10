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

class ImmersiveInteractor;
class RegularInteractor;
class AnimationByMatrixProcessor;
class PositionBasedDeformProcessor;
class SphereRenderable;
class MatrixMgrRenderable;
class DeformFrameRenderable;
class GlyphRenderable;
class PolyRenderable;
class PolyMesh;
class MarchingCube2;

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
	bool useIsoAdjust = false;
	std::vector<float3> views;

	//std::shared_ptr<RayCastingParameters> rcp;

	std::shared_ptr<PolyMesh> polyMesh;
	std::vector<std::shared_ptr<PolyMesh>> polyMeshes;

	std::shared_ptr<PositionBasedDeformProcessor> positionBasedDeformProcessor = 0;

	std::shared_ptr<GLWidget> openGL;
	std::shared_ptr<GLMatrixManager> matrixMgr;

	std::shared_ptr<ImmersiveInteractor> immersiveInteractor;
	std::shared_ptr<RegularInteractor> regularInteractor;

	std::shared_ptr<AnimationByMatrixProcessor> animationByMatrixProcessor;


	std::shared_ptr<GLMatrixManager> matrixMgrExocentric;


	//for main view
	std::shared_ptr<MatrixMgrRenderable> matrixMgrRenderable;
	std::shared_ptr<DeformFrameRenderable> deformFrameRenderable;
	std::shared_ptr<PolyRenderable> polyRenderable;

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

	std::shared_ptr<MarchingCube2> mc;
	QLineEdit *eyePosLineEdit;

	QLabel *isoValueLabel;

	std::shared_ptr<QRadioButton> immerRb;
	std::shared_ptr<QRadioButton> nonImmerRb;

	private slots:

	void SlotSaveState();
	void SlotLoadState();
	void applyEyePos();

	void isDeformEnabledClicked(bool b);
	void isForceDeformEnabledClicked(bool b);
	void isDeformColoringEnabledClicked(bool b);
	
	void toggleWireframeClicked(bool b);

	void SlotImmerRb(bool);
	void SlotNonImmerRb(bool);
	
	void isoValueSliderValueChanged(int v);

	void doTourBtnClicked();
	void saveScreenBtnClicked();
};

#endif
