#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <QVector3D>
#include <memory>
#include "CMakeConfig.h"
#include <vector_types.h>
#include <Vector>
#include <helper_timer.h>

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
class QTimer;

class ImmersiveInteractor;
class RegularInteractor;
class AnimationByMatrixProcessor;
class PositionBasedDeformProcessor;
class SphereRenderable;
class MatrixMgrRenderable;
class DeformFrameRenderable;
class GlyphRenderable;
struct RayCastingParameters;
class PolyRenderable;
class PolyMesh;
class TimeVaryingParticleDeformerManager;

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
	std::vector<float3> views;
	
	std::shared_ptr<PolyMesh> polyMesh;
	std::shared_ptr<PolyMesh> polyMeshWall;

	
	std::shared_ptr<PositionBasedDeformProcessor> positionBasedDeformProcessor = 0;
	std::shared_ptr<TimeVaryingParticleDeformerManager> tvParticleDeformerManager = 0;

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

	std::shared_ptr<PolyRenderable> polyWallRenderable;


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

	std::shared_ptr<QRadioButton> immerRb;
	std::shared_ptr<QRadioButton> nonImmerRb;

	StopWatchInterface *timer = 0;

	bool startTV = false;
	
private slots:
	
	void SlotSaveState();
	void SlotLoadState();
	void applyEyePos();

	void isDeformEnabledClicked(bool b);
	void isForceDeformEnabledClicked(bool b);
	void isDeformColoringEnabledClicked(bool b);


	void SlotImmerRb(bool);
	void SlotNonImmerRb(bool);

	void doTourBtnClicked();
	void saveScreenBtnClicked();

	void startTVBtnClicked();
	void backToFirstTimestepBtnClicked();
};

#endif
