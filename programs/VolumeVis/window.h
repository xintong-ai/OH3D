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
class MeshRenderable;
class MeshDeformProcessor;
class Volume;
class PhysicalVolumeDeformProcessor;
class VolumeRenderableCUDA;
class Lens;
class RegularInteractor;
class LensInteractor;

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
	QPushButton* addLensBtn;
	QPushButton* addLineLensBtn;

	std::shared_ptr<QPushButton> delLensBtn;


	std::shared_ptr<QPushButton> saveStateBtn;
	std::shared_ptr<QPushButton> loadStateBtn;
	
	std::shared_ptr<VolumeRenderableCUDA> volumeRenderable;
	std::shared_ptr<LensRenderable> lensRenderable;
	std::shared_ptr<MeshRenderable> meshRenderable;

	std::shared_ptr<RegularInteractor> rInteractor;
	std::shared_ptr<LensInteractor> lensInteractor;

	std::shared_ptr<DataMgr> dataMgr;
	std::shared_ptr<GLMatrixManager> matrixMgr;
	QPushButton *addCurveLensBtn;
	std::shared_ptr<MeshDeformProcessor> meshDeformer;
	std::shared_ptr<Volume> inputVolume;
	std::shared_ptr<PhysicalVolumeDeformProcessor> modelVolumeDeformer;
	std::vector<Lens*> lenses; //can change Lens* to shared pointer, to avoid manually deleting

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

private slots:
	void AddLens();
	void AddLineLens();
	void AddCurveLens(); 
	void SlotToggleGrid(bool b);
	void SlotToggleBackFace(bool b);
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

};

#endif
