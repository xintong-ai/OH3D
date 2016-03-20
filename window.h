#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <memory>

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
class VRWidget;
class VRGlyphRenderable;
class LeapListener;
class DataMgr;
class GLMatrixManager;
class ModelGridRenderable;
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
	std::unique_ptr<VRWidget> vrWidget;
	//QSlider* sliceSlider;
	//QSlider* heightScaleSlider;
	//QSlider* sizeScaleSlider;
	Cubemap* cubemap;
	//GlyphRenderable* glyphRenderable;
	//QRadioButton *radioX;
	//QRadioButton *radioY;
	//QRadioButton *radioZ;
	QTimer *aTimer;
	const int nScale = 20;
	QPushButton* addLensBtn;
	QPushButton* addLineLensBtn;

	std::unique_ptr<QPushButton> delLensBtn;
	QPushButton* adjustOffsetBtn;
	QPushButton*refineBoundaryBtn;
	//GlyphRenderable* glyphRenderable;
	//LensRenderable* lensRenderable;
	//GridRenderable* gridRenderable;

	QCheckBox* usingSnapCheck;
	QCheckBox* highlightingFeatureCheck;
	QCheckBox* usingFeatureSnappingCheck;

	std::unique_ptr<GlyphRenderable> glyphRenderable;
	std::unique_ptr<LensRenderable> lensRenderable;
	std::unique_ptr<GridRenderable> gridRenderable;
	std::unique_ptr<VRGlyphRenderable> vrGlyphRenderable;
	std::unique_ptr<ModelGridRenderable> modelGridRenderable;
	std::unique_ptr<DataMgr> dataMgr;
	std::shared_ptr<GLMatrixManager> matrixMgr;

	LeapListener* listener;
	Leap::Controller* controller;
		//QPushButton* addNodeBtn;
	//QPushButton* viewBtn;
	//QSlider* xSlider;
	//QSlider* ySlider;
	//QSlider* zSlider;
	//QSlider* lensWidSlider;
	//QLabel* statusLabel;

	//Renderable* SlicePlaneRenderable0;
	//Renderable* SlicePlaneRenderable1;
	//Renderable* SlicePlaneRenderable2;
	//Renderable* lensRenderable;
	//Renderable* lineRenderable;

	//std::map < std::string, Renderable* > meshRenderables;
	//std::map < std::string, QCheckBox* > meshCheckBoxes;

	//static const int btnSize[2];// = { 60, 40 };

	//QPushButton* CreateRegularButton(const char* name);
	//QSlider* CreateSliceSlider(TexPlaneRenderable* renderable);


	QPushButton *addCurveBLensBtn;

private slots:
	void AddLens();
	void AddLineLens();
	void AddCurveBLens();

	void adjustOffset();
	void RefineLensBoundary();
//void SlotSliceOrieChanged(bool clicked);
	//void animate();
	//void SlotSetAnimation(bool doAnimation);
	void SlotToggleGrid(bool b);
	void UpdateRightHand(QVector3D thumbTip, QVector3D indexTip, QVector3D indexDir);
	void SlotToggleUsingSnap(bool b);
	void SlotToggleHighlightingFeature(bool b);
	void SlotToggleUsingFeatureSnapping(bool b);

	//void XSliderChanged(int i);
	//void YSliderChanged(int i);
	//void ZSliderChanged(int i);

	//void LensWidSliderChanged(int i);
	//void transSizeSliderChanged(int i);

	//void xSliceToggled(bool b);
	//void ySliceToggled(bool b);
	//void zSliceToggled(bool b);

	//void linesToggled(bool b);
	//void MeshToggled(bool b);

	//void AddLens();
	//void AddLensNode();
	//void DeleteLens();
	//void DeleteLensNode();
	//void SetToNavigation();
	//void UpdateStatusLabel();

	//void RedoDeform();
};

#endif
