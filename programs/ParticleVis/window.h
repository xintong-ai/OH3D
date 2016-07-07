#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
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

#ifdef USE_LEAP
class LeapListener;
namespace Leap{
	class Controller;
}
#endif

#ifdef USE_OSVR
class VRWidget;
class VRGlyphRenderable;
#endif

class Window : public QWidget
{
	Q_OBJECT	//without this line, the slot does not work
public:
    Window();
    ~Window();
	void init();

private:
    std::unique_ptr<GLWidget> openGL;
	QTimer *aTimer;
	const int nScale = 20;
	QPushButton* addLensBtn;
	QPushButton* addLineLensBtn;

	std::unique_ptr<QPushButton> delLensBtn;
	std::unique_ptr<QRadioButton> radioDeformScreen;
	std::unique_ptr<QRadioButton> radioDeformObject;

	std::unique_ptr<QPushButton> saveStateBtn;
	std::unique_ptr<QPushButton> loadStateBtn;

	QCheckBox* usingGlyphSnappingCheck;
	QCheckBox* usingGlyphPickingCheck;

	std::unique_ptr<GlyphRenderable> glyphRenderable;
	std::unique_ptr<LensRenderable> lensRenderable;
	std::unique_ptr<GridRenderable> gridRenderable;
	std::unique_ptr<ModelGridRenderable> modelGridRenderable;
	std::unique_ptr<DataMgr> dataMgr;
	std::shared_ptr<GLMatrixManager> matrixMgr;
	QPushButton *addCurveBLensBtn;
	std::shared_ptr<ModelGrid> modelGrid;

#ifdef USE_OSVR
	std::unique_ptr<VRWidget> vrWidget;
	std::unique_ptr<VRGlyphRenderable> vrGlyphRenderable;
#endif
#ifdef USE_LEAP
	LeapListener* listener;
	Leap::Controller* controller;
#endif

private slots:
	void AddLens();
	void AddLineLens();
	void AddCurveBLens();
	void SlotToggleGrid(bool b);
	void SlotToggleUsingGlyphSnapping(bool b);
	void SlotTogglePickingGlyph(bool b);
	void SlotToggleGlyphPickingFinished();
	void SlotDeformModeChanged(bool clicked);
	void SlotSaveState();
	void SlotLoadState();
#ifdef USE_LEAP
	void SlotUpdateHands(QVector3D leftIndexTip, QVector3D rightIndexTip, int numHands);
#endif
};

#endif
