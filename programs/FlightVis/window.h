#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <QVector3D>

#include <memory>
#include "CMakeConfig.h"

class DataMgr;
class GLWidget;
class QPushButton;
class QSlider;
class Renderable;
class QCheckBox;
class QLabel;
class GlyphRenderable;
class QRadioButton;
class QTimer;
class DataMgr;
class GLMatrixManager;

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
    std::shared_ptr<GLWidget> openGL;
	QTimer *aTimer;
	const int nScale = 20;
	std::shared_ptr<QRadioButton> radioDeformScreen;
	std::shared_ptr<QRadioButton> radioDeformObject;

	std::shared_ptr<QPushButton> saveStateBtn;
	std::shared_ptr<QPushButton> loadStateBtn;

	QCheckBox* usingGlyphSnappingCheck;
	QCheckBox* usingGlyphPickingCheck;

	std::shared_ptr<GlyphRenderable> glyphRenderable;
	std::shared_ptr<DataMgr> dataMgr;
	std::shared_ptr<GLMatrixManager> matrixMgr;

#ifdef USE_OSVR
	std::shared_ptr<VRWidget> vrWidget;
	std::shared_ptr<VRGlyphRenderable> vrGlyphRenderable;
#endif
#ifdef USE_LEAP
	LeapListener* listener;
	Leap::Controller* controller;
#endif

private slots:
	void SlotSaveState();
	void SlotLoadState();
};

#endif
