#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>

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
class SphereRenderable;
class QRadioButton;
class QTimer;
class LensRenderable;
class Window : public QWidget
{
	Q_OBJECT	//without this line, the slot does not work
public:
    Window();
    ~Window();

private:
    GLWidget *openGL;
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
	SphereRenderable* sphereRenderable;
	LensRenderable* lensRenderable;
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

private slots:
	void AddLens();
//void SlotSliceOrieChanged(bool clicked);
	//void animate();
	//void SlotSetAnimation(bool doAnimation);
	
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
