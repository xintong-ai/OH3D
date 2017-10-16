#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <QVector3D>
#include <memory>
#include <vector_types.h>

class QPushButton;
class GlyphRenderable;
class GLMatrixManager;
class Particle;
class RegularInteractor;
class GLWidget;

class Window : public QWidget
{
	Q_OBJECT	//without this line, the slot does not work
public:
    Window();
    ~Window();
	void init();

private:
	std::shared_ptr<GLWidget> openGL;

	std::shared_ptr<GLMatrixManager> matrixMgr;
	std::shared_ptr<Particle> inputParticle;
	std::shared_ptr<RegularInteractor> rInteractor;
	std::shared_ptr<GlyphRenderable> glyphRenderable;

	QPushButton* changeColorMapBtn;
	
private slots:
	void ChangeColorMap();
};

#endif
