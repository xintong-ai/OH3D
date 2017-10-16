#include "window.h"

#include <iostream>
#include <helper_math.h>

#include "GLWidget.h"
#include "SphereRenderable.h"
#include "GLMatrixManager.h"
#include "ColorGradient.h"
#include "Particle.h"
#include "mouse/RegularInteractor.h"

/*
This is a tutorial program, which renders a group of particles, using a sphere for each particle.

The executable program follows the Model–View–Controller design.
Generally you need to first prepare objects of data (Model);
then prepare objects of the Renderable class (View), and connect it with data objects;
besides QT interaction, you can prepare objects of Interactor class (Controller), and connect it with data objects.
At last, connect the Renderable objects and the Interactor objects using QT
*/

Window::Window()
{
	////////////////////////////	Data Model	////////////////////////////
	//Here we create the objects of the data model. 
	//Existing data model classes are in 'dataModel' folder.
	
	//First we create an object of GLMatrixManager, which stores the matrices used for 3D visualization,
	//such as model, view, projection matrices.
	float3 posMin = make_float3(-10, -10, -10), posMax = make_float3(10, 10, 10);
	matrixMgr = std::make_shared<GLMatrixManager>(posMin, posMax);
	//Next create an object of Particle.
	inputParticle = std::make_shared<Particle>();

	inputParticle->createSyntheticData(posMin, posMax, 100);
	//Here we use the default values for the matrixMgr object, and use a function to set the values of inputParticle
	//In most cases, you need to fill the data object in a certain way


	////////////////////////////	View	////////////////////////////
	//Here we create the views, which generally connects to an data object, and will draw it.
	//Existing view classes are in 'render' folder.

	//Create an object of SphereRenderable, which draw particles as spheres.
	glyphRenderable = std::make_shared<SphereRenderable>(inputParticle);
	glyphRenderable->setColorMap(COLOR_MAP::RDYIGN, true);

	////////////////////////////	Controller	////////////////////////////
	//Here we create the controllers, which generally connects to an data object, 
	//and process interactions from users to apply centain change on the data object
	//Most existing controller classes are in 'interact' folder. 
	//They are further organized by the interaction tools, and placed in subfolders 'mouse', 'Leap', 'touch', etc.
	
	//Create an object of RegularInteractor, which takes interaction from mouse to update objects of GLMatrixManager
	//Therefore it is connected with matrixMgr
	rInteractor = std::make_shared<RegularInteractor>();
	rInteractor->setMatrixMgr(matrixMgr);

	//Some classes involve complex deform computations, and we create helping classes in 'deform' folder
	//They can be treated similarly as a special kind of controllers, 
	//becaues once activated, they automatically modify the data object at each iteration even without further interaction
	//We will not show the example here. Refer to ParticleVis, VolumeVis or TensorVis if interested.

	//Besides our classes, some QT GUI are also essentially controllers.
	//Here is an example of using QPushButton
	changeColorMapBtn = new QPushButton("Change Color Coding");
	connect(changeColorMapBtn, SIGNAL(clicked()), this, SLOT(ChangeColorMap()));


	////////////////////////////	QT GL widget	////////////////////////////
	//We use an object of GLWidget (inherited from QOpenGLWidget). 
	//A class inherited from QOpenGLWidget is frequently used in pupolar QT-OpenGL tutorials

	openGL = std::make_shared<GLWidget>(matrixMgr);
	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setStencilBufferSize(8);
	format.setVersion(2, 0);
	format.setProfile(QSurfaceFormat::CoreProfile);
	openGL->setFormat(format);

	//One of the major goal of using GLWidget is to connect all View objects and Controller objects together
	//During the OpenGl interation, each added View objects and Controller objects will be called
	//Note that when multiple view/controller objects are connected, they are called with an alphabetic order of the assigned name
	openGL->AddRenderable("glyph", glyphRenderable.get());
	openGL->AddInteractor("regular", rInteractor.get());

	//COMING SOON:
	//When multiple interactors are connected and sometimes you only want to trigger one of them,
	//our strategy used in ParticleVis or VolumeVis is, for example when using a mouse,
	//when mouse is pressed, detect which interactor is active, then set the 'isActive' attribute of the interactor,
	//or change the 'interactMode' attribute stored in the glwidget class, then when mouse is moved or released,
	//the interactor can perform required processing.
	//Examples is not given now but will be in the future

	//COMING SOON:
	//more examples about using touchpad, leap motion, and VR headset


	//////////////////////////// QT interfaces
	setWindowTitle(tr("Interactive Glyph Visualization"));
	QHBoxLayout *mainLayout = new QHBoxLayout;
	QVBoxLayout *controlLayout = new QVBoxLayout;
	controlLayout->addWidget(changeColorMapBtn);
	controlLayout->addStretch();
	mainLayout->addWidget(openGL.get(), 3);
	mainLayout->addLayout(controlLayout,1);
	setLayout(mainLayout);
}


Window::~Window() {
}

void Window::init()
{
}


void Window::ChangeColorMap()
{
	int v = rand() % 5;
	switch (v){
	case 0:
		glyphRenderable->setColorMap(COLOR_MAP::RAINBOW, false);
		break;
	case 1:
		glyphRenderable->setColorMap(COLOR_MAP::SIMPLE_BLUE_RED, false);
		break;
	case 2:
		glyphRenderable->setColorMap(COLOR_MAP::BrBG, false);
		break;
	case 3:
		glyphRenderable->setColorMap(COLOR_MAP::RDYIGN, false);
		break;
	case 4:
		glyphRenderable->setColorMap(COLOR_MAP::PU_OR, false);
		break;
	}
}
