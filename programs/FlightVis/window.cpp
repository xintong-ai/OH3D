#include "window.h"
#include "glwidget.h"
#include "BoxRenderable.h"
#include "LensRenderable.h"

#include "SphereRenderable.h"
#include "FlightReader.h"
#include "DataMgr.h"
#include "GLMatrixManager.h"
#include "PolyRenderable.h"
#include "MeshReader.h"

#ifdef USE_LEAP
#include <leap/LeapListener.h>
#include <Leap.h>
#endif

#ifdef USE_OSVR
#include "VRWidget.h"
#include "VRGlyphRenderable.h"
#endif

QSlider* CreateSlider()
{
	QSlider* slider = new QSlider(Qt::Horizontal);
	slider->setRange(0, 10);
	slider->setValue(5);
	return slider;
}

class GLTextureCube;
Window::Window()
{
    setWindowTitle(tr("Interactive Glyph Visualization"));
	QHBoxLayout *mainLayout = new QHBoxLayout;

	dataMgr = std::make_shared<DataMgr>();
	
	std::shared_ptr<Reader> reader;

	const std::string dataPath = dataMgr->GetConfig("DATA_PATH");
	reader = std::make_shared<FlightReader>(dataPath.c_str());
	glyphRenderable = std::make_shared<SphereRenderable>(
		((FlightReader*)reader.get())->GetPos(),
		((FlightReader*)reader.get())->GetVal());
	glyphRenderable->SetGlyphSizeAdjust(6);
	//glyphRenderable->EnableDisplace(false);
	std::cout << "number of rendered glyphs: " << (((FlightReader*)reader.get())->GetVal()).size() << std::endl;
	std::cout << "number of rendered glyphs: " << glyphRenderable->GetNumOfGlyphs() << std::endl;

	/********GL widget******/
#ifdef USE_OSVR
	matrixMgr = std::make_shared<GLMatrixManager>(true);
#else
	matrixMgr = std::make_shared<GLMatrixManager>(false);
#endif
	openGL = std::make_shared<GLWidget>(matrixMgr);
#ifdef USE_OSVR
		vrWidget = std::make_shared<VRWidget>(matrixMgr, openGL.get());
		vrWidget->setWindowFlags(Qt::Window);
		vrGlyphRenderable = std::make_shared<VRGlyphRenderable>(glyphRenderable.get());
		vrWidget->AddRenderable("glyph", vrGlyphRenderable.get());
		openGL->SetVRWidget(vrWidget.get());
#endif
	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setStencilBufferSize(8);
#ifdef _WIN32
	format.setVersion(2, 0);
#elif __APPLE__
	format.setVersion(4, 1);
#endif
	format.setProfile(QSurfaceFormat::CoreProfile);
	openGL->setFormat(format); // must be called before the widget or its parent window gets shown

	float3 posMin, posMax;
	reader->GetPosRange(posMin, posMax);
	matrixMgr->SetVol(posMin, posMax);// cubemap->GetInnerDim());
	//openGL->AddRenderable("bbox", bbox);
	openGL->AddRenderable("glyph", glyphRenderable.get());
	///********controls******/
	saveStateBtn = std::make_shared<QPushButton>("Save State");
	loadStateBtn = std::make_shared<QPushButton>("Load State");

	QLabel* transSizeLabel = new QLabel("Transition region size:", this);
	QSlider* transSizeSlider = CreateSlider();
#ifdef USE_LEAP
	listener = new LeapListener();
	controller = new Leap::Controller();
	controller->setPolicyFlags(Leap::Controller::PolicyFlag::POLICY_OPTIMIZE_HMD);
	controller->addListener(*listener);
#endif

	QGroupBox *groupBox = new QGroupBox(tr("Deformation Mode"));
	QHBoxLayout *deformModeLayout = new QHBoxLayout;
	groupBox->setLayout(deformModeLayout);

	QVBoxLayout *controlLayout = new QVBoxLayout;
	controlLayout->addWidget(saveStateBtn.get());
	controlLayout->addWidget(loadStateBtn.get());
	controlLayout->addWidget(groupBox);
	controlLayout->addStretch();

	connect(saveStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotSaveState()));
	connect(loadStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotLoadState()));
	
	mainLayout->addWidget(openGL.get(), 3);
	mainLayout->addLayout(controlLayout,1);
	setLayout(mainLayout);
}
Window::~Window() {
}

void Window::init()
{
#ifdef USE_OSVR
		vrWidget->show();
#endif
}

void Window::SlotSaveState()
{
	matrixMgr->SaveState("current.state");
}

void Window::SlotLoadState()
{
	matrixMgr->LoadState("current.state");
}
