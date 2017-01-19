#include "window.h"
#include "DeformGLWidget.h"
#include <iostream>

#include "Volume.h"

#include "GLMatrixManager.h"

#include "VolumeRenderableCUDA.h"
#include "mouse/ImmersiveInteractor.h"

#ifdef USE_OSVR
#include "VRWidget.h"
#include "VRVolumeRenderableCUDA.h"
#endif



Window::Window(std::shared_ptr<Volume> v)
{
    setWindowTitle(tr("Interactive Glyph Visualization"));
	QHBoxLayout *mainLayout = new QHBoxLayout;

	inputVolume = v;

	/********GL widget******/
#ifdef USE_OSVR
	matrixMgr = std::make_shared<GLMatrixManager>(true);
#else
	matrixMgr = std::make_shared<GLMatrixManager>(false);
#endif

	bool isImmersive =  true;
	if (isImmersive){
		matrixMgr->SetImmersiveMode();
	}

	openGL = std::make_shared<DeformGLWidget>(matrixMgr);
	openGL->SetDeformModel(DEFORM_MODEL::OBJECT_SPACE);


	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setStencilBufferSize(8);
	format.setVersion(2, 0);
	format.setProfile(QSurfaceFormat::CoreProfile);
	openGL->setFormat(format); // must be called before the widget or its parent window gets shown


	float3 posMin, posMax;
	inputVolume->GetPosRange(posMin, posMax);
	matrixMgr->SetVol(posMin, posMax);
	
	
	volumeRenderable = std::make_shared<VolumeRenderableCUDA>(inputVolume);
	openGL->AddRenderable("1volume", volumeRenderable.get()); //make sure the volume is rendered first since it does not use depth test

	volumeRenderable->rcp = RayCastingParameters(1.0, 0.2, 0.7, 0.44, 0.29, 1.25, 512, 0.25f, 1.3, false);


	if (isImmersive){
		immersiveInteractor = std::make_shared<ImmersiveInteractor>();
		immersiveInteractor->setMatrixMgr(matrixMgr);
		openGL->AddInteractor("model", immersiveInteractor.get());
	}




	///********controls******/

	saveStateBtn = std::make_shared<QPushButton>("Save State");
	loadStateBtn = std::make_shared<QPushButton>("Load State");
std::cout << posMin.x << " " << posMin.y << " " << posMin.z << std::endl;
std::cout << posMax.x << " " << posMax.y << " " << posMax.z << std::endl;

	QVBoxLayout *controlLayout = new QVBoxLayout;

	controlLayout->addWidget(saveStateBtn.get());
	controlLayout->addWidget(loadStateBtn.get());



	QLabel *transFuncP1SliderLabelLit = new QLabel("Transfer Function Higher Cut Off");
	//controlLayout->addWidget(transFuncP1SliderLabelLit);
	QSlider *transFuncP1LabelSlider = new QSlider(Qt::Horizontal);
	transFuncP1LabelSlider->setRange(0, 100);
	transFuncP1LabelSlider->setValue(volumeRenderable->rcp.transFuncP1 * 100);
	connect(transFuncP1LabelSlider, SIGNAL(valueChanged(int)), this, SLOT(transFuncP1LabelSliderValueChanged(int)));
	transFuncP1Label = new QLabel(QString::number(volumeRenderable->rcp.transFuncP1));
	QHBoxLayout *transFuncP1Layout = new QHBoxLayout;
	transFuncP1Layout->addWidget(transFuncP1LabelSlider);
	transFuncP1Layout->addWidget(transFuncP1Label);
	//controlLayout->addLayout(transFuncP1Layout);

	QLabel *transFuncP2SliderLabelLit = new QLabel("Transfer Function Lower Cut Off");
	//controlLayout->addWidget(transFuncP2SliderLabelLit);
	QSlider *transFuncP2LabelSlider = new QSlider(Qt::Horizontal);
	transFuncP2LabelSlider->setRange(0, 100);
	transFuncP2LabelSlider->setValue(volumeRenderable->rcp.transFuncP2 * 100);
	connect(transFuncP2LabelSlider, SIGNAL(valueChanged(int)), this, SLOT(transFuncP2LabelSliderValueChanged(int)));
	transFuncP2Label = new QLabel(QString::number(volumeRenderable->rcp.transFuncP2));
	QHBoxLayout *transFuncP2Layout = new QHBoxLayout;
	transFuncP2Layout->addWidget(transFuncP2LabelSlider);
	transFuncP2Layout->addWidget(transFuncP2Label);
	//controlLayout->addLayout(transFuncP2Layout);

	QLabel *brLabelLit = new QLabel("Brightness of the volume: ");
	//controlLayout->addWidget(brLabelLit);
	QSlider* brSlider = new QSlider(Qt::Horizontal);
	brSlider->setRange(0, 40);
	brSlider->setValue(volumeRenderable->rcp.brightness * 20);
	connect(brSlider, SIGNAL(valueChanged(int)), this, SLOT(brSliderValueChanged(int)));
	brLabel = new QLabel(QString::number(volumeRenderable->rcp.brightness));
	QHBoxLayout *brLayout = new QHBoxLayout;
	brLayout->addWidget(brSlider);
	brLayout->addWidget(brLabel);
	//controlLayout->addLayout(brLayout);

	QLabel *dsLabelLit = new QLabel("Density of the volume: ");
	//controlLayout->addWidget(dsLabelLit);
	QSlider* dsSlider = new QSlider(Qt::Horizontal);
	dsSlider->setRange(0, 40);
	dsSlider->setValue(volumeRenderable->rcp.density * 20);
	connect(dsSlider, SIGNAL(valueChanged(int)), this, SLOT(dsSliderValueChanged(int)));
	dsLabel = new QLabel(QString::number(volumeRenderable->rcp.density));
	QHBoxLayout *dsLayout = new QHBoxLayout;
	dsLayout->addWidget(dsSlider);
	dsLayout->addWidget(dsLabel);
	//controlLayout->addLayout(dsLayout);


	QLabel *laSliderLabelLit = new QLabel("Coefficient for Ambient Lighting: ");
	//controlLayout->addWidget(laSliderLabelLit);
	QSlider* laSlider = new QSlider(Qt::Horizontal);
	laSlider->setRange(0, 50);
	laSlider->setValue(volumeRenderable->rcp.la * 10);
	connect(laSlider, SIGNAL(valueChanged(int)), this, SLOT(laSliderValueChanged(int)));
	laLabel = new QLabel(QString::number(volumeRenderable->rcp.la));
	QHBoxLayout *laLayout = new QHBoxLayout;
	laLayout->addWidget(laSlider);
	laLayout->addWidget(laLabel);
	//controlLayout->addLayout(laLayout);

	QLabel *ldSliderLabelLit = new QLabel("Coefficient for Diffusial Lighting: ");
	//controlLayout->addWidget(ldSliderLabelLit);
	QSlider* ldSlider = new QSlider(Qt::Horizontal);
	ldSlider->setRange(0, 50);
	ldSlider->setValue(volumeRenderable->rcp.ld * 10);
	connect(ldSlider, SIGNAL(valueChanged(int)), this, SLOT(ldSliderValueChanged(int)));
	ldLabel = new QLabel(QString::number(volumeRenderable->rcp.ld));
	QHBoxLayout *ldLayout = new QHBoxLayout;
	ldLayout->addWidget(ldSlider);
	ldLayout->addWidget(ldLabel);
	//controlLayout->addLayout(ldLayout);

	QLabel *lsSliderLabelLit = new QLabel("Coefficient for Specular Lighting: ");
	//controlLayout->addWidget(lsSliderLabelLit);
	QSlider* lsSlider = new QSlider(Qt::Horizontal);
	lsSlider->setRange(0, 50);
	lsSlider->setValue(volumeRenderable->rcp.ls * 10);
	connect(lsSlider, SIGNAL(valueChanged(int)), this, SLOT(lsSliderValueChanged(int)));
	lsLabel = new QLabel(QString::number(volumeRenderable->rcp.ls));
	QHBoxLayout *lsLayout = new QHBoxLayout;
	lsLayout->addWidget(lsSlider);
	lsLayout->addWidget(lsLabel);
	//controlLayout->addLayout(lsLayout);


	QGroupBox *rcGroupBox = new QGroupBox(tr("Ray Casting setting"));
	QVBoxLayout *rcLayout = new QVBoxLayout;
	rcLayout->addWidget(transFuncP1SliderLabelLit);
	rcLayout->addLayout(transFuncP1Layout); 
	rcLayout->addWidget(transFuncP2SliderLabelLit);
	rcLayout->addLayout(transFuncP2Layout);
	rcLayout->addWidget(brLabelLit);
	rcLayout->addLayout(brLayout); 
	rcLayout->addWidget(dsLabelLit);
	rcLayout->addLayout(dsLayout); 
	rcLayout->addWidget(laSliderLabelLit);
	rcLayout->addLayout(laLayout);
	rcLayout->addWidget(ldSliderLabelLit);
	rcLayout->addLayout(ldLayout);
	rcLayout->addWidget(lsSliderLabelLit);
	rcLayout->addLayout(lsLayout);
	rcGroupBox->setLayout(rcLayout);

	controlLayout->addWidget(rcGroupBox);

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


}

void Window::SlotLoadState()
{

}




void Window::transFuncP1LabelSliderValueChanged(int v)
{
	volumeRenderable->rcp.transFuncP1 = 1.0*v / 100;
	transFuncP1Label->setText(QString::number(1.0*v / 100));
}
void Window::transFuncP2LabelSliderValueChanged(int v)
{
	volumeRenderable->rcp.transFuncP2 = 1.0*v / 100;
	transFuncP2Label->setText(QString::number(1.0*v / 100));
}

void Window::brSliderValueChanged(int v)
{
	volumeRenderable->rcp.brightness = v*1.0 / 20.0;
	brLabel->setText(QString::number(volumeRenderable->rcp.brightness));
}
void Window::dsSliderValueChanged(int v)
{
	volumeRenderable->rcp.density = v*1.0 / 20.0;
	dsLabel->setText(QString::number(volumeRenderable->rcp.density));
}

void Window::laSliderValueChanged(int v)
{
	volumeRenderable->rcp.la = 1.0*v / 10;
	laLabel->setText(QString::number(1.0*v / 10));

}
void Window::ldSliderValueChanged(int v)
{
	volumeRenderable->rcp.ld = 1.0*v / 10;
	ldLabel->setText(QString::number(1.0*v / 10));
}
void Window::lsSliderValueChanged(int v)
{
	volumeRenderable->rcp.ls = 1.0*v / 10;
	lsLabel->setText(QString::number(1.0*v / 10));
}
