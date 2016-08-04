#include "window.h"
#include "DeformGLWidget.h"
#include "BoxRenderable.h"
#include "LensRenderable.h"
#include "GridRenderable.h"
#include <iostream>

#include "SphereRenderable.h"
#include "SolutionParticleReader.h"
#include "RawVolumeReader.h"
#include "Volume.h"

#include "DataMgr.h"
#include "ModelGridRenderable.h"
#include <ModelGrid.h>
#include "GLMatrixManager.h"
#include "PolyRenderable.h"
#include "MeshReader.h"
#include "VecReader.h"
#include "VolumeRenderableCUDA.h"
#include "ModelVolumeDeformer.h"
#include "Lens.h"

#ifdef USE_LEAP
#include <leap/LeapListener.h>
#include <Leap.h>
#endif

#ifdef USE_NEW_LEAP
#include <leap/LeapListener.h>
#include <Leap.h>
#endif

#ifdef USE_OSVR
#include "VRWidget.h"
//#include "VRGlyphRenderable.h"
#include "VRVolumeRenderableCUDA.h"
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
	reader = std::make_shared<SolutionParticleReader>(dataPath.c_str());
	glyphRenderable = std::make_shared<SphereRenderable>(
		((ParticleReader*)reader.get())->GetPos(),
		((ParticleReader*)reader.get())->GetVal());
	std::cout << "number of rendered glyphs: " << (((ParticleReader*)reader.get())->GetVal()).size() << std::endl;
	std::cout << "number of rendered glyphs: " << glyphRenderable->GetNumOfGlyphs() << std::endl;


	const std::string dataPath2 = dataMgr->GetConfig("VOLUME_DATA_PATH");

	int3 dims;
	float3 spacing;
	if (string(dataPath2).find("MGHT2") != std::string::npos){
		dims = make_int3(320, 320, 256);
		spacing = make_float3(0.7, 0.7, 0.7);
	}
	else if (string(dataPath2).find("MGHT1") != std::string::npos){
		dims = make_int3(256, 256, 176);
		spacing = make_float3(1.0, 1.0, 1.0);
	}
	else if (string(dataPath2).find("nek128") != std::string::npos){
		dims = make_int3(128, 128, 128);
		spacing = make_float3(2, 2, 2); //to fit the streamline of nek256
	}
	inputVolume = std::make_shared<Volume>();

	if (string(dataPath2).find(".vec") != std::string::npos){
		std::shared_ptr<VecReader> reader2;
		reader2 = std::make_shared<VecReader>(dataPath2.c_str());
		reader2->OutputToVolumeByNormalizedVecMag(inputVolume);
		reader2.reset();
	}
	else{
		std::shared_ptr<RawVolumeReader> reader2;
		reader2 = std::make_shared<RawVolumeReader>(dataPath2.c_str(), dims);
		reader2->OutputToVolumeByNormalizedValue(inputVolume);
		reader2.reset();
	}
	inputVolume->spacing = spacing;
	inputVolume->initVolumeCuda(0);

	volumeRenderable = std::make_shared<VolumeRenderableCUDA>(inputVolume);

	if (string(dataPath2).find("nek128") != std::string::npos){
		volumeRenderable->useColor = true;
	}
	
	
	/********GL widget******/
#ifdef USE_OSVR
	matrixMgr = std::make_shared<GLMatrixManager>(true);
#else
	matrixMgr = std::make_shared<GLMatrixManager>(false);
#endif
	openGL = std::make_shared<DeformGLWidget>(matrixMgr);
	lensRenderable = std::make_shared<LensRenderable>();
	//lensRenderable->SetDrawScreenSpace(false);
#ifdef USE_OSVR
		vrWidget = std::make_shared<VRWidget>(matrixMgr, openGL.get());
		vrWidget->setWindowFlags(Qt::Window);
		vrGlyphRenderable = std::make_shared<VRVolumeRenderableCUDA>(volumeRenderable.get());
		vrWidget->AddRenderable("glyph", vrGlyphRenderable.get());
		vrWidget->AddRenderable("lens", lensRenderable.get());
		openGL->SetVRWidget(vrWidget.get());
#endif
	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setStencilBufferSize(8);
	format.setVersion(2, 0);
	format.setProfile(QSurfaceFormat::CoreProfile);
	openGL->setFormat(format); // must be called before the widget or its parent window gets shown


	float3 posMin, posMax;
	inputVolume->GetPosRange(posMin, posMax);
	gridRenderable = std::make_shared<GridRenderable>(64);
	matrixMgr->SetVol(posMin, posMax);// cubemap->GetInnerDim());
	modelGrid = std::make_shared<ModelGrid>(&posMin.x, &posMax.x, 20, true);
	modelGridRenderable = std::make_shared<ModelGridRenderable>(modelGrid.get());
	glyphRenderable->SetModelGrid(modelGrid.get());
	glyphRenderable->SetVisibility(false);


	modelVolumeDeformer = std::make_shared<ModelVolumeDeformer>();
	modelVolumeDeformer->SetModelGrid(modelGrid.get());
	modelVolumeDeformer->Init(inputVolume.get());
	volumeRenderable->SetModelVolumeDeformer(modelVolumeDeformer);
	volumeRenderable->lenses = lensRenderable->GetLensesAddr();
	volumeRenderable->SetModelGrid(modelGrid.get());

	openGL->AddRenderable("glyph", glyphRenderable.get());
	openGL->AddRenderable("lenses", lensRenderable.get());
	//openGL->AddRenderable("grid", gridRenderable.get());
	openGL->AddRenderable("1volume", volumeRenderable.get()); //make sure the volume is rendered first since it does not use depth test

	openGL->AddRenderable("model", modelGridRenderable.get());
	///********controls******/
	addLensBtn = new QPushButton("Add circle lens");
	addLineLensBtn = new QPushButton("Add straight band lens");
	delLensBtn = std::make_shared<QPushButton>("Delete a lens");
	addCurveBLensBtn = new QPushButton("Add curved band lens");
	saveStateBtn = std::make_shared<QPushButton>("Save State");
	loadStateBtn = std::make_shared<QPushButton>("Load State");
std::cout << posMin.x << " " << posMin.y << " " << posMin.z << std::endl;
std::cout << posMax.x << " " << posMax.y << " " << posMax.z << std::endl;
	QCheckBox* gridCheck = new QCheckBox("Grid", this);
	QCheckBox* udbeCheck = new QCheckBox("Use Density Based Elasticity", this);
	udbeCheck->setChecked(modelGrid->useDensityBasedElasticity);

	QLabel* transSizeLabel = new QLabel("Transition region size:", this);
	QSlider* transSizeSlider = CreateSlider();
#ifdef USE_LEAP
	listener = new LeapListener();
	controller = new Leap::Controller();
	controller->setPolicyFlags(Leap::Controller::PolicyFlag::POLICY_OPTIMIZE_HMD);
	controller->addListener(*listener);
#endif
#ifdef USE_NEW_LEAP
	listener = new LeapListener();
	controller = new Leap::Controller();
	controller->setPolicyFlags(Leap::Controller::PolicyFlag::POLICY_OPTIMIZE_HMD);
	controller->addListener(*listener);
#endif
	QGroupBox *groupBox = new QGroupBox(tr("Deformation Mode"));
	QHBoxLayout *deformModeLayout = new QHBoxLayout;
	radioDeformScreen = std::make_shared<QRadioButton>(tr("&screen space"));
	radioDeformObject = std::make_shared<QRadioButton>(tr("&object space"));
	radioDeformScreen->setChecked(true);
	deformModeLayout->addWidget(radioDeformScreen.get());
	deformModeLayout->addWidget(radioDeformObject.get());
	groupBox->setLayout(deformModeLayout);

	usingGlyphSnappingCheck = new QCheckBox("Snapping Glyph", this);
	usingGlyphPickingCheck = new QCheckBox("Picking Glyph", this);

	connect(glyphRenderable.get(), SIGNAL(glyphPickingFinished()), this, SLOT(SlotToggleGlyphPickingFinished()));


	QVBoxLayout *controlLayout = new QVBoxLayout;
	controlLayout->addWidget(addLensBtn);
	controlLayout->addWidget(addLineLensBtn);

	controlLayout->addWidget(addCurveBLensBtn);
	controlLayout->addWidget(delLensBtn.get());
	controlLayout->addWidget(saveStateBtn.get());
	controlLayout->addWidget(loadStateBtn.get());
	controlLayout->addWidget(groupBox);
	controlLayout->addWidget(transSizeLabel);
	controlLayout->addWidget(transSizeSlider);
	controlLayout->addWidget(usingGlyphSnappingCheck);
	controlLayout->addWidget(usingGlyphPickingCheck);
	controlLayout->addWidget(gridCheck);
	controlLayout->addWidget(udbeCheck);


	modelGrid->setDeformForce(1024);
	QLabel *deformForceLabelLit = new QLabel("Deform Force");
	controlLayout->addWidget(deformForceLabelLit);
	QSlider *deformForceSlider = new QSlider(Qt::Horizontal);
	deformForceSlider->setRange(12, 48);
	deformForceSlider->setValue(log2(modelGrid->getDeformForce())*4.0);
	connect(deformForceSlider, SIGNAL(valueChanged(int)), this, SLOT(deformForceSliderValueChanged(int)));
	deformForceLabel = new QLabel(QString::number(modelGrid->getDeformForce()));
	QHBoxLayout *deformForceLayout = new QHBoxLayout;
	deformForceLayout->addWidget(deformForceSlider);
	deformForceLayout->addWidget(deformForceLabel);
	controlLayout->addLayout(deformForceLayout);


	QLabel *transFuncP1LabelSliderLabelLit = new QLabel("Transfer Function Higher Cut Off");
	controlLayout->addWidget(transFuncP1LabelSliderLabelLit);
	QSlider *transFuncP1LabelSlider = new QSlider(Qt::Horizontal);
	transFuncP1LabelSlider->setRange(0, 100);
	transFuncP1LabelSlider->setValue(volumeRenderable->transFuncP1 * 100);
	connect(transFuncP1LabelSlider, SIGNAL(valueChanged(int)), this, SLOT(transFuncP1LabelSliderValueChanged(int)));
	transFuncP1Label = new QLabel(QString::number(volumeRenderable->transFuncP1));
	QHBoxLayout *transFuncP1Layout = new QHBoxLayout;
	transFuncP1Layout->addWidget(transFuncP1LabelSlider);
	transFuncP1Layout->addWidget(transFuncP1Label);
	controlLayout->addLayout(transFuncP1Layout);

	QLabel *transFuncP2LabelSliderLabelLit = new QLabel("Transfer Function Lower Cut Off");
	controlLayout->addWidget(transFuncP2LabelSliderLabelLit);
	QSlider *transFuncP2LabelSlider = new QSlider(Qt::Horizontal);
	transFuncP2LabelSlider->setRange(0, 100);
	transFuncP2LabelSlider->setValue(volumeRenderable->transFuncP2 * 100);
	connect(transFuncP2LabelSlider, SIGNAL(valueChanged(int)), this, SLOT(transFuncP2LabelSliderValueChanged(int)));
	transFuncP2Label = new QLabel(QString::number(volumeRenderable->transFuncP2));
	QHBoxLayout *transFuncP2Layout = new QHBoxLayout;
	transFuncP2Layout->addWidget(transFuncP2LabelSlider);
	transFuncP2Layout->addWidget(transFuncP2Label);
	controlLayout->addLayout(transFuncP2Layout);

	QLabel *brLabelLit = new QLabel("Brightness of the volume: ");
	controlLayout->addWidget(brLabelLit);
	QSlider* brSlider = new QSlider(Qt::Horizontal);
	brSlider->setRange(0, 40);
	brSlider->setValue(volumeRenderable->brightness * 20);
	connect(brSlider, SIGNAL(valueChanged(int)), this, SLOT(brSliderValueChanged(int)));
	brLabel = new QLabel(QString::number(volumeRenderable->brightness));
	QHBoxLayout *brLayout = new QHBoxLayout;
	brLayout->addWidget(brSlider);
	brLayout->addWidget(brLabel);
	controlLayout->addLayout(brLayout);

	QLabel *dsLabelLit = new QLabel("Density of the volume: ");
	controlLayout->addWidget(dsLabelLit);
	QSlider* dsSlider = new QSlider(Qt::Horizontal);
	dsSlider->setRange(0, 40);
	dsSlider->setValue(volumeRenderable->density * 20);
	connect(dsSlider, SIGNAL(valueChanged(int)), this, SLOT(dsSliderValueChanged(int)));
	dsLabel = new QLabel(QString::number(volumeRenderable->density));
	QHBoxLayout *dsLayout = new QHBoxLayout;
	dsLayout->addWidget(dsSlider);
	dsLayout->addWidget(dsLabel);
	controlLayout->addLayout(dsLayout);


	QLabel *laSliderLabelLit = new QLabel("Coefficient for Ambient Lighting: ");
	controlLayout->addWidget(laSliderLabelLit);
	QSlider* laSlider = new QSlider(Qt::Horizontal);
	laSlider->setRange(0, 50);
	laSlider->setValue(volumeRenderable->la * 10);
	connect(laSlider, SIGNAL(valueChanged(int)), this, SLOT(laSliderValueChanged(int)));
	laLabel = new QLabel(QString::number(volumeRenderable->la));
	QHBoxLayout *laLayout = new QHBoxLayout;
	laLayout->addWidget(laSlider);
	laLayout->addWidget(laLabel);
	controlLayout->addLayout(laLayout);

	QLabel *ldSliderLabelLit = new QLabel("Coefficient for Diffusial Lighting: ");
	controlLayout->addWidget(ldSliderLabelLit);
	QSlider* ldSlider = new QSlider(Qt::Horizontal);
	ldSlider->setRange(0, 50);
	ldSlider->setValue(volumeRenderable->ld * 10);
	connect(ldSlider, SIGNAL(valueChanged(int)), this, SLOT(ldSliderValueChanged(int)));
	ldLabel = new QLabel(QString::number(volumeRenderable->ld));
	QHBoxLayout *ldLayout = new QHBoxLayout;
	ldLayout->addWidget(ldSlider);
	ldLayout->addWidget(ldLabel);
	controlLayout->addLayout(ldLayout);

	QLabel *lsSliderLabelLit = new QLabel("Coefficient for Specular Lighting: ");
	controlLayout->addWidget(lsSliderLabelLit);
	QSlider* lsSlider = new QSlider(Qt::Horizontal);
	lsSlider->setRange(0, 50);
	lsSlider->setValue(volumeRenderable->ls * 10);
	connect(lsSlider, SIGNAL(valueChanged(int)), this, SLOT(lsSliderValueChanged(int)));
	lsLabel = new QLabel(QString::number(volumeRenderable->ls));
	QHBoxLayout *lsLayout = new QHBoxLayout;
	lsLayout->addWidget(lsSlider);
	lsLayout->addWidget(lsLabel);
	controlLayout->addLayout(lsLayout);

	controlLayout->addStretch();

	connect(addLensBtn, SIGNAL(clicked()), this, SLOT(AddLens()));
	connect(addLineLensBtn, SIGNAL(clicked()), this, SLOT(AddLineLens()));
	connect(addCurveBLensBtn, SIGNAL(clicked()), this, SLOT(AddCurveBLens()));
	connect(delLensBtn.get(), SIGNAL(clicked()), lensRenderable.get(), SLOT(SlotDelLens()));
	connect(saveStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotSaveState()));
	connect(loadStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotLoadState()));

	
	connect(gridCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleGrid(bool)));
	connect(udbeCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleUdbe(bool)));
	connect(transSizeSlider, SIGNAL(valueChanged(int)), lensRenderable.get(), SLOT(SlotFocusSizeChanged(int)));
#ifdef USE_LEAP
	connect(listener, SIGNAL(UpdateHands(QVector3D, QVector3D, int)),
		this, SLOT(SlotUpdateHands(QVector3D, QVector3D, int)));
#endif
#ifdef USE_NEW_LEAP
	connect(listener, SIGNAL(UpdateHands(QVector3D, QVector3D, int)),
		this, SLOT(SlotUpdateHands(QVector3D, QVector3D, int)));
#endif
	connect(usingGlyphSnappingCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleUsingGlyphSnapping(bool)));
	connect(usingGlyphPickingCheck, SIGNAL(clicked(bool)), this, SLOT(SlotTogglePickingGlyph(bool)));
	connect(radioDeformObject.get(), SIGNAL(clicked(bool)), this, SLOT(SlotDeformModeChanged(bool)));
	connect(radioDeformScreen.get(), SIGNAL(clicked(bool)), this, SLOT(SlotDeformModeChanged(bool)));
	
	mainLayout->addWidget(openGL.get(), 3);
	mainLayout->addLayout(controlLayout,1);
	setLayout(mainLayout);
}

void Window::AddLens()
{
	lensRenderable->AddCircleLens();
}

void Window::AddLineLens()
{
	lensRenderable->AddLineLens();
}


void Window::AddCurveBLens()
{
	lensRenderable->AddCurveBLens();
}


void Window::SlotTogglePickingGlyph(bool b)
{
	glyphRenderable->isPickingGlyph = b;
}


void Window::SlotToggleUsingGlyphSnapping(bool b)
{
	lensRenderable->isSnapToGlyph = b;
	if (!b){
		glyphRenderable->SetSnappedGlyphId(-1);
	}
}

void Window::SlotToggleGrid(bool b)
{
	modelGridRenderable->SetVisibility(b);
}

void Window::SlotToggleUdbe(bool b)
{
	modelGrid->useDensityBasedElasticity = b;
	modelGrid->setReinitiationNeed();
	std::vector<Lens*> *lenses = volumeRenderable->lenses;
}

Window::~Window() {
}

void Window::init()
{
#ifdef USE_OSVR
		vrWidget->show();
#endif
}

#ifdef USE_LEAP
void Window::SlotUpdateHands(QVector3D leftIndexTip, QVector3D rightIndexTip, int numHands)
{
	if (1 == numHands){
		lensRenderable->SlotOneHandChanged(make_float3(rightIndexTip.x(), rightIndexTip.y(), rightIndexTip.z()));
	}
	else if(2 == numHands){
		//
		lensRenderable->SlotTwoHandChanged(
			make_float3(leftIndexTip.x(), leftIndexTip.y(), leftIndexTip.z()),
			make_float3(rightIndexTip.x(), rightIndexTip.y(), rightIndexTip.z()));
		
	}
}
#endif
#ifdef USE_NEW_LEAP
void Window::SlotUpdateHands(QVector3D leftIndexTip, QVector3D rightIndexTip, int numHands)
{
	if (1 == numHands){
		lensRenderable->SlotOneHandChanged(make_float3(rightIndexTip.x(), rightIndexTip.y(), rightIndexTip.z()));
	}
	else if (2 == numHands){
		//
		lensRenderable->SlotTwoHandChanged(
			make_float3(leftIndexTip.x(), leftIndexTip.y(), leftIndexTip.z()),
			make_float3(rightIndexTip.x(), rightIndexTip.y(), rightIndexTip.z()));

	}
}
#endif

void Window::SlotSaveState()
{
	matrixMgr->SaveState("current.state");
}

void Window::SlotLoadState()
{
	matrixMgr->LoadState("current.state");
}


void Window::SlotToggleGlyphPickingFinished()
{
	usingGlyphPickingCheck->setChecked(false);
}

void Window::SlotDeformModeChanged(bool clicked)
{
	if (radioDeformScreen->isChecked()){
		openGL->SetDeformModel(DEFORM_MODEL::SCREEN_SPACE);
	}
	else if (radioDeformObject->isChecked()){
		openGL->SetDeformModel(DEFORM_MODEL::OBJECT_SPACE);
	}
}

void Window::deformForceSliderValueChanged(int v)
{
	float newForce = pow(2, v / 4.0);
	deformForceLabel->setText(QString::number(newForce));
	modelGrid->setDeformForce(newForce);
}



void Window::transFuncP1LabelSliderValueChanged(int v)
{
	volumeRenderable->transFuncP1 = 1.0*v / 100;
	transFuncP1Label->setText(QString::number(1.0*v / 100));
}
void Window::transFuncP2LabelSliderValueChanged(int v)
{
	volumeRenderable->transFuncP2 = 1.0*v / 100;
	transFuncP2Label->setText(QString::number(1.0*v / 100));
}

void Window::brSliderValueChanged(int v)
{
	volumeRenderable->brightness = v*1.0 / 20.0;
	brLabel->setText(QString::number(volumeRenderable->brightness));
}
void Window::dsSliderValueChanged(int v)
{
	volumeRenderable->density = v*1.0 / 20.0;
	dsLabel->setText(QString::number(volumeRenderable->density));
}

void Window::laSliderValueChanged(int v)
{
	volumeRenderable->la = 1.0*v / 10;
	laLabel->setText(QString::number(1.0*v / 10));

}
void Window::ldSliderValueChanged(int v)
{
	volumeRenderable->ld = 1.0*v / 10;
	ldLabel->setText(QString::number(1.0*v / 10));
}
void Window::lsSliderValueChanged(int v)
{
	volumeRenderable->ls = 1.0*v / 10;
	lsLabel->setText(QString::number(1.0*v / 10));
}