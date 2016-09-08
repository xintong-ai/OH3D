#include "window.h"
#include "DeformGLWidget.h"
#include "BoxRenderable.h"
#include "LensRenderable.h"
#include "GridRenderable.h"
#include <iostream>
#include <algorithm>    // std::min_element, std::max_element

#include "SphereRenderable.h"
#include "SolutionParticleReader.h"
#include "DataMgr.h"
#include "ModelGridRenderable.h"
#include <LineSplitModelGrid.h>
#include "GLMatrixManager.h"
#include "PolyRenderable.h"
#include "MeshReader.h"
#include <ColorGradient.h>
#include <Particle.h>

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
	
	const std::string dataPath = dataMgr->GetConfig("DATA_PATH");


	float3 posMin, posMax;
	inputParticle;


	if (std::string(dataPath).find(".vtu") != std::string::npos){
		std::shared_ptr<SolutionParticleReader> reader;
		reader = std::make_shared<SolutionParticleReader>(dataPath.c_str());

		reader->GetPosRange(posMin, posMax);

		inputParticle = std::make_shared<Particle>();
		reader->OutputToParticleData(inputParticle);

		glyphRenderable = std::make_shared<SphereRenderable>(inputParticle);

		reader.reset();
	}
	else{
		FILE *pFile;
		pFile = fopen(dataPath.c_str(), "rb");
		if (pFile == NULL) { fputs("particle file error", stderr); exit(1); }
		int numParticles;
		fread(&numParticles, sizeof(int), 1, pFile);
		float *coords = new float[numParticles * 4];
		fread(coords, sizeof(float), numParticles * 4, pFile);
		std::vector<float4> posVec;
		std::vector<float> valVec;
		std::vector<char> feature;

		posVec.resize(numParticles);
		valVec.resize(numParticles);
		feature.resize(numParticles);
		for (int i = 0; i < numParticles; i++){
			posVec[i] = make_float4(coords[4 * i], coords[4 * i + 1], coords[4 * i + 2], 1.0);
			valVec[i] = coords[4 * i + 3];
			feature[i] = (char)coords[4 * i + 3];
		}
		delete[] coords;
		
		inputParticle = std::make_shared<Particle>(posVec, valVec);
		inputParticle->setFeature(feature);
		posMax = inputParticle->posMax;
		posMin = inputParticle->posMin;

		glyphRenderable = std::make_shared<SphereRenderable>(inputParticle);
		glyphRenderable->resetColorMap(COLOR_MAP::RAINBOW_COSMOLOGY);
	}

	std::cout << "number of rendered glyphs: " << inputParticle->numParticles << std::endl;

	/********GL widget******/
#ifdef USE_OSVR
	matrixMgr = std::make_shared<GLMatrixManager>(true);
#else
	matrixMgr = std::make_shared<GLMatrixManager>(false);
#endif
	openGL = std::make_shared<DeformGLWidget>(matrixMgr);
	lensRenderable = std::make_shared<LensRenderable>();
	glyphRenderable->lenses = lensRenderable->GetLensesAddr();

	//lensRenderable->SetDrawScreenSpace(false);
#ifdef USE_OSVR
		vrWidget = std::make_shared<VRWidget>(matrixMgr, openGL.get());
		vrWidget->setWindowFlags(Qt::Window);
		vrGlyphRenderable = std::make_shared<VRGlyphRenderable>(glyphRenderable.get());
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



	gridRenderable = std::make_shared<GridRenderable>(64);
	matrixMgr->SetVol(posMin, posMax);// cubemap->GetInnerDim());
	modelGrid = std::make_shared<LineSplitModelGrid>(&posMin.x, &posMax.x, 20);
	modelGrid->initThrustVectors(inputParticle);
	modelGridRenderable = std::make_shared<ModelGridRenderable>(modelGrid.get());
	modelGridRenderable->SetLenses(lensRenderable->GetLensesAddr());
	
	glyphRenderable->SetModelGrid(modelGrid.get());
	//openGL->AddRenderable("bbox", bbox);
	openGL->AddRenderable("glyph", glyphRenderable.get());
	openGL->AddRenderable("lenses", lensRenderable.get());
	//openGL->AddRenderable("grid", gridRenderable.get());
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
	radioDeformObject->setChecked(true);
	openGL->SetDeformModel(DEFORM_MODEL::OBJECT_SPACE);
	deformModeLayout->addWidget(radioDeformScreen.get());
	deformModeLayout->addWidget(radioDeformObject.get());
	groupBox->setLayout(deformModeLayout);

	usingGlyphSnappingCheck = new QCheckBox("Snapping Glyph", this);
	usingGlyphPickingCheck = new QCheckBox("Picking Glyph", this);
	freezingFeatureCheck = new QCheckBox("Freezing Feature", this);
	usingFeatureSnappingCheck = new QCheckBox("Snapping Feature", this);
	usingFeaturePickingCheck = new QCheckBox("Picking Feature", this);

	connect(glyphRenderable.get(), SIGNAL(glyphPickingFinished()), this, SLOT(SlotToggleGlyphPickingFinished()));
	connect(glyphRenderable.get(), SIGNAL(featurePickingFinished()), this, SLOT(SlotToggleFeaturePickingFinished()));


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
	controlLayout->addWidget(freezingFeatureCheck);
	controlLayout->addWidget(usingFeatureSnappingCheck);
	controlLayout->addWidget(usingFeaturePickingCheck); 
	controlLayout->addWidget(gridCheck);
	controlLayout->addWidget(udbeCheck);
	
	
	QLabel *deformForceLabelLit = new QLabel("Deform Force");
	controlLayout->addWidget(deformForceLabelLit);
	QSlider *deformForceSlider = new QSlider(Qt::Horizontal);
	deformForceSlider->setRange(0, 44);
	deformForceSlider->setValue(log2(modelGrid->getDeformForce())*4.0);
	connect(deformForceSlider, SIGNAL(valueChanged(int)), this, SLOT(deformForceSliderValueChanged(int)));
	deformForceLabel = new QLabel(QString::number(modelGrid->getDeformForce()));
	QHBoxLayout *deformForceLayout = new QHBoxLayout;
	deformForceLayout->addWidget(deformForceSlider);
	deformForceLayout->addWidget(deformForceLabel);
	controlLayout->addLayout(deformForceLayout);
	
	
	
	
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
	connect(freezingFeatureCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleFreezingFeature(bool)));
	connect(usingFeatureSnappingCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleUsingFeatureSnapping(bool)));
	connect(usingFeaturePickingCheck, SIGNAL(clicked(bool)), this, SLOT(SlotTogglePickingFeature(bool)));
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
	lensRenderable->AddLineLens3D();
}


void Window::AddCurveBLens()
{
	lensRenderable->AddCurveBLens();
}

void Window::SlotToggleUsingGlyphSnapping(bool b)
{
	lensRenderable->isSnapToGlyph = b;
	if (!b){
		glyphRenderable->SetSnappedGlyphId(-1);
	}
	else{
		usingFeatureSnappingCheck->setChecked(false);
		SlotToggleUsingFeatureSnapping(false);
		usingFeaturePickingCheck->setChecked(false);
		SlotTogglePickingFeature(false);
	}
}

void Window::SlotTogglePickingGlyph(bool b)
{
	glyphRenderable->isPickingGlyph = b;
	if (b){
		usingFeatureSnappingCheck->setChecked(false);
		SlotToggleUsingFeatureSnapping(false);
		usingFeaturePickingCheck->setChecked(false);
		SlotTogglePickingFeature(false);
	}
}


void Window::SlotToggleFreezingFeature(bool b)
{
	glyphRenderable->isFreezingFeature = b;
	glyphRenderable->RecomputeTarget();
}

void Window::SlotToggleUsingFeatureSnapping(bool b)
{
	lensRenderable->isSnapToFeature = b;
	if (!b){
		glyphRenderable->SetSnappedFeatureId(-1);
		glyphRenderable->RecomputeTarget();
	}
	else{
		usingGlyphSnappingCheck->setChecked(false);
		SlotToggleUsingGlyphSnapping(false);
		usingGlyphPickingCheck->setChecked(false);
		SlotTogglePickingGlyph(false);
	}
}

void Window::SlotTogglePickingFeature(bool b)
{
	glyphRenderable->isPickingFeature = b;
	if (b){
		usingGlyphSnappingCheck->setChecked(false);
		SlotToggleUsingGlyphSnapping(false);
		usingGlyphPickingCheck->setChecked(false);
		SlotTogglePickingGlyph(false);
	}
	else{
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
	
	//modelGrid->SetElasticityForParticle(inputParticle);
	//modelGrid->UpdateMeshDevElasticity();
	//comparing to reinitiate the whole mesh, this does not work well
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

void Window::SlotToggleFeaturePickingFinished()
{
	usingFeaturePickingCheck->setChecked(false);
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
