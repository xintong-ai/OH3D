#include "window.h"
#include "DeformGLWidget.h"
#include "BoxRenderable.h"
#include "LensRenderable.h"
#include "GridRenderable.h"
#include "ArrowNoDeformRenderable.h"
#include <iostream>
#include <algorithm>    // std::min_element, std::max_element

#include "SphereRenderable.h"
#include "CosmoRenderable.h"

#include "SolutionParticleReader.h"
#include "BinaryParticleReader.h"
#include "DataMgr.h"
#include "ModelGridRenderable.h"
#include <LineSplitModelGrid.h>
#include "GLMatrixManager.h"
#include "PolyRenderable.h"
#include "MeshReader.h"
#include <ColorGradient.h>
#include <Particle.h>
#include <helper_math.h>

#include <ModelGrid.h>

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
	slider->setRange(0, 50);
	slider->setValue(25);
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
	inputParticle = std::make_shared<Particle>();

	if (std::string(dataPath).find(".vtu") != std::string::npos){
		std::shared_ptr<SolutionParticleReader> reader;
		reader = std::make_shared<SolutionParticleReader>(dataPath.c_str(), 130);
		//case study candidata: smoothinglength_0.44/run06/119.vtu, thr 70
		//case study candidata2: smoothinglength_0.44/run11/119.vtu, thr 130

		reader->GetPosRange(posMin, posMax);
		reader->OutputToParticleData(inputParticle);
		reader.reset();

		glyphRenderable = std::make_shared<SphereRenderable>(inputParticle);
		glyphRenderable->setColorMap(COLOR_MAP::RDYIGN, true);
	}
	else{
		std::shared_ptr<BinaryParticleReader> reader;
		reader = std::make_shared<BinaryParticleReader>(dataPath.c_str());

		reader->GetPosRange(posMin, posMax);
		reader->OutputToParticleData(inputParticle);
		reader.reset();
		
		//inputParticle->featureReshuffle();
		//posMin = inputParticle->posMin;
		//posMax = inputParticle->posMax;
		
		//glyphRenderable = std::make_shared<CosmoRenderable>(inputParticle);
		glyphRenderable = std::make_shared<SphereRenderable>(inputParticle);
		glyphRenderable->colorByFeature = true;
		glyphRenderable->setColorMap(COLOR_MAP::RAINBOW_COSMOLOGY);
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

	//vrWidget = std::make_shared<VRWidget>(matrixMgr, openGL.get());
	//vrWidget->setWindowFlags(Qt::Window);
	//std::shared_ptr<DeformGlyphRenderable> glyphRenderable2;
	//if (std::string(dataPath).find(".vtu") != std::string::npos){
	//	glyphRenderable2 = std::make_shared<SphereRenderable>(inputParticle);
	//	glyphRenderable2->setColorMap(COLOR_MAP::RDYIGN, true);
	//}
	//else{
	//	glyphRenderable2 = std::make_shared<SphereRenderable>(inputParticle);
	//	glyphRenderable2->colorByFeature = true;
	//	glyphRenderable2->setColorMap(COLOR_MAP::RAINBOW_COSMOLOGY);
	//}
	//glyphRenderable2->SetDisplace(false);
	//vrWidget->AddRenderable("glyph", glyphRenderable2.get());
	//vrWidget->AddRenderable("lens", lensRenderable.get());
	//openGL->SetVRWidget(vrWidget.get());
#endif
	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setStencilBufferSize(8);
	format.setVersion(2, 0);
	format.setProfile(QSurfaceFormat::CoreProfile);
	openGL->setFormat(format); // must be called before the widget or its parent window gets shown
	//std::shared_ptr<ModelGrid> mgs = std::make_shared<ModelGrid>(&posMin.x, &posMax.x, 22);


	gridRenderable = std::make_shared<GridRenderable>(64);
	matrixMgr->SetVol(posMin, posMax);// cubemap->GetInnerDim());
	modelGrid = std::make_shared<LineSplitModelGrid>(&posMin.x, &posMax.x, meshResolution);
	modelGrid->initThrustVectors(inputParticle);
	modelGridRenderable = std::make_shared<ModelGridRenderable>(modelGrid.get());
	modelGridRenderable->SetLenses(lensRenderable->GetLensesAddr());
	
	glyphRenderable->SetModelGrid(modelGrid.get());
	//openGL->AddRenderable("bbox", bbox);

	//glyphRenderable->SetVisibility(false);
	//lensRenderable->SetVisibility(false);

	openGL->AddRenderable("2glyph", glyphRenderable.get());
	openGL->AddRenderable("3lenses", lensRenderable.get());
	//openGL->AddRenderable("grid", gridRenderable.get());
	openGL->AddRenderable("4model", modelGridRenderable.get());
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
	udbeCheck->setChecked(modelGrid->elasticityMode >0);

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
	
	std::vector<float4> pos;
	pos.push_back(make_float4((posMin + posMax) / 2, 1.0));
	pos.push_back(make_float4((posMin + posMax) / 2, 1.0));
	std::vector<float> val;
	val.push_back(0);
	val.push_back(0);
	leapFingerIndicators = std::make_shared<Particle>(pos, val);
	leapFingerIndicatorVecs.push_back(make_float3(1, 0, 0));
	leapFingerIndicatorVecs.push_back(make_float3(1, 0, 0));  //actaully only the length of these two vectors have an effect

	arrowNoDeformRenderable = std::make_shared<ArrowNoDeformRenderable>(leapFingerIndicatorVecs,leapFingerIndicators);
	//arrowNoDeformRenderable->SetVisibility(false);
	leapFingerIndicators->numParticles = 0; //use numParticles to control how many indicators are drawn on screen
	openGL->AddRenderable("1LeapArrow", arrowNoDeformRenderable.get());


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


	QHBoxLayout *meshResLayout = new QHBoxLayout;
	QLabel *meshResLitLabel = new QLabel(("Mesh resolution:  "));
	QPushButton* addMeshResPushButton = new QPushButton(tr("&+"));
	addMeshResPushButton->setFixedSize(24, 24);
	QPushButton* minusMeshResPushButton = new QPushButton(tr("&-"));
	minusMeshResPushButton->setFixedSize(24, 24);
	meshResLabel = new QLabel(QString::number(modelGrid->meshResolution));
	meshResLayout->addWidget(meshResLitLabel);
	meshResLayout->addWidget(minusMeshResPushButton);
	meshResLayout->addWidget(addMeshResPushButton);
	meshResLayout->addWidget(meshResLabel);
	meshResLayout->addStretch();

	QVBoxLayout *controlLayout = new QVBoxLayout;
	controlLayout->addWidget(addLensBtn);
	controlLayout->addWidget(addLineLensBtn);

	//controlLayout->addWidget(addCurveBLensBtn);
	controlLayout->addWidget(delLensBtn.get());
	controlLayout->addWidget(saveStateBtn.get());
	controlLayout->addWidget(loadStateBtn.get());
	//controlLayout->addWidget(groupBox);
	controlLayout->addWidget(transSizeLabel);
	controlLayout->addWidget(transSizeSlider);
	controlLayout->addWidget(usingGlyphSnappingCheck);
	controlLayout->addWidget(usingGlyphPickingCheck);
	controlLayout->addWidget(freezingFeatureCheck);
	controlLayout->addWidget(usingFeatureSnappingCheck);
	controlLayout->addWidget(usingFeaturePickingCheck); 
	controlLayout->addWidget(gridCheck);
	controlLayout->addWidget(udbeCheck);
	controlLayout->addLayout(meshResLayout);

	
	QLabel *deformForceLabelLit = new QLabel("Deform Force");
	controlLayout->addWidget(deformForceLabelLit);
	QSlider *deformForceSlider = new QSlider(Qt::Horizontal);
	deformForceSlider->setRange(0, 80);
	deformForceSlider->setValue(modelGrid->getDeformForce() / deformForceConstant);
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
	connect(delLensBtn.get(), SIGNAL(clicked()), this, SLOT(SlotDelLens()));
	connect(saveStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotSaveState()));
	connect(loadStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotLoadState()));

	connect(addMeshResPushButton, SIGNAL(clicked()), this, SLOT(SlotAddMeshRes()));
	connect(minusMeshResPushButton, SIGNAL(clicked()), this, SLOT(SlotMinusMeshRes()));

	connect(gridCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleGrid(bool)));
	connect(udbeCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleUdbe(bool)));
	connect(transSizeSlider, SIGNAL(valueChanged(int)), lensRenderable.get(), SLOT(SlotFocusSizeChanged(int)));
#ifdef USE_LEAP
	connect(listener, SIGNAL(UpdateHands(QVector3D, QVector3D, int)),
		this, SLOT(SlotUpdateHands(QVector3D, QVector3D, int)));
#endif
#ifdef USE_NEW_LEAP
	connect(listener, SIGNAL(UpdateHands(QVector3D, QVector3D, QVector3D, QVector3D, int)),
		this, SLOT(SlotUpdateHands(QVector3D, QVector3D, QVector3D, QVector3D, int)));
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
	modelGrid->gridType = GRID_TYPE::UNIFORM_GRID;
	modelGrid->InitializeUniformGrid(inputParticle); //call this function must set gridType = GRID_TYPE::UNIFORM_GRID first
	lensRenderable->AddCircleLens();
}

void Window::AddLineLens()
{
	modelGrid->gridType = GRID_TYPE::LINESPLIT_UNIFORM_GRID;
	lensRenderable->AddLineLens3D();
}


void Window::AddCurveBLens()
{
	lensRenderable->AddCurveBLens();
}

void Window::SlotDelLens()
{
	lensRenderable->SlotDelLens();
	inputParticle->pos = inputParticle->posOrig;
	glyphRenderable->glyphBright.assign(inputParticle->numParticles, 1.0);
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
	if (b)
		modelGrid->elasticityMode = 1;
	else
		modelGrid->elasticityMode = 0;

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
void Window::SlotUpdateHands(QVector3D rightThumbTip, QVector3D rightIndexTip, QVector3D leftThumbTip, QVector3D leftIndexTip, int numHands)
{
	if (1 == numHands){
		float4 markerPos;
		if (lensRenderable->SlotOneHandChanged_lc(make_float3(rightThumbTip.x(), rightThumbTip.y(), rightThumbTip.z()), make_float3(rightIndexTip.x(), rightIndexTip.y(), rightIndexTip.z()), markerPos)){

			openGL->blendOthers = true;
		}
		else{
			openGL->blendOthers = false;
		}
		leapFingerIndicators->numParticles = 1;

		leapFingerIndicators->pos[0] = markerPos;
	}
	else if (2 == numHands){
		
		float4 markerPosRight, markerPosLeft;
		bool forceChanged = false;
		float f = modelGrid->getDeformForce();
		if (lensRenderable->SlotTwoHandChanged_lc(
			make_float3(rightThumbTip.x(), rightThumbTip.y(), rightThumbTip.z()),
			make_float3(rightIndexTip.x(), rightIndexTip.y(), rightIndexTip.z()), 
			make_float3(leftThumbTip.x(), leftThumbTip.y(), leftThumbTip.z()),
			make_float3(leftIndexTip.x(), leftIndexTip.y(), leftIndexTip.z()),
			markerPosRight, markerPosLeft, leapFingerIndicators->val[0], leapFingerIndicators->val[1], f)){
			openGL->blendOthers = true;
			deformForceSliderValueChanged(f / deformForceConstant);
		}
		else{
			openGL->blendOthers = false;
		}
		leapFingerIndicators->numParticles = 2;

		leapFingerIndicators->pos[0] = markerPosRight;
		leapFingerIndicators->pos[1] = markerPosLeft;
	}
}
#endif

void Window::SlotSaveState()
{
	matrixMgr->SaveState("current.state");
	lensRenderable->SaveState("lens.state");

	std::ofstream myfile;
	myfile.open("system.state");

	myfile << modelGrid->getDeformForce() << std::endl;
	myfile << meshResolution << std::endl;
	myfile.close();
}

void Window::SlotLoadState()
{
	matrixMgr->LoadState("current.state");
	lensRenderable->LoadState("lens.state");
	std::ifstream ifs("system.state", std::ifstream::in);
	if (ifs.is_open()) {
		float f;
		int res;
		ifs >> f;
		ifs >> res;

		deformForceSliderValueChanged(f / deformForceConstant);
		meshResolution = res;
		modelGrid->meshResolution = meshResolution;
		modelGrid->setReinitiationNeed();
		meshResLabel->setText(QString::number(meshResolution));
	}
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
	float newForce = deformForceConstant*v;
	deformForceLabel->setText(QString::number(newForce));
	modelGrid->setDeformForce(newForce);
}

void Window::SlotAddMeshRes()
{
	meshResolution++;
	modelGrid->meshResolution = meshResolution;
	modelGrid->setReinitiationNeed();
	meshResLabel->setText(QString::number(meshResolution));
}
void Window::SlotMinusMeshRes()
{
	meshResolution--;
	modelGrid->meshResolution = meshResolution;
	modelGrid->setReinitiationNeed();
	meshResLabel->setText(QString::number(meshResolution));
}