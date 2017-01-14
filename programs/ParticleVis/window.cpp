#include "window.h"
#include "DeformGLWidget.h"
#include "LensRenderable.h"
#include "ArrowRenderable.h"
#include <iostream>
#include <algorithm>    // std::min_element, std::max_element
#include <helper_math.h>

#include "SphereRenderable.h"
#include "CosmoRenderable.h"

#include "SolutionParticleReader.h"
#include "BinaryParticleReader.h"
#include "DataMgr.h"
#include "MeshRenderable.h"
#include "MeshDeformProcessor.h"
#include "GLMatrixManager.h"
#include "PolyRenderable.h"
#include "MeshReader.h"
#include "ColorGradient.h"
#include "Particle.h"

#include "ScreenLensDisplaceProcessor.h"
#include "PhysicalParticleDeformProcessor.h"
#include "mouse/RegularInteractor.h"
#include "mouse/LensInteractor.h"

#include <CMakeConfig.h>

#ifdef USE_LEAP
#include <leap/LeapListener.h>
#include <Leap.h>
#endif

#ifdef USE_NEW_LEAP
#include <LeapListener.h>
#include <Leap.h>
#endif

#ifdef USE_OSVR
#include "VRWidget.h"
#include "VRGlyphRenderable.h"
#endif



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
		reader = std::make_shared<SolutionParticleReader>(dataPath.c_str(),130);		//case study candidata: smoothinglength_0.44/run06/119.vtu, thr 70
		//case study candidata2: smoothinglength_0.44/run11/119.vtu, thr 130
		reader->GetPosRange(posMin, posMax);
		reader->OutputToParticleData(inputParticle);
		reader.reset();
	}
	else{
		std::shared_ptr<BinaryParticleReader> reader;
		reader = std::make_shared<BinaryParticleReader>(dataPath.c_str());
		reader->GetPosRange(posMin, posMax);
		reader->OutputToParticleData(inputParticle);
		reader.reset();
		deformForceConstant = 2;
		//inputParticle->featureReshuffle();
	}

	std::cout << "number of rendered glyphs: " << inputParticle->numParticles << std::endl;

	/********GL widget******/
#ifdef USE_OSVR
	matrixMgr = std::make_shared<GLMatrixManager>(true);
#else
	matrixMgr = std::make_shared<GLMatrixManager>(false);
#endif
	matrixMgr->SetVol(posMin, posMax);

	openGL = std::make_shared<DeformGLWidget>(matrixMgr);
	openGL->SetDeformModel(DEFORM_MODEL::SCREEN_SPACE);
	
	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setStencilBufferSize(8);
	format.setVersion(2, 0);
	format.setProfile(QSurfaceFormat::CoreProfile);
	openGL->setFormat(format); // must be called before the widget or its parent window gets shown

	screenLensDisplaceProcessor = std::make_shared<ScreenLensDisplaceProcessor>(&lenses, inputParticle);
	meshDeformer = std::make_shared<MeshDeformProcessor>(&posMin.x, &posMax.x, meshResolution);
	meshDeformer->data_type = DATA_TYPE::USE_PARTICLE;
	meshDeformer->setParticleData(inputParticle);
	meshDeformer->SetLenses(&lenses);
	physicalParticleDeformer = std::make_shared<PhysicalParticleDeformProcessor>(meshDeformer, inputParticle);
	physicalParticleDeformer->lenses = &lenses;
	if (openGL->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE){
		screenLensDisplaceProcessor->isActive = true;
		meshDeformer->isActive = false;
		physicalParticleDeformer->isActive = false;
	}
	else{
		screenLensDisplaceProcessor->isActive = false;
		meshDeformer->isActive = true;
		physicalParticleDeformer->isActive = true;
	}

	//order matters!
	openGL->AddProcessor("1screen", screenLensDisplaceProcessor.get());
	openGL->AddProcessor("2meshdeform", meshDeformer.get());
	openGL->AddProcessor("3physicalParticleDeform", physicalParticleDeformer.get());

	if (std::string(dataPath).find(".vtu") != std::string::npos){
		glyphRenderable = std::make_shared<SphereRenderable>(inputParticle);
		glyphRenderable->setColorMap(COLOR_MAP::RDYIGN, true);
	}
	else{
		//glyphRenderable = std::make_shared<CosmoRenderable>(inputParticle);
		glyphRenderable = std::make_shared<SphereRenderable>(inputParticle);
		//glyphRenderable->setColorMap(COLOR_MAP::SIMPLE_BLUE_RED);
		glyphRenderable->colorByFeature = true;
		glyphRenderable->setColorMap(COLOR_MAP::RAINBOW_COSMOLOGY);
	}
	meshRenderable = std::make_shared<MeshRenderable>(meshDeformer.get());
	lensRenderable = std::make_shared<LensRenderable>(&lenses);
	//glyphRenderable->SetVisibility(false);
	//lensRenderable->SetVisibility(false);

	openGL->AddRenderable("2glyph", glyphRenderable.get());
	openGL->AddRenderable("3lenses", lensRenderable.get());
	openGL->AddRenderable("4model", meshRenderable.get());




	rInteractor = std::make_shared<RegularInteractor>();
	rInteractor->setMatrixMgr(matrixMgr);
	
	lensInteractor = std::make_shared<LensInteractor>();
	lensInteractor->SetLenses(&lenses);
	openGL->AddInteractor("regular", rInteractor.get());
	openGL->AddInteractor("lens", lensInteractor.get());





	///********controls******/
	addLensBtn = new QPushButton("Add old lens");
	addLineLensBtn = new QPushButton("Add a Virtual Retractor");
	delLensBtn = std::make_shared<QPushButton>("Delete the Virtual Retractor");
	addCurveLensBtn = new QPushButton("Add curved band lens");
	saveStateBtn = std::make_shared<QPushButton>("Save State");
	loadStateBtn = std::make_shared<QPushButton>("Load State");
std::cout << posMin.x << " " << posMin.y << " " << posMin.z << std::endl;
std::cout << posMax.x << " " << posMax.y << " " << posMax.z << std::endl;
	QCheckBox* gridCheck = new QCheckBox("Show the Mesh", this);
	QCheckBox* cbBackFace = new QCheckBox("Show the Back Face", this);
	QCheckBox* udbeCheck = new QCheckBox("Use Density Based Stiffness");
	udbeCheck->setChecked(meshDeformer->elasticityMode >0);
	
	
	QCheckBox* cbChangeLensWhenRotateData = new QCheckBox("View Dependency", this); 
	cbChangeLensWhenRotateData->setChecked(lensInteractor->changeLensWhenRotateData);
	QCheckBox* cbDrawInsicionOnCenterFace = new QCheckBox("Draw the Incision at the Center Face", this);
	cbDrawInsicionOnCenterFace->setChecked(lensRenderable->drawInsicionOnCenterFace);

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

	arrowRenderable = std::make_shared<ArrowRenderable>(leapFingerIndicatorVecs,leapFingerIndicators);
	//arrowRenderable->SetVisibility(false);
	leapFingerIndicators->numParticles = 0; //use numParticles to control how many indicators are drawn on screen
	
	
	//openGL->AddRenderable("1LeapArrow", arrowRenderable.get()); //must be drawn first than glyphs
#endif

#ifdef USE_OSVR 
	vrWidget = std::make_shared<VRWidget>(matrixMgr, openGL.get());
	vrWidget->setWindowFlags(Qt::Window);
	vrGlyphRenderable = std::make_shared<VRGlyphRenderable>(glyphRenderable.get());
	
	lensRenderable2 = std::make_shared<LensRenderable>();;
	
	vrWidget->AddRenderable("6glyph", vrGlyphRenderable.get());
	vrWidget->AddRenderable("7lens", lensRenderable.get());

	openGL->SetVRWidget(vrWidget.get());

#endif


	QGroupBox *deformModeGroupBox = new QGroupBox(tr("Deformation Mode"));
	QHBoxLayout *deformModeLayout = new QHBoxLayout;
	radioDeformScreen = std::make_shared<QRadioButton>(tr("&screen space"));
	radioDeformObject = std::make_shared<QRadioButton>(tr("&object space"));
	radioDeformObject->setChecked(openGL->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE);
	radioDeformScreen->setChecked(openGL->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE);

	deformModeLayout->addWidget(radioDeformScreen.get());
	deformModeLayout->addWidget(radioDeformObject.get());
	deformModeGroupBox->setLayout(deformModeLayout);

	usingGlyphSnappingCheck = new QCheckBox("Snapping Glyph");
	usingGlyphPickingCheck = new QCheckBox("Picking Glyph");
	freezingFeatureCheck = new QCheckBox("Freezing Feature");
	usingFeatureSnappingCheck = new QCheckBox("Snapping Feature");
	usingFeaturePickingCheck = new QCheckBox("Picking Feature");

	connect(glyphRenderable.get(), SIGNAL(glyphPickingFinished()), this, SLOT(SlotToggleGlyphPickingFinished()));
	connect(glyphRenderable.get(), SIGNAL(featurePickingFinished()), this, SLOT(SlotToggleFeaturePickingFinished()));


	QHBoxLayout *meshResLayout = new QHBoxLayout;
	QLabel *meshResLitLabel = new QLabel(("Mesh Resolution:  "));
	QPushButton* addMeshResPushButton = new QPushButton(tr("&+"));
	addMeshResPushButton->setFixedSize(24, 24);
	QPushButton* minusMeshResPushButton = new QPushButton(tr("&-"));
	minusMeshResPushButton->setFixedSize(24, 24);
	meshResLabel = new QLabel(QString::number(meshDeformer->meshResolution));
	meshResLayout->addWidget(meshResLitLabel);
	meshResLayout->addWidget(minusMeshResPushButton);
	meshResLayout->addWidget(addMeshResPushButton);
	meshResLayout->addWidget(meshResLabel);
	meshResLayout->addStretch();

	QVBoxLayout *controlLayout = new QVBoxLayout;
	controlLayout->addWidget(addLensBtn);
	controlLayout->addWidget(addLineLensBtn);

	controlLayout->addWidget(addCurveLensBtn);
	controlLayout->addWidget(delLensBtn.get());
	controlLayout->addWidget(saveStateBtn.get());
	controlLayout->addWidget(loadStateBtn.get());
	controlLayout->addWidget(deformModeGroupBox);
	
	//controlLayout->addWidget(usingGlyphSnappingCheck);
	controlLayout->addWidget(usingGlyphPickingCheck);
	//controlLayout->addWidget(freezingFeatureCheck);
	//controlLayout->addWidget(usingFeatureSnappingCheck);
	//controlLayout->addWidget(usingFeaturePickingCheck); 
	controlLayout->addWidget(gridCheck); 
	controlLayout->addWidget(cbBackFace); 
	controlLayout->addWidget(cbChangeLensWhenRotateData);
	controlLayout->addWidget(cbDrawInsicionOnCenterFace);
	controlLayout->addLayout(meshResLayout);

	
	QLabel *deformForceLabelLit = new QLabel("Deform Force:");
	controlLayout->addWidget(deformForceLabelLit);
	deformForceSlider = new QSlider(Qt::Horizontal);
	deformForceSlider->setRange(0, 80);
	deformForceSlider->setValue(meshDeformer->getDeformForce() / deformForceConstant);
	connect(deformForceSlider, SIGNAL(valueChanged(int)), this, SLOT(deformForceSliderValueChanged(int)));
	deformForceLabel = new QLabel(QString::number(meshDeformer->getDeformForce()));
	QHBoxLayout *deformForceLayout = new QHBoxLayout;
	deformForceLayout->addWidget(deformForceSlider);
	deformForceLayout->addWidget(deformForceLabel);
	controlLayout->addLayout(deformForceLayout);
	//controlLayout->addWidget(udbeCheck);

	QGroupBox *gbStiffnessMode = new QGroupBox(tr("Stiffness Mode:"));
	QVBoxLayout *layoutStiffnessMode = new QVBoxLayout;
	QRadioButton* rbUniform = new QRadioButton(tr("&Uniform Stiffness"));
	QRadioButton* rbDensity = new QRadioButton(tr("&Density Based Stiffness"));
	QRadioButton* rbTransfer = new QRadioButton(tr("&Transfer Density Based Stiffness"));
	QRadioButton* rbGradient = new QRadioButton(tr("&Gradient Based Stiffness"));
	if (meshDeformer->elasticityMode == 0){
		rbUniform->setChecked(true);
	}
	else if (meshDeformer->elasticityMode == 1){
		rbDensity->setChecked(true);
	}
	else if (meshDeformer->elasticityMode == 2){
		rbTransfer->setChecked(true);
	}
	else if (meshDeformer->elasticityMode == 3){
		rbGradient->setChecked(true);
	}
	layoutStiffnessMode->addWidget(rbDensity);
	layoutStiffnessMode->addWidget(rbTransfer);
	layoutStiffnessMode->addWidget(rbGradient);
	layoutStiffnessMode->addWidget(rbUniform);
	gbStiffnessMode->setLayout(layoutStiffnessMode);
	controlLayout->addWidget(gbStiffnessMode);
	
	controlLayout->addStretch();


	connect(addLensBtn, SIGNAL(clicked()), this, SLOT(AddLens()));
	connect(addLineLensBtn, SIGNAL(clicked()), this, SLOT(AddLineLens()));
	connect(addCurveLensBtn, SIGNAL(clicked()), this, SLOT(AddCurveLens()));
	connect(delLensBtn.get(), SIGNAL(clicked()), this, SLOT(SlotDelLens()));
	connect(saveStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotSaveState()));
	connect(loadStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotLoadState()));

	connect(addMeshResPushButton, SIGNAL(clicked()), this, SLOT(SlotAddMeshRes()));
	connect(minusMeshResPushButton, SIGNAL(clicked()), this, SLOT(SlotMinusMeshRes()));
	
	connect(gridCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleGrid(bool)));
	connect(cbBackFace, SIGNAL(clicked(bool)), this, SLOT(SlotToggleBackFace(bool)));
	connect(udbeCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleUdbe(bool)));
	connect(cbChangeLensWhenRotateData, SIGNAL(clicked(bool)), this, SLOT(SlotToggleCbChangeLensWhenRotateData(bool)));
	connect(cbDrawInsicionOnCenterFace, SIGNAL(clicked(bool)), this, SLOT(SlotToggleCbDrawInsicionOnCenterFace(bool)));

#ifdef USE_LEAP
	connect(listener, SIGNAL(UpdateHands(QVector3D, QVector3D, int)),
		this, SLOT(SlotUpdateHands(QVector3D, QVector3D, int)));
#endif
#ifdef USE_NEW_LEAP
	//connect(listener, SIGNAL(UpdateHands(QVector3D, QVector3D, QVector3D, QVector3D, int)),
	//	this, SLOT(SlotUpdateHands(QVector3D, QVector3D, QVector3D, QVector3D, int)));
	connect(listener, SIGNAL(UpdateHandsNew(QVector3D, QVector3D, QVector3D, QVector3D, QVector3D, QVector3D, int)),
		this, SLOT(SlotUpdateHands(QVector3D, QVector3D, QVector3D, QVector3D, QVector3D, QVector3D, int)));

#endif
	connect(usingGlyphSnappingCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleUsingGlyphSnapping(bool)));
	connect(usingGlyphPickingCheck, SIGNAL(clicked(bool)), this, SLOT(SlotTogglePickingGlyph(bool)));
	connect(freezingFeatureCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleFreezingFeature(bool)));
	connect(usingFeatureSnappingCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleUsingFeatureSnapping(bool)));
	connect(usingFeaturePickingCheck, SIGNAL(clicked(bool)), this, SLOT(SlotTogglePickingFeature(bool)));
	connect(radioDeformObject.get(), SIGNAL(clicked(bool)), this, SLOT(SlotDeformModeChanged(bool)));
	connect(radioDeformScreen.get(), SIGNAL(clicked(bool)), this, SLOT(SlotDeformModeChanged(bool)));

	connect(rbUniform, SIGNAL(clicked(bool)), this, SLOT(SlotRbUniformChanged(bool)));
	connect(rbDensity, SIGNAL(clicked(bool)), this, SLOT(SlotRbDensityChanged(bool)));
	connect(rbTransfer, SIGNAL(clicked(bool)), this, SLOT(SlotRbTransferChanged(bool)));
	connect(rbGradient, SIGNAL(clicked(bool)), this, SLOT(SlotRbGradientChanged(bool)));
	
	mainLayout->addWidget(openGL.get(), 3);
	mainLayout->addLayout(controlLayout,1);
	setLayout(mainLayout);
}

void Window::AddLens()
{
	if (openGL->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE){
		meshDeformer->gridType = GRID_TYPE::UNIFORM_GRID;
		meshDeformer->InitializeUniformGrid(inputParticle); //call this function must set gridType = GRID_TYPE::UNIFORM_GRID first
		lensRenderable->AddCircleLens3D();
	}
	else if (openGL->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE){
		lensRenderable->AddCircleLens();
	}
}

void Window::AddLineLens()
{
	if (openGL->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE){
		meshDeformer->gridType = GRID_TYPE::LINESPLIT_UNIFORM_GRID;
		lensRenderable->AddLineLens3D();
	}
	else if (openGL->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE){
		lensRenderable->AddLineLens();
	}
}


void Window::AddCurveLens()
{
	lensRenderable->AddCurveLens();
}

void Window::SlotDelLens()
{
	lensRenderable->DelLens();
	inputParticle->reset();
	screenLensDisplaceProcessor->reset();
	openGL->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
}

void Window::SlotToggleUsingGlyphSnapping(bool b)
{
	lensInteractor->isSnapToGlyph = b;
	if (!b){
		inputParticle->SetSnappedGlyphId(-1);
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
	inputParticle->isPickingGlyph = b;
	if (b){
		usingFeatureSnappingCheck->setChecked(false);
		SlotToggleUsingFeatureSnapping(false);
		usingFeaturePickingCheck->setChecked(false);
		SlotTogglePickingFeature(false);
	}
}


void Window::SlotToggleFreezingFeature(bool b)
{
	inputParticle->isFreezingFeature = b;
	screenLensDisplaceProcessor->setRecomputeNeeded();
}

void Window::SlotToggleUsingFeatureSnapping(bool b)
{
	lensInteractor->isSnapToFeature = b;
	if (!b){
		inputParticle->SetSnappedFeatureId(-1);
		screenLensDisplaceProcessor->setRecomputeNeeded();
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
	inputParticle->isPickingFeature = b;
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
	meshRenderable->SetVisibility(b);
}

void Window::SlotToggleBackFace(bool b)
{
	lensRenderable->drawFullRetractor = b;
}


void Window::SlotToggleUdbe(bool b)
{
	if (b)
		meshDeformer->elasticityMode = 1;
	else
		meshDeformer->elasticityMode = 0;

	meshDeformer->setReinitiationNeed();
	
	//meshDeformer->SetElasticityForParticle(inputParticle);
	//meshDeformer->UpdateMeshDevElasticity();
	//comparing to reinitiate the whole mesh, this does not work well
}

void Window::SlotToggleCbChangeLensWhenRotateData(bool b)
{
	if (b)
		lensInteractor->changeLensWhenRotateData = true;
	else
		lensInteractor->changeLensWhenRotateData = false;
}

void Window::SlotToggleCbDrawInsicionOnCenterFace(bool b)
{
	if (b)
		lensRenderable->drawInsicionOnCenterFace = true;
	else
		lensRenderable->drawInsicionOnCenterFace = false;
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
void Window::SlotUpdateHands(QVector3D rightThumbTip, QVector3D rightIndexTip, QVector3D leftThumbTip, QVector3D leftIndexTip, QVector3D rightMiddleTip, QVector3D rightRingTip, int numHands)
{
	if (1 == numHands){
		float4 markerPos;
		float f = meshDeformer->getDeformForce();
		if (lensRenderable->SlotOneHandChangedNew_lc(
			make_float3(rightThumbTip.x(), rightThumbTip.y(), rightThumbTip.z()), 
			make_float3(rightIndexTip.x(), rightIndexTip.y(), rightIndexTip.z()), 
			make_float3(rightMiddleTip.x(), rightMiddleTip.y(), rightMiddleTip.z()),
			make_float3(rightRingTip.x(), rightRingTip.y(), rightRingTip.z()),
			markerPos, leapFingerIndicators->val[0], f)){

			openGL->blendOthers = true;
			deformForceSlider->setValue(f / deformForceConstant);

		}
		else{
			openGL->blendOthers = false;
		}
		leapFingerIndicators->numParticles = 1;
		leapFingerIndicators->pos[0] = markerPos;

		lensRenderable->activedCursors = 1;
		lensRenderable->cursorPos[0] = make_float3(markerPos);
		lensRenderable->cursorColor[0] = leapFingerIndicators->val[0];
	}

}
#endif

void Window::SlotSaveState()
{
	matrixMgr->SaveState("current.state");
	lensRenderable->SaveState("lens.state");

	std::ofstream myfile;
	myfile.open("system.state");

	myfile << meshDeformer->getDeformForce() << std::endl;
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

		meshResolution = res;
		meshDeformer->meshResolution = meshResolution;
		meshDeformer->setReinitiationNeed();
		meshResLabel->setText(QString::number(meshResolution));
		deformForceSlider->setValue(f / deformForceConstant); //will also call the connected slot
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
		
		screenLensDisplaceProcessor->isActive = true;
		meshDeformer->isActive = false;
		physicalParticleDeformer->isActive = false;
	}
	else if (radioDeformObject->isChecked()){
		openGL->SetDeformModel(DEFORM_MODEL::OBJECT_SPACE);

		screenLensDisplaceProcessor->isActive = false;
		meshDeformer->isActive = true;
		physicalParticleDeformer->isActive = true;
	}
}

void Window::deformForceSliderValueChanged(int v)
{
	float newForce = deformForceConstant*v;
	deformForceLabel->setText(QString::number(newForce));
	meshDeformer->setDeformForce(newForce);

}

void Window::SlotAddMeshRes()
{
	meshResolution++;
	meshDeformer->meshResolution = meshResolution;
	meshDeformer->setReinitiationNeed();
	meshResLabel->setText(QString::number(meshResolution));
}
void Window::SlotMinusMeshRes()
{
	meshResolution--;
	meshDeformer->meshResolution = meshResolution;
	meshDeformer->setReinitiationNeed();
	meshResLabel->setText(QString::number(meshResolution));
}


void Window::SlotRbUniformChanged(bool b)
{
	if (b){
		meshDeformer->elasticityMode = 0;
		meshDeformer->setReinitiationNeed();
		inputParticle->reset();
	}
}
void Window::SlotRbDensityChanged(bool b)
{
	if (b){
		meshDeformer->elasticityMode = 1;
		meshDeformer->setReinitiationNeed();
		inputParticle->reset();
	}
}
void Window::SlotRbTransferChanged(bool b)
{
	if (b){
		meshDeformer->elasticityMode = 2;
		meshDeformer->setReinitiationNeed();
		inputParticle->reset();
	}
}
void Window::SlotRbGradientChanged(bool b)
{
	if (b){
		meshDeformer->elasticityMode = 3;
		meshDeformer->setReinitiationNeed();
		inputParticle->reset();
	}
}
