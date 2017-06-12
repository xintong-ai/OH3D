#include "window.h"
#include "DeformGLWidget.h"
//#include "VecReader.h"
#include "BoxRenderable.h"
#include "LensRenderable.h"
#include "ScreenLensDisplaceProcessor.h"
#include "Lens.h"
#include "PolyMesh.h"

#include "DTIVolumeReader.h"
#include "SQRenderable.h"
#include "MeshDeformProcessor.h"
#include "ScreenLensDisplaceProcessor.h"
#include "PhysicalParticleDeformProcessor.h"

#include "mouse/RegularInteractor.h"
#include "mouse/LensInteractor.h"

#include "DataMgr.h"

#include "GLMatrixManager.h"
#include "PolyRenderable.h"
#include "PlyMeshReader.h"
#include "Particle.h"
#include "MeshRenderable.h"

#ifdef USE_LEAP
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
	
	std::shared_ptr<DTIVolumeReader> reader;
	inputParticle = std::make_shared<Particle>();
	const std::string dataPath = dataMgr->GetConfig("DATA_PATH_TENSOR");
	reader = std::make_shared<DTIVolumeReader>(dataPath.c_str());
	std::vector<float4> pos;
	std::vector<float> val;

	if (false){//existing in old prog. forget what it is used for...
	//if (((DTIVolumeReader*)reader.get())->LoadFeatureNew(dataMgr->GetConfig("FEATURE_PATH").c_str())){
		/*std::vector<char> feature;
		((DTIVolumeReader*)reader.get())->GetSamplesWithFeature(pos, val, feature);
		glyphRenderable = std::make_shared<SQRenderable>(pos, val);
		glyphRenderable->SetFeature(feature, ((DTIVolumeReader*)reader.get())->featureCenter);*/
	}
	else
	{
		((DTIVolumeReader*)reader.get())->OutputToParticleData(inputParticle);
		glyphRenderable = std::make_shared<SQRenderable>(inputParticle);
	}

	std::cout << "number of rendered glyphs: " << inputParticle->pos.size() << std::endl;

	/********GL widget******/
	float3 posMin = inputParticle->posMin, posMax = inputParticle->posMax;
#ifdef USE_OSVR
	matrixMgr = std::make_shared<GLMatrixManager>(true);
#else
	matrixMgr = std::make_shared<GLMatrixManager>(posMin, posMax);
#endif
	openGL = std::make_shared<DeformGLWidget>(matrixMgr);
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
	
	PlyMeshReader *meshReader;
	meshReader = new PlyMeshReader();
	polyMeshFeature0 = std::make_shared<PolyMesh>();
	meshReader->LoadPLY(dataMgr->GetConfig("ventricles").c_str(), polyMeshFeature0);
	polyFeature0 = new PolyRenderable(polyMeshFeature0);
	polyFeature0->SetAmbientColor(0.2, 0, 0);
	openGL->AddRenderable("ventricles", polyFeature0);

	meshReader = new PlyMeshReader();
	polyMeshFeature1 = std::make_shared<PolyMesh>();
	meshReader->LoadPLY(dataMgr->GetConfig("tumor1").c_str(), polyMeshFeature1);
	polyFeature1 = new PolyRenderable(polyMeshFeature1);
	polyFeature1->SetAmbientColor(0.0, 0.2, 0);
	openGL->AddRenderable("tumor1", polyFeature1);
	
	meshReader = new PlyMeshReader();
	polyMeshFeature2 = std::make_shared<PolyMesh>();
	meshReader->LoadPLY(dataMgr->GetConfig("tumor2").c_str(), polyMeshFeature2);
	polyFeature2 = new PolyRenderable(polyMeshFeature2);
	polyFeature2->SetAmbientColor(0.0, 0.0, 0.2);
	openGL->AddRenderable("tumor2", polyFeature2);
	
	featuresLw = new QListWidget();
	featuresLw->addItem(QString("ventricles"));
	featuresLw->addItem(QString("tumor1"));
	featuresLw->addItem(QString("tumor2"));
	featuresLw->setEnabled(false);



	///////////////////////deformer
	screenLensDisplaceProcessor = std::make_shared<ScreenLensDisplaceProcessor>(&lenses, inputParticle);
	meshDeformer = std::make_shared<MeshDeformProcessor>(&posMin.x, &posMax.x, meshResolution);

	std::cout << posMin.x << " " << posMin.y << " " << posMin.z << "  "
		<< posMax.x << " " << posMax.y << " " << posMax.z << "  " << std::endl;

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

	meshDeformer->setDeformForce(meshDeformer->getDeformForce()*100);

	/////////////////////renderable
	lensRenderable = std::make_shared<LensRenderable>(&lenses);

	meshRenderable = std::make_shared<MeshRenderable>(meshDeformer.get());

	openGL->AddRenderable("glyph", glyphRenderable.get());
	openGL->AddRenderable("lenses", lensRenderable.get());
	openGL->AddRenderable("4model", meshRenderable.get());

	/////////////////////interactor
	rInteractor = std::make_shared<RegularInteractor>();
	rInteractor->setMatrixMgr(matrixMgr);

	lensInteractor = std::make_shared<LensInteractor>();
	lensInteractor->SetLenses(&lenses);
	openGL->AddInteractor("regular", rInteractor.get());
	openGL->AddInteractor("lens", lensInteractor.get());


	/////////////////////processor

	openGL->AddProcessor("1screen", screenLensDisplaceProcessor.get());
	openGL->AddProcessor("2meshdeform", meshDeformer.get());
	openGL->AddProcessor("3physicalParticleDeform", physicalParticleDeformer.get());


	///********controls******/
	addLensBtn = new QPushButton("Add circle lens");
	addLineLensBtn = new QPushButton("Add straight band lens");
	delLensBtn = std::make_shared<QPushButton>("Delete a lens");
	addCurveLensBtn = new QPushButton("Add curved band lens");
	saveStateBtn = std::make_shared<QPushButton>("Save State");
	loadStateBtn = std::make_shared<QPushButton>("Load State");

	QCheckBox* gridCheck = new QCheckBox("Grid", this);
#ifdef USE_LEAP
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
	freezingFeatureCheck = new QCheckBox("Freezing Feature", this);
	usingFeatureSnappingCheck = new QCheckBox("Snapping Feature", this);
	usingFeaturePickingCheck = new QCheckBox("Picking Feature", this);

	connect(glyphRenderable.get(), SIGNAL(glyphPickingFinished()), this, SLOT(SlotToggleGlyphPickingFinished()));
	connect(glyphRenderable.get(), SIGNAL(featurePickingFinished()), this, SLOT(SlotToggleFeaturePickingFinished()));


	QVBoxLayout *controlLayout = new QVBoxLayout;
	controlLayout->addWidget(addLensBtn);
	controlLayout->addWidget(addLineLensBtn);

	controlLayout->addWidget(addCurveLensBtn);
	controlLayout->addWidget(delLensBtn.get());
	controlLayout->addWidget(saveStateBtn.get());
	controlLayout->addWidget(loadStateBtn.get());
	controlLayout->addWidget(groupBox);
	controlLayout->addWidget(usingGlyphSnappingCheck);
	controlLayout->addWidget(usingGlyphPickingCheck);
	controlLayout->addWidget(freezingFeatureCheck);
	controlLayout->addWidget(usingFeatureSnappingCheck); 
	controlLayout->addWidget(usingFeaturePickingCheck);
	controlLayout->addWidget(gridCheck);
	controlLayout->addStretch();

	connect(addLensBtn, SIGNAL(clicked()), this, SLOT(AddLens()));
	connect(addLineLensBtn, SIGNAL(clicked()), this, SLOT(AddLineLens()));
	connect(addCurveLensBtn, SIGNAL(clicked()), this, SLOT(AddCurveLens()));
	connect(delLensBtn.get(), SIGNAL(clicked()), this, SLOT(SlotDelLens()));
	connect(saveStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotSaveState()));
	connect(loadStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotLoadState()));


	connect(gridCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleGrid(bool)));
	//connect(listener, SIGNAL(UpdateRightHand(QVector3D, QVector3D, QVector3D)),
	//	this, SLOT(UpdateRightHand(QVector3D, QVector3D, QVector3D)));
#ifdef USE_LEAP
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

	if (featuresLw != NULL) {
		controlLayout->addWidget(featuresLw);
		connect(featuresLw, SIGNAL(currentRowChanged(int)), this, SLOT(SlotFeaturesLwRowChanged(int)));
	}
	
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


void Window::SlotToggleGrid(bool b)
{
	meshRenderable->SetVisibility(b);
}

Window::~Window() {
}

void Window::init()
{
#ifdef USE_OSVR
		vrWidget->show();
#endif
}


void Window::SlotToggleUsingGlyphSnapping(bool b)
{
	//lensRenderable->isSnapToGlyph = b;
	//if (!b){
	//	glyphRenderable->SetSnappedGlyphId(-1);
	//}
	//else{
	//	usingFeatureSnappingCheck->setChecked(false);
	//	SlotToggleUsingFeatureSnapping(false);
	//	usingFeaturePickingCheck->setChecked(false);
	//	SlotTogglePickingFeature(false);
	//}
}

void Window::SlotTogglePickingGlyph(bool b)
{
	//glyphRenderable->isPickingGlyph = b;
	//if (b){
	//	usingFeatureSnappingCheck->setChecked(false);
	//	SlotToggleUsingFeatureSnapping(false);
	//	usingFeaturePickingCheck->setChecked(false);
	//	SlotTogglePickingFeature(false);
	//}
}

void Window::SlotToggleFreezingFeature(bool b)
{
	//glyphRenderable->isFreezingFeature = b;
	//glyphRenderable->RecomputeTarget();
}

void Window::SlotToggleUsingFeatureSnapping(bool b)
{
	//lensRenderable->isSnapToFeature = b;
	//if (!b){
	//	glyphRenderable->SetSnappedFeatureId(-1);
	//	glyphRenderable->RecomputeTarget();
	//}
	//else{
	//	usingGlyphSnappingCheck->setChecked(false);
	//	SlotToggleUsingGlyphSnapping(false);
	//	usingGlyphPickingCheck->setChecked(false);
	//	SlotTogglePickingGlyph(false);
	//}
}

void Window::SlotTogglePickingFeature(bool b)
{
	//glyphRenderable->isPickingFeature = b;
	//if (b){
	//	usingGlyphSnappingCheck->setChecked(false);
	//	SlotToggleUsingGlyphSnapping(false);
	//	usingGlyphPickingCheck->setChecked(false);
	//	SlotTogglePickingGlyph(false);
	//	featuresLw->setEnabled(true);
	//}
	//else{
	//	featuresLw->clearSelection();
	//	featuresLw->setEnabled(false);
	//}
}

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

		screenLensDisplaceProcessor->isActive = true;
		meshDeformer->isActive = false;
		physicalParticleDeformer->isActive = false;

		addCurveLensBtn->setDisabled(false);
	}
	else if (radioDeformObject->isChecked()){
		openGL->SetDeformModel(DEFORM_MODEL::OBJECT_SPACE);

		screenLensDisplaceProcessor->isActive = false;
		meshDeformer->isActive = true;
		physicalParticleDeformer->isActive = true;

		addCurveLensBtn->setDisabled(true);
	}
}

void Window::SlotFeaturesLwRowChanged(int currentRow)
{
	//if (currentRow == 0){
	//	polyFeature0->isSnapped = true;
	//	polyFeature1->isSnapped = false;
	//	polyFeature2->isSnapped = false;
	//	glyphRenderable->SetSnappedFeatureId(1);
	//	lensRenderable->snapPos = glyphRenderable->featureCenter[currentRow];// polyFeature0->GetPolyCenter();
	//	lensRenderable->SnapLastLens();
	//}
	//else if (currentRow == 1){
	//	polyFeature0->isSnapped = false;
	//	polyFeature1->isSnapped = true;
	//	polyFeature2->isSnapped = false;
	//	glyphRenderable->SetSnappedFeatureId(2);
	//	lensRenderable->snapPos = glyphRenderable->featureCenter[currentRow]; //lensRenderable->snapPos = polyFeature1->GetPolyCenter();
	//	lensRenderable->SnapLastLens();
	//}
	//else if (currentRow == 2){
	//	polyFeature0->isSnapped = false;
	//	polyFeature1->isSnapped = false;
	//	polyFeature2->isSnapped = true;
	//	glyphRenderable->SetSnappedFeatureId(3); 
	//	lensRenderable->snapPos = glyphRenderable->featureCenter[currentRow]; //lensRenderable->snapPos = polyFeature2->GetPolyCenter();
	//	lensRenderable->SnapLastLens();
	//}
	//glyphRenderable->RecomputeTarget();
	//featuresLw->setCurrentRow(-1);
	//usingFeaturePickingCheck->setChecked(false);
	//featuresLw->setEnabled(false);
}


void Window::SlotDelLens()
{
	lensRenderable->DelLens();
	inputParticle->reset();
	screenLensDisplaceProcessor->reset();
	openGL->SetInteractMode(INTERACT_MODE::OPERATE_MATRIX);
}
