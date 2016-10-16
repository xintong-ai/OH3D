#include "window.h"
#include "DeformGLWidget.h"
#include "LensRenderable.h"
#include <iostream>

#include "RawVolumeReader.h"
#include "Volume.h"

#include "DataMgr.h"
#include "ModelGridRenderable.h"
#include "LineSplitModelGrid.h"
#include "GLMatrixManager.h"
#include "VecReader.h"
#include "VolumeRenderableCUDA.h"
#include "ModelVolumeDeformer.h"

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
#include "VRVolumeRenderableCUDA.h"
#endif


Window::Window()
{
    setWindowTitle(tr("Interactive Glyph Visualization"));
	QHBoxLayout *mainLayout = new QHBoxLayout;

	dataMgr = std::make_shared<DataMgr>();
	

	const std::string dataPath = dataMgr->GetConfig("VOLUME_DATA_PATH");

	int3 dims;
	float3 spacing;
	if (std::string(dataPath).find("MGHT2") != std::string::npos){
		dims = make_int3(320, 320, 256);
		spacing = make_float3(0.7, 0.7, 0.7);
	}
	else if (std::string(dataPath).find("MGHT1") != std::string::npos){
		dims = make_int3(256, 256, 176);
		spacing = make_float3(1.0, 1.0, 1.0);
	}
	else if (std::string(dataPath).find("nek128") != std::string::npos){
		dims = make_int3(128, 128, 128);
		spacing = make_float3(2, 2, 2); //to fit the streamline of nek256
	}
	else if (std::string(dataPath).find("nek256") != std::string::npos){
		dims = make_int3(256, 256, 256);
		spacing = make_float3(1, 1, 1);
	}
	else if (std::string(dataPath).find("cthead") != std::string::npos){
		dims = make_int3(208, 256, 225);
		spacing = make_float3(1, 1, 1);
	}
	else{
		std::cout << "volume data name not recognized" << std::endl;
		exit(0);
	}
	inputVolume = std::make_shared<Volume>();

	if (std::string(dataPath).find(".vec") != std::string::npos){
		std::shared_ptr<VecReader> reader2;
		reader2 = std::make_shared<VecReader>(dataPath.c_str());
		reader2->OutputToVolumeByNormalizedVecMag(inputVolume);
		//reader2->OutputToVolumeByNormalizedVecDownSample(inputVolume,2);
		//reader2->OutputToVolumeByNormalizedVecUpSample(inputVolume, 2);
		//reader2->OutputToVolumeByNormalizedVecMagWithPadding(inputVolume,10);
		
		reader2.reset();
	}
	else{
		std::shared_ptr<RawVolumeReader> reader2;
		reader2 = std::make_shared<RawVolumeReader>(dataPath.c_str(), dims);
		reader2->OutputToVolumeByNormalizedValue(inputVolume);
		reader2.reset();
	}
	inputVolume->spacing = spacing;
	inputVolume->initVolumeCuda(0);

	volumeRenderable = std::make_shared<VolumeRenderableCUDA>(inputVolume);

	if (std::string(dataPath).find("nek") != std::string::npos){
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
		vrVolumeRenderable = std::make_shared<VRVolumeRenderableCUDA>(volumeRenderable.get());
		vrWidget->AddRenderable("glyph", vrVolumeRenderable.get());
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
	matrixMgr->SetVol(posMin, posMax);// cubemap->GetInnerDim());
	modelGrid = std::make_shared<LineSplitModelGrid>(&posMin.x, &posMax.x, meshResolution);
	modelGrid->SetLenses(lensRenderable->GetLensesAddr());
	modelGridRenderable = std::make_shared<ModelGridRenderable>(modelGrid.get());
	modelGridRenderable->SetLenses(lensRenderable->GetLensesAddr());

	modelVolumeDeformer = std::make_shared<ModelVolumeDeformer>();
	modelVolumeDeformer->SetModelGrid(modelGrid);
	modelVolumeDeformer->Init(inputVolume.get());
	volumeRenderable->SetModelVolumeDeformer(modelVolumeDeformer);
	volumeRenderable->SetModelGrid(modelGrid);

	openGL->AddRenderable("lenses", lensRenderable.get());
	openGL->AddRenderable("1volume", volumeRenderable.get()); //make sure the volume is rendered first since it does not use depth test

	openGL->AddRenderable("model", modelGridRenderable.get());
	///********controls******/
	addLensBtn = new QPushButton("Add circle lens");
	addLineLensBtn = new QPushButton("Add a Virtual Retractor");
	delLensBtn = std::make_shared<QPushButton>("Delete the Virtual Retractor");
	//addCurveBLensBtn = new QPushButton("Add curved band lens");
	saveStateBtn = std::make_shared<QPushButton>("Save State");
	loadStateBtn = std::make_shared<QPushButton>("Load State");
std::cout << posMin.x << " " << posMin.y << " " << posMin.z << std::endl;
std::cout << posMax.x << " " << posMax.y << " " << posMax.z << std::endl;

	QCheckBox* gridCheck = new QCheckBox("Show Back Face and Mesh", this);
	QCheckBox* cbBackFace = new QCheckBox("Show the Back Face", this);
	QCheckBox* cbChangeLensWhenRotateData = new QCheckBox("View Dependency", this);
	cbChangeLensWhenRotateData->setChecked(lensRenderable->changeLensWhenRotateData);
	QCheckBox* cbDrawInsicionOnCenterFace = new QCheckBox("Draw the Incision at the Center Face", this);
	cbDrawInsicionOnCenterFace->setChecked(lensRenderable->drawInsicionOnCenterFace);

	modelGrid->setDeformForce(2000);
	QLabel *deformForceLabelLit = new QLabel("Deform Force:");
	//controlLayout->addWidget(deformForceLabelLit);
	QSlider *deformForceSlider = new QSlider(Qt::Horizontal);
	deformForceSlider->setRange(0, 50);
	deformForceSlider->setValue(modelGrid->getDeformForce() / deformForceConstant);
	connect(deformForceSlider, SIGNAL(valueChanged(int)), this, SLOT(deformForceSliderValueChanged(int)));
	deformForceLabel = new QLabel(QString::number(modelGrid->getDeformForce()));
	QHBoxLayout *deformForceLayout = new QHBoxLayout;
	deformForceLayout->addWidget(deformForceSlider);
	deformForceLayout->addWidget(deformForceLabel);


	QGroupBox *gbStiffnessMode = new QGroupBox(tr("Stiffness Mode:"));
	QVBoxLayout *layoutStiffnessMode = new QVBoxLayout;
	QRadioButton* rbUniform = new QRadioButton(tr("&Uniform Stiffness"));
	QRadioButton* rbDensity = new QRadioButton(tr("&Density Based Stiffness"));
	QRadioButton* rbTransfer = new QRadioButton(tr("&Transfer Density Based Stiffness"));
	QRadioButton* rbGradient = new QRadioButton(tr("&Gradient Based Stiffness"));
	if (modelGrid->elasticityMode == 0){
		rbUniform->setChecked(true);
	}
	else if (modelGrid->elasticityMode == 1){
		rbDensity->setChecked(true);
	}
	else if (modelGrid->elasticityMode == 2){
		rbTransfer->setChecked(true);
	}
	else if (modelGrid->elasticityMode == 3){
		rbGradient->setChecked(true);
	}

	layoutStiffnessMode->addWidget(rbDensity);
	layoutStiffnessMode->addWidget(rbTransfer);
	layoutStiffnessMode->addWidget(rbGradient);
	layoutStiffnessMode->addWidget(rbUniform);
	gbStiffnessMode->setLayout(layoutStiffnessMode);

	QHBoxLayout *meshResLayout = new QHBoxLayout;
	QLabel *meshResLitLabel = new QLabel(("Mesh Resolution:  "));
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

	QVBoxLayout *controlLayout = new QVBoxLayout;
	//controlLayout->addWidget(addLensBtn);
	controlLayout->addWidget(addLineLensBtn);

	//controlLayout->addWidget(addCurveBLensBtn);
	controlLayout->addWidget(delLensBtn.get());
	controlLayout->addWidget(saveStateBtn.get());
	controlLayout->addWidget(loadStateBtn.get());
	//controlLayout->addWidget(groupBox);
	controlLayout->addWidget(gridCheck);
	controlLayout->addWidget(cbBackFace);
	controlLayout->addWidget(cbChangeLensWhenRotateData);
	controlLayout->addWidget(cbDrawInsicionOnCenterFace); 
	controlLayout->addLayout(meshResLayout);
	controlLayout->addWidget(deformForceLabelLit);
	controlLayout->addLayout(deformForceLayout);
	controlLayout->addWidget(gbStiffnessMode);


	


	QLabel *transFuncP1SliderLabelLit = new QLabel("Transfer Function Higher Cut Off");
	//controlLayout->addWidget(transFuncP1SliderLabelLit);
	QSlider *transFuncP1LabelSlider = new QSlider(Qt::Horizontal);
	transFuncP1LabelSlider->setRange(0, 100);
	transFuncP1LabelSlider->setValue(volumeRenderable->transFuncP1 * 100);
	connect(transFuncP1LabelSlider, SIGNAL(valueChanged(int)), this, SLOT(transFuncP1LabelSliderValueChanged(int)));
	transFuncP1Label = new QLabel(QString::number(volumeRenderable->transFuncP1));
	QHBoxLayout *transFuncP1Layout = new QHBoxLayout;
	transFuncP1Layout->addWidget(transFuncP1LabelSlider);
	transFuncP1Layout->addWidget(transFuncP1Label);
	//controlLayout->addLayout(transFuncP1Layout);

	QLabel *transFuncP2SliderLabelLit = new QLabel("Transfer Function Lower Cut Off");
	//controlLayout->addWidget(transFuncP2SliderLabelLit);
	QSlider *transFuncP2LabelSlider = new QSlider(Qt::Horizontal);
	transFuncP2LabelSlider->setRange(0, 100);
	transFuncP2LabelSlider->setValue(volumeRenderable->transFuncP2 * 100);
	connect(transFuncP2LabelSlider, SIGNAL(valueChanged(int)), this, SLOT(transFuncP2LabelSliderValueChanged(int)));
	transFuncP2Label = new QLabel(QString::number(volumeRenderable->transFuncP2));
	QHBoxLayout *transFuncP2Layout = new QHBoxLayout;
	transFuncP2Layout->addWidget(transFuncP2LabelSlider);
	transFuncP2Layout->addWidget(transFuncP2Label);
	//controlLayout->addLayout(transFuncP2Layout);

	QLabel *brLabelLit = new QLabel("Brightness of the volume: ");
	//controlLayout->addWidget(brLabelLit);
	QSlider* brSlider = new QSlider(Qt::Horizontal);
	brSlider->setRange(0, 40);
	brSlider->setValue(volumeRenderable->brightness * 20);
	connect(brSlider, SIGNAL(valueChanged(int)), this, SLOT(brSliderValueChanged(int)));
	brLabel = new QLabel(QString::number(volumeRenderable->brightness));
	QHBoxLayout *brLayout = new QHBoxLayout;
	brLayout->addWidget(brSlider);
	brLayout->addWidget(brLabel);
	//controlLayout->addLayout(brLayout);

	QLabel *dsLabelLit = new QLabel("Density of the volume: ");
	//controlLayout->addWidget(dsLabelLit);
	QSlider* dsSlider = new QSlider(Qt::Horizontal);
	dsSlider->setRange(0, 40);
	dsSlider->setValue(volumeRenderable->density * 20);
	connect(dsSlider, SIGNAL(valueChanged(int)), this, SLOT(dsSliderValueChanged(int)));
	dsLabel = new QLabel(QString::number(volumeRenderable->density));
	QHBoxLayout *dsLayout = new QHBoxLayout;
	dsLayout->addWidget(dsSlider);
	dsLayout->addWidget(dsLabel);
	//controlLayout->addLayout(dsLayout);


	QLabel *laSliderLabelLit = new QLabel("Coefficient for Ambient Lighting: ");
	//controlLayout->addWidget(laSliderLabelLit);
	QSlider* laSlider = new QSlider(Qt::Horizontal);
	laSlider->setRange(0, 50);
	laSlider->setValue(volumeRenderable->la * 10);
	connect(laSlider, SIGNAL(valueChanged(int)), this, SLOT(laSliderValueChanged(int)));
	laLabel = new QLabel(QString::number(volumeRenderable->la));
	QHBoxLayout *laLayout = new QHBoxLayout;
	laLayout->addWidget(laSlider);
	laLayout->addWidget(laLabel);
	//controlLayout->addLayout(laLayout);

	QLabel *ldSliderLabelLit = new QLabel("Coefficient for Diffusial Lighting: ");
	//controlLayout->addWidget(ldSliderLabelLit);
	QSlider* ldSlider = new QSlider(Qt::Horizontal);
	ldSlider->setRange(0, 50);
	ldSlider->setValue(volumeRenderable->ld * 10);
	connect(ldSlider, SIGNAL(valueChanged(int)), this, SLOT(ldSliderValueChanged(int)));
	ldLabel = new QLabel(QString::number(volumeRenderable->ld));
	QHBoxLayout *ldLayout = new QHBoxLayout;
	ldLayout->addWidget(ldSlider);
	ldLayout->addWidget(ldLabel);
	//controlLayout->addLayout(ldLayout);

	QLabel *lsSliderLabelLit = new QLabel("Coefficient for Specular Lighting: ");
	//controlLayout->addWidget(lsSliderLabelLit);
	QSlider* lsSlider = new QSlider(Qt::Horizontal);
	lsSlider->setRange(0, 50);
	lsSlider->setValue(volumeRenderable->ls * 10);
	connect(lsSlider, SIGNAL(valueChanged(int)), this, SLOT(lsSliderValueChanged(int)));
	lsLabel = new QLabel(QString::number(volumeRenderable->ls));
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

	//controlLayout->addWidget(rcGroupBox);


	controlLayout->addStretch();

	connect(addLensBtn, SIGNAL(clicked()), this, SLOT(AddLens()));
	connect(addLineLensBtn, SIGNAL(clicked()), this, SLOT(AddLineLens()));
	//connect(addCurveBLensBtn, SIGNAL(clicked()), this, SLOT(AddCurveBLens()));
	connect(delLensBtn.get(), SIGNAL(clicked()), this, SLOT(SlotDelLens()));
	connect(saveStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotSaveState()));
	connect(loadStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotLoadState()));
	connect(addMeshResPushButton, SIGNAL(clicked()), this, SLOT(SlotAddMeshRes()));
	connect(minusMeshResPushButton, SIGNAL(clicked()), this, SLOT(SlotMinusMeshRes()));
	
	connect(gridCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleGrid(bool)));
	connect(cbBackFace, SIGNAL(clicked(bool)), this, SLOT(SlotToggleBackFace(bool)));
	connect(cbDrawInsicionOnCenterFace, SIGNAL(clicked(bool)), this, SLOT(SlotToggleCbDrawInsicionOnCenterFace(bool)));

#ifdef USE_LEAP
	connect(listener, SIGNAL(UpdateHands(QVector3D, QVector3D, int)),
		this, SLOT(SlotUpdateHands(QVector3D, QVector3D, int)));
#endif
#ifdef USE_NEW_LEAP
	connect(listener, SIGNAL(UpdateHands(QVector3D, QVector3D, int)),
		this, SLOT(SlotUpdateHands(QVector3D, QVector3D, int)));
#endif


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


void Window::SlotToggleGrid(bool b)
{
	modelGridRenderable->SetVisibility(b);
}

void Window::SlotToggleBackFace(bool b)
{
	lensRenderable->drawFullRetractor = b;
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

void Window::SlotRbUniformChanged(bool b)
{
	if (b){
		modelGrid->elasticityMode = 0;
		modelGrid->setReinitiationNeed();
	}
}
void Window::SlotRbDensityChanged(bool b)
{
	if (b){
		modelGrid->elasticityMode = 1;
		modelGrid->setReinitiationNeed();
	}
}
void Window::SlotRbTransferChanged(bool b)
{
	if (b){
		modelGrid->elasticityMode = 2;
		modelGrid->setReinitiationNeed();
	}
}
void Window::SlotRbGradientChanged(bool b)
{
	if (b){
		modelGrid->elasticityMode = 3;
		modelGrid->setReinitiationNeed();
	}
}


void Window::SlotDelLens()
{
	lensRenderable->SlotDelLens();
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

void Window::SlotToggleCbDrawInsicionOnCenterFace(bool b)
{
	if (b)
		lensRenderable->drawInsicionOnCenterFace = true;
	else
		lensRenderable->drawInsicionOnCenterFace = false;
}