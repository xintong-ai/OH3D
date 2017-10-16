#include "window.h"
#include "DeformGLWidget.h"
#include "LensRenderable.h"
#include <iostream>

#include "RawVolumeReader.h"
#include "Volume.h"

#include "DataMgr.h"
#include "MeshRenderable.h"
#include "MeshDeformProcessor.h"
#include "GLMatrixManager.h"
#include "VecReader.h"
#include "VolumeRenderableCUDA.h"
#include "PhysicalVolumeDeformProcessor.h"
#include "mouse/RegularInteractor.h"
#include "mouse/LensInteractor.h"

#ifdef USE_OSVR
#include "VRWidget.h"
#include "VRVolumeRenderableCUDA.h"
#endif

#include"Lens.h"


Window::Window()
{
    setWindowTitle(tr("Interactive Glyph Visualization"));
	QHBoxLayout *mainLayout = new QHBoxLayout;

	dataMgr = std::make_shared<DataMgr>();
	const std::string dataPath = dataMgr->GetConfig("VOLUME_DATA_PATH");

	int3 dims;
	float3 spacing;

	std::shared_ptr<RayCastingParameters> rcp = std::make_shared<RayCastingParameters>();
	std::string temp;
	Volume::rawFileInfo(dataPath, dims, spacing, rcp, temp);
	inputVolume = std::make_shared<Volume>(true);

	if (std::string(dataPath).find("synthetic") != std::string::npos){
		inputVolume->createSyntheticData();
		dims = inputVolume->size;
		spacing = inputVolume->spacing;
	}
	else if (std::string(dataPath).find(".vec") != std::string::npos){
		std::shared_ptr<VecReader> reader;
		reader = std::make_shared<VecReader>(dataPath.c_str());
		//reader->OutputToVolumeByNormalizedVecMag(inputVolume);
		//reader->OutputToVolumeByNormalizedVecDownSample(inputVolume,2);
		//reader->OutputToVolumeByNormalizedVecUpSample(inputVolume, 2);
		reader->OutputToVolumeByNormalizedVecMagWithPadding(inputVolume,10);
		
		reader.reset();
	}
	else{
		std::shared_ptr<RawVolumeReader> reader;
		if (std::string(dataPath).find("engine") != std::string::npos || std::string(dataPath).find("knee") != std::string::npos || std::string(dataPath).find("181") != std::string::npos){
			reader = std::make_shared<RawVolumeReader>(dataPath.c_str(), dims, RawVolumeReader::dtUint8);
		}
		else{
			reader = std::make_shared<RawVolumeReader>(dataPath.c_str(), dims);
		}
		reader->OutputToVolumeByNormalizedValue(inputVolume);
		reader.reset();
	}
	inputVolume->spacing = spacing;
	inputVolume->initVolumeCuda();
	float3 posMin, posMax;
	inputVolume->GetPosRange(posMin, posMax);

	/********GL widget******/

	matrixMgr = std::make_shared<GLMatrixManager>(posMin, posMax);

	openGL = std::make_shared<DeformGLWidget>(matrixMgr);
	openGL->SetDeformModel(DEFORM_MODEL::OBJECT_SPACE);


	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setStencilBufferSize(8);
	format.setVersion(2, 0);
	format.setProfile(QSurfaceFormat::CoreProfile);
	openGL->setFormat(format); // must be called before the widget or its parent window gets shown

	meshDeformer = std::make_shared<MeshDeformProcessor>(&posMin.x, &posMax.x, meshResolution);
	meshDeformer->data_type = DATA_TYPE::USE_VOLUME;
	meshDeformer->setVolumeData(inputVolume);
	meshDeformer->SetLenses(&lenses);
	modelVolumeDeformer = std::make_shared<PhysicalVolumeDeformProcessor>(meshDeformer, inputVolume);
	
	//order matters!
	openGL->AddProcessor("1meshdeform", meshDeformer.get());
	openGL->AddProcessor("2physicalVolumeDeform", modelVolumeDeformer.get());
	
	volumeRenderable = std::make_shared<VolumeRenderableCUDA>(inputVolume);
	lensRenderable = std::make_shared<LensRenderable>(&lenses);
	meshRenderable = std::make_shared<MeshRenderable>(meshDeformer.get());
	
	volumeRenderable->rcp = rcp;
	volumeRenderable->setBlending(true);

	openGL->AddRenderable("2lenses", lensRenderable.get());
	openGL->AddRenderable("3model", meshRenderable.get());
	openGL->AddRenderable("4volume", volumeRenderable.get()); 
	//NOTE!! if need to blend other renderable result into the dvr, volume need to be rendered last
	//or else volume needs to be rendered first, since it is set to always pass the depth test
	

	rInteractor = std::make_shared<RegularInteractor>();
	rInteractor->setMatrixMgr(matrixMgr);
	lensInteractor = std::make_shared<LensInteractor>();
	lensInteractor->SetLenses(&lenses);
	openGL->AddInteractor("regular", rInteractor.get());
	openGL->AddInteractor("lens", lensInteractor.get());


#ifdef USE_OSVR
	vrWidget = std::make_shared<VRWidget>(matrixMgr);
	vrWidget->setWindowFlags(Qt::Window);
	vrVolumeRenderable = std::make_shared<VRVolumeRenderableCUDA>(inputVolume);
	vrWidget->AddRenderable("volume", vrVolumeRenderable.get());
	vrWidget->AddRenderable("lens", lensRenderable.get());
	openGL->SetVRWidget(vrWidget.get());
	vrVolumeRenderable->rcp = volumeRenderable->rcp;
#endif


	///********controls******/
	addLensBtn = new QPushButton("Add circle lens");
	addLineLensBtn = new QPushButton("Add a Virtual Retractor");
	delLensBtn = std::make_shared<QPushButton>("Delete the Virtual Retractor");
	//addCurveLensBtn = new QPushButton("Add curved band lens");
	saveStateBtn = std::make_shared<QPushButton>("Save State");
	loadStateBtn = std::make_shared<QPushButton>("Load State");
std::cout << posMin.x << " " << posMin.y << " " << posMin.z << std::endl;
std::cout << posMax.x << " " << posMax.y << " " << posMax.z << std::endl;

	QCheckBox* gridCheck = new QCheckBox("Show Back Face and Mesh", this);
	QCheckBox* cbBackFace = new QCheckBox("Show the Back Face", this);
	QCheckBox* cbChangeLensWhenRotateData = new QCheckBox("View Dependency", this);
	cbChangeLensWhenRotateData->setChecked(lensInteractor->changeLensWhenRotateData);
	QCheckBox* cbDrawInsicionOnCenterFace = new QCheckBox("Draw the Incision at the Center Face", this);
	cbDrawInsicionOnCenterFace->setChecked(lensRenderable->drawInsicionOnCenterFace);

	meshDeformer->setDeformForce(2000);
	QLabel *deformForceLabelLit = new QLabel("Deform Force:");
	//controlLayout->addWidget(deformForceLabelLit);
	QSlider *deformForceSlider = new QSlider(Qt::Horizontal);
	deformForceSlider->setRange(0, 50);
	deformForceSlider->setValue(meshDeformer->getDeformForce() / deformForceConstant);
	connect(deformForceSlider, SIGNAL(valueChanged(int)), this, SLOT(deformForceSliderValueChanged(int)));
	deformForceLabel = new QLabel(QString::number(meshDeformer->getDeformForce()));
	QHBoxLayout *deformForceLayout = new QHBoxLayout;
	deformForceLayout->addWidget(deformForceSlider);
	deformForceLayout->addWidget(deformForceLabel);


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
	//controlLayout->addWidget(addLensBtn);
	controlLayout->addWidget(addLineLensBtn);

	//controlLayout->addWidget(addCurveLensBtn);
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
	QSlider *transFuncP1LabelSlider = new QSlider(Qt::Horizontal);
	transFuncP1LabelSlider->setRange(0, 100);
	transFuncP1LabelSlider->setValue(volumeRenderable->rcp->transFuncP1 * 100);
	connect(transFuncP1LabelSlider, SIGNAL(valueChanged(int)), this, SLOT(transFuncP1LabelSliderValueChanged(int)));
	transFuncP1Label = new QLabel(QString::number(volumeRenderable->rcp->transFuncP1));
	QHBoxLayout *transFuncP1Layout = new QHBoxLayout;
	transFuncP1Layout->addWidget(transFuncP1LabelSlider);
	transFuncP1Layout->addWidget(transFuncP1Label);

	QLabel *transFuncP2SliderLabelLit = new QLabel("Transfer Function Lower Cut Off");
	QSlider *transFuncP2LabelSlider = new QSlider(Qt::Horizontal);
	transFuncP2LabelSlider->setRange(0, 100);
	transFuncP2LabelSlider->setValue(volumeRenderable->rcp->transFuncP2 * 100);
	connect(transFuncP2LabelSlider, SIGNAL(valueChanged(int)), this, SLOT(transFuncP2LabelSliderValueChanged(int)));
	transFuncP2Label = new QLabel(QString::number(volumeRenderable->rcp->transFuncP2));
	QHBoxLayout *transFuncP2Layout = new QHBoxLayout;
	transFuncP2Layout->addWidget(transFuncP2LabelSlider);
	transFuncP2Layout->addWidget(transFuncP2Label);

	QLabel *brLabelLit = new QLabel("Brightness of the volume: ");
	//controlLayout->addWidget(brLabelLit);
	QSlider* brSlider = new QSlider(Qt::Horizontal);
	brSlider->setRange(0, 40);
	brSlider->setValue(volumeRenderable->rcp->brightness * 20);
	connect(brSlider, SIGNAL(valueChanged(int)), this, SLOT(brSliderValueChanged(int)));
	brLabel = new QLabel(QString::number(volumeRenderable->rcp->brightness));
	QHBoxLayout *brLayout = new QHBoxLayout;
	brLayout->addWidget(brSlider);
	brLayout->addWidget(brLabel);

	QLabel *dsLabelLit = new QLabel("Density of the volume: ");
	//controlLayout->addWidget(dsLabelLit);
	QSlider* dsSlider = new QSlider(Qt::Horizontal);
	dsSlider->setRange(0, 40);
	dsSlider->setValue(volumeRenderable->rcp->density * 5);
	connect(dsSlider, SIGNAL(valueChanged(int)), this, SLOT(dsSliderValueChanged(int)));
	dsLabel = new QLabel(QString::number(volumeRenderable->rcp->density));
	QHBoxLayout *dsLayout = new QHBoxLayout;
	dsLayout->addWidget(dsSlider);
	dsLayout->addWidget(dsLabel);


	QLabel *laSliderLabelLit = new QLabel("Coefficient for Ambient Lighting: ");
	//controlLayout->addWidget(laSliderLabelLit);
	QSlider* laSlider = new QSlider(Qt::Horizontal);
	laSlider->setRange(0, 50);
	laSlider->setValue(volumeRenderable->rcp->la * 10);
	connect(laSlider, SIGNAL(valueChanged(int)), this, SLOT(laSliderValueChanged(int)));
	laLabel = new QLabel(QString::number(volumeRenderable->rcp->la));
	QHBoxLayout *laLayout = new QHBoxLayout;
	laLayout->addWidget(laSlider);
	laLayout->addWidget(laLabel);

	QLabel *ldSliderLabelLit = new QLabel("Coefficient for Diffusial Lighting: ");
	//controlLayout->addWidget(ldSliderLabelLit);
	QSlider* ldSlider = new QSlider(Qt::Horizontal);
	ldSlider->setRange(0, 50);
	ldSlider->setValue(volumeRenderable->rcp->ld * 10);
	connect(ldSlider, SIGNAL(valueChanged(int)), this, SLOT(ldSliderValueChanged(int)));
	ldLabel = new QLabel(QString::number(volumeRenderable->rcp->ld));
	QHBoxLayout *ldLayout = new QHBoxLayout;
	ldLayout->addWidget(ldSlider);
	ldLayout->addWidget(ldLabel);

	QLabel *lsSliderLabelLit = new QLabel("Coefficient for Specular Lighting: ");
	//controlLayout->addWidget(lsSliderLabelLit);
	QSlider* lsSlider = new QSlider(Qt::Horizontal);
	lsSlider->setRange(0, 50);
	lsSlider->setValue(volumeRenderable->rcp->ls * 10);
	connect(lsSlider, SIGNAL(valueChanged(int)), this, SLOT(lsSliderValueChanged(int)));
	lsLabel = new QLabel(QString::number(volumeRenderable->rcp->ls));
	QHBoxLayout *lsLayout = new QHBoxLayout;
	lsLayout->addWidget(lsSlider);
	lsLayout->addWidget(lsLabel);


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
	//connect(addCurveLensBtn, SIGNAL(clicked()), this, SLOT(AddCurveLens()));
	connect(delLensBtn.get(), SIGNAL(clicked()), this, SLOT(SlotDelLens()));
	connect(saveStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotSaveState()));
	connect(loadStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotLoadState()));
	connect(addMeshResPushButton, SIGNAL(clicked()), this, SLOT(SlotAddMeshRes()));
	connect(minusMeshResPushButton, SIGNAL(clicked()), this, SLOT(SlotMinusMeshRes()));
	
	connect(gridCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleGrid(bool)));
	connect(cbBackFace, SIGNAL(clicked(bool)), this, SLOT(SlotToggleBackFace(bool)));
	connect(cbDrawInsicionOnCenterFace, SIGNAL(clicked(bool)), this, SLOT(SlotToggleCbDrawInsicionOnCenterFace(bool)));


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


void Window::AddCurveLens()
{
	lensRenderable->AddCurveLens();
}


void Window::SlotToggleGrid(bool b)
{
	meshRenderable->SetVisibility(b);
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

		deformForceSliderValueChanged(f / deformForceConstant);
		meshResolution = res;
		meshDeformer->meshResolution = meshResolution;
		meshDeformer->setReinitiationNeed();
		meshResLabel->setText(QString::number(meshResolution));
	}
}

void Window::deformForceSliderValueChanged(int v)
{
	float newForce = deformForceConstant*v;
	deformForceLabel->setText(QString::number(newForce));
	meshDeformer->setDeformForce(newForce);
}



void Window::transFuncP1LabelSliderValueChanged(int v)
{
	volumeRenderable->rcp->transFuncP1 = 1.0*v / 100;
	transFuncP1Label->setText(QString::number(1.0*v / 100));
}
void Window::transFuncP2LabelSliderValueChanged(int v)
{
	volumeRenderable->rcp->transFuncP2 = 1.0*v / 100;
	transFuncP2Label->setText(QString::number(1.0*v / 100));
}

void Window::brSliderValueChanged(int v)
{
	volumeRenderable->rcp->brightness = v*1.0 / 20.0;
	brLabel->setText(QString::number(volumeRenderable->rcp->brightness));
}
void Window::dsSliderValueChanged(int v)
{
	volumeRenderable->rcp->density = v*1.0 / 5.0;
	dsLabel->setText(QString::number(volumeRenderable->rcp->density));
}

void Window::laSliderValueChanged(int v)
{
	volumeRenderable->rcp->la = 1.0*v / 10;
	laLabel->setText(QString::number(1.0*v / 10));

}
void Window::ldSliderValueChanged(int v)
{
	volumeRenderable->rcp->ld = 1.0*v / 10;
	ldLabel->setText(QString::number(1.0*v / 10));
}
void Window::lsSliderValueChanged(int v)
{
	volumeRenderable->rcp->ls = 1.0*v / 10;
	lsLabel->setText(QString::number(1.0*v / 10));
}

void Window::SlotRbUniformChanged(bool b)
{
	if (b){
		meshDeformer->elasticityMode = 0;
		meshDeformer->setReinitiationNeed();
		inputVolume->reset();
	}
}
void Window::SlotRbDensityChanged(bool b)
{
	if (b){
		meshDeformer->elasticityMode = 1;
		meshDeformer->setReinitiationNeed();
		inputVolume->reset();
	}
}
void Window::SlotRbTransferChanged(bool b)
{
	if (b){
		meshDeformer->elasticityMode = 2;
		meshDeformer->setReinitiationNeed();
		inputVolume->reset();
	}
}
void Window::SlotRbGradientChanged(bool b)
{
	if (b){
		meshDeformer->elasticityMode = 3;
		meshDeformer->setReinitiationNeed();
		inputVolume->reset();
	}
}


void Window::SlotDelLens()
{
	lensRenderable->DelLens();
	inputVolume->reset();
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

void Window::SlotToggleCbDrawInsicionOnCenterFace(bool b)
{
	if (b)
		lensRenderable->drawInsicionOnCenterFace = true;
	else
		lensRenderable->drawInsicionOnCenterFace = false;
}