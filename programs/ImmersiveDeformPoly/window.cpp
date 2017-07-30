#include <window.h>
#include <iostream>

#include "myDefineRayCasting.h"
#include "GLWidget.h"
#include "Volume.h"
#include "RawVolumeReader.h"
#include "DataMgr.h"
#include "VecReader.h"
#include "GLMatrixManager.h"
#include "LabelVolumeProcessor.h"
#include "VolumeRenderableCUDA.h"
#include "VolumeRenderableImmerCUDA.h"
#include "mouse/RegularInteractor.h"
#include "mouse/ImmersiveInteractor.h"
#include "mouse/ScreenBrushInteractor.h"
#include "LabelVolumeProcessor.h"
#include "ViewpointEvaluator.h"
#include "AnimationByMatrixProcessor.h"
#include "Particle.h"

#include "MatrixMgrRenderable.h"
#include "DeformFrameRenderable.h"
#include "SphereRenderable.h"
#include "PolyRenderable.h"
#include "PolyMesh.h"

#include "PlyVTKReader.h"
#include "VTPReader.h"

#include "PositionBasedDeformProcessor.h"

#include <thrust/device_vector.h>

#ifdef USE_OSVR
#include "VRWidget.h"
#include "VRVolumeRenderableCUDA.h"
#endif

#ifdef USE_LEAP
#include <Leap.h>
#include "leap/LeapListener.h"
#include "leap/MatrixLeapInteractor.h"
#endif

#include "VolumeRenderableCUDAKernel.h"

bool channelSkelViewReady = true;

void rawfileInfo(std::string dataPath, DataType & channelVolDataType)
{
	if (std::string(dataPath).find("sphere") != std::string::npos){
		channelVolDataType = RawVolumeReader::dtUint16;
	}
	else if (std::string(dataPath).find("iso_t") != std::string::npos){
		channelVolDataType = RawVolumeReader::dtFloat32;
	}
	else{
		std::cout << "file name not defined" << std::endl;
		exit(0);
	}
}


Window::Window()
{
	setWindowTitle(tr("Egocentric VR Volume Visualization"));

	////////////////////////////////prepare data////////////////////////////////
	//////////////////Volume and RayCastingParameters
	std::shared_ptr<DataMgr> dataMgr;
	dataMgr = std::make_shared<DataMgr>();
	const std::string polyDataPath = dataMgr->GetConfig("POLY_DATA_PATH");

	std::shared_ptr<RayCastingParameters> rcpMini = std::make_shared<RayCastingParameters>(1.8, 1.0, 1.5, 1.0, 0.3, 2.6, 512, 0.25f, 1.0, false);
	std::string subfolder;


	float disThr;
	float3 shift;
	int3 dims;
	float3 spacing;
	PolyMesh::dataParameters(polyDataPath, dims, spacing, disThr, shift, subfolder);


	DataType channelVolDataType;
	if (channelSkelViewReady){
		rawfileInfo(polyDataPath, channelVolDataType);
	}

	if (channelSkelViewReady){
		channelVolume = std::make_shared<Volume>(true);
		std::shared_ptr<RawVolumeReader> reader2 = std::make_shared<RawVolumeReader>((subfolder + "/cleanedChannel.raw").c_str(), dims, channelVolDataType);
		reader2->OutputToVolumeByNormalizedValue(channelVolume);
		channelVolume->initVolumeCuda();
		reader2.reset();
	}

	polyMesh = std::make_shared<PolyMesh>();
	if (std::string(polyDataPath).find(".ply") != std::string::npos){
		PlyVTKReader plyVTKReader;
		plyVTKReader.readPLYByVTK(polyDataPath.c_str(), polyMesh.get());
	}
	else{
		VTPReader reader;
		reader.readFile(polyDataPath.c_str(), polyMesh.get());
	}
	polyMesh->doShift(shift); //do it before setVertexCoordsOri()!!!
	polyMesh->setVertexCoordsOri();

	polyMesh->opacity = 1.0;// 0.5;

	////////////////matrix manager
	float3 posMin, posMax;
	polyMesh->GetPosRange(posMin, posMax);
	std::cout << "posMin: " << posMin.x << " " << posMin.y << " " << posMin.z << std::endl;
	std::cout << "posMax: " << posMax.x << " " << posMax.y << " " << posMax.z << std::endl;
	matrixMgr = std::make_shared<GLMatrixManager>(posMin, posMax);
	matrixMgr->setDefaultForImmersiveMode();		

	/********GL widget******/
	openGL = std::make_shared<GLWidget>(matrixMgr);
	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setStencilBufferSize(8);
	format.setVersion(2, 0);
	format.setProfile(QSurfaceFormat::CoreProfile);
	openGL->setFormat(format); // must be called before the widget or its parent window gets shown


	//////////////////////////////// Processor ////////////////////////////////
	if (channelSkelViewReady){
		positionBasedDeformProcessor = std::make_shared<PositionBasedDeformProcessor>(polyMesh, matrixMgr, channelVolume);
		openGL->AddProcessor("1positionBasedDeformProcessor", positionBasedDeformProcessor.get());


		positionBasedDeformProcessor->deformationScale = 2; 
		positionBasedDeformProcessor->deformationScaleVertical = 2.5;


		//animationByMatrixProcessor = std::make_shared<AnimationByMatrixProcessor>(matrixMgr);
		//animationByMatrixProcessor->setViews(views);
		//openGL->AddProcessor("animationByMatrixProcessor", animationByMatrixProcessor.get());
	}


	//////////////////////////////// Renderable ////////////////////////////////	


	//deformFrameRenderable = std::make_shared<DeformFrameRenderable>(matrixMgr, positionBasedDeformProcessor);
	//openGL->AddRenderable("0deform", deformFrameRenderable.get()); 
	//volumeRenderable->setBlending(true); //only when needed when want the deformFrameRenderable


	if (channelSkelViewReady){
		volumeRenderable = std::make_shared<VolumeRenderableCUDA>(channelVolume);
		volumeRenderable->rcp = rcpMini;
		openGL->AddRenderable("2volume", volumeRenderable.get());
		volumeRenderable->SetVisibility(false);
	}

	matrixMgrRenderable = std::make_shared<MatrixMgrRenderable>(matrixMgr);
	openGL->AddRenderable("3matrix", matrixMgrRenderable.get()); 

	polyRenderable = std::make_shared<PolyRenderable>(polyMesh);
	openGL->AddRenderable("1poly", polyRenderable.get());
	
	//////////////////////////////// Interactor ////////////////////////////////
	immersiveInteractor = std::make_shared<ImmersiveInteractor>();
	immersiveInteractor->setMatrixMgr(matrixMgr);
	regularInteractor = std::make_shared<RegularInteractor>();
	regularInteractor->setMatrixMgr(matrixMgrMini);
	regularInteractor->isActive = false;
	immersiveInteractor->isActive = true;

	if (channelSkelViewReady){
		//immersiveInteractor->noMoveMode = true;
	}
	openGL->AddInteractor("1modelImmer", immersiveInteractor.get());
	openGL->AddInteractor("2modelReg", regularInteractor.get());
	


#ifdef USE_LEAP
	listener = new LeapListener();
	controller = new Leap::Controller();
	controller->setPolicyFlags(Leap::Controller::PolicyFlag::POLICY_OPTIMIZE_HMD);
	controller->addListener(*listener);

	matrixMgrLeapInteractor = std::make_shared<MatrixLeapInteractor>(matrixMgr);
	matrixMgrLeapInteractor->SetActor(openGL.get());
	listener->AddLeapInteractor("matrixMgr", (LeapInteractor*)(matrixMgrLeapInteractor.get()));
#endif

	
	///********controls******/
	QHBoxLayout *mainLayout = new QHBoxLayout;
	
	QVBoxLayout *controlLayout = new QVBoxLayout;
	
	saveStateBtn = std::make_shared<QPushButton>("Save State");
	loadStateBtn = std::make_shared<QPushButton>("Load State");

	controlLayout->addWidget(saveStateBtn.get());
	controlLayout->addWidget(loadStateBtn.get());

	if (channelSkelViewReady){
		QCheckBox* isDeformEnabled = new QCheckBox("Enable Deform", this);
		//isDeformEnabled->setChecked(positionBasedDeformProcessor->isActive);
		controlLayout->addWidget(isDeformEnabled);
		connect(isDeformEnabled, SIGNAL(clicked(bool)), this, SLOT(isDeformEnabledClicked(bool)));
	}


	QGroupBox *eyePosGroup = new QGroupBox(tr("eye position"));
	QHBoxLayout *eyePosLayout = new QHBoxLayout;
	QVBoxLayout *eyePosLayout2 = new QVBoxLayout;
	QLabel *eyePosxLabel = new QLabel("x");
	QLabel *eyePosyLabel = new QLabel("y");
	QLabel *eyePoszLabel = new QLabel("z");
	eyePosLineEdit = new QLineEdit;
	QPushButton *eyePosBtn = new QPushButton("Apply");
	eyePosLayout->addWidget(eyePosxLabel);
	eyePosLayout->addWidget(eyePosyLabel);
	eyePosLayout->addWidget(eyePoszLabel);
	eyePosLayout->addWidget(eyePosLineEdit);
	eyePosLayout2->addLayout(eyePosLayout);
	eyePosLayout2->addWidget(eyePosBtn);
	eyePosGroup->setLayout(eyePosLayout2);
	controlLayout->addWidget(eyePosGroup);

	QGroupBox *groupBox = new QGroupBox(tr("volume selection"));
	QHBoxLayout *deformModeLayout = new QHBoxLayout;
	oriVolumeRb = std::make_shared<QRadioButton>(tr("&original"));
	channelVolumeRb = std::make_shared<QRadioButton>(tr("&channel"));
	skelVolumeRb = std::make_shared<QRadioButton>(tr("&skeleton"));
	oriVolumeRb->setChecked(true);
	deformModeLayout->addWidget(oriVolumeRb.get());
	deformModeLayout->addWidget(channelVolumeRb.get());
	deformModeLayout->addWidget(skelVolumeRb.get());
	groupBox->setLayout(deformModeLayout);
	controlLayout->addWidget(groupBox);
	connect(oriVolumeRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotOriVolumeRb(bool)));
	connect(channelVolumeRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotChannelVolumeRb(bool)));
	connect(skelVolumeRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotSkelVolumeRb(bool)));

	QGroupBox *groupBox2 = new QGroupBox(tr("volume selection"));
	QHBoxLayout *deformModeLayout2 = new QHBoxLayout;
	immerRb = std::make_shared<QRadioButton>(tr("&immersive mode"));
	nonImmerRb = std::make_shared<QRadioButton>(tr("&non immer"));
	immerRb->setChecked(true);
	deformModeLayout2->addWidget(immerRb.get());
	deformModeLayout2->addWidget(nonImmerRb.get());
	groupBox2->setLayout(deformModeLayout2);
	controlLayout->addWidget(groupBox2);
	connect(immerRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotImmerRb(bool)));
	connect(nonImmerRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotNonImmerRb(bool)));

	QPushButton *saveScreenBtn = new QPushButton("Save the current screen");
	controlLayout->addWidget(saveScreenBtn);
	connect(saveScreenBtn, SIGNAL(clicked()), this, SLOT(saveScreenBtnClicked()));


	

	controlLayout->addStretch();

	connect(saveStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotSaveState()));
	connect(loadStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotLoadState()));
	connect(eyePosBtn, SIGNAL(clicked()), this, SLOT(applyEyePos()));


	mainLayout->addWidget(openGL.get(), 5);
	mainLayout->addLayout(controlLayout, 1);
	setLayout(mainLayout);


#ifdef USE_OSVR
	vrWidget = std::make_shared<VRWidget>(matrixMgr);
	vrWidget->setWindowFlags(Qt::Window);
	vrVolumeRenderable = std::make_shared<VRVolumeRenderableCUDA>(inputVolume);

	vrWidget->AddRenderable("1volume", vrVolumeRenderable.get());
	
	openGL->SetVRWidget(vrWidget.get());
	vrVolumeRenderable->rcp = rcp;
#endif

}






Window::~Window()
{

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

void Window::applyEyePos()
{
	QString s = eyePosLineEdit->text();
	QStringList sl = s.split(QRegExp("[\\s,]+"));
	matrixMgr->moveEyeInLocalByModeMat(make_float3(sl[0].toFloat(), sl[1].toFloat(), sl[2].toFloat()));
}

void Window::isDeformEnabledClicked(bool b)
{
	if (b){
		positionBasedDeformProcessor->isActive = true;
		positionBasedDeformProcessor->reset();
	}
	else{
		positionBasedDeformProcessor->isActive = false;
		//inputVolume->reset();
		channelVolume->reset();
	}
}


void Window::SlotOriVolumeRb(bool b)
{
	volumeRenderable->SetVisibility(false);
	polyRenderable->SetVisibility(true);
}

void Window::SlotChannelVolumeRb(bool b)
{
	if (b)
	{
		if (channelVolume){
			//volumeRenderable->setVolume(channelVolume);
			volumeRenderable->SetVisibility(true);
			polyRenderable->SetVisibility(false);
		}
		else{
			std::cout << "channelVolume not set!!" << std::endl;
			oriVolumeRb->setChecked(true);
			SlotOriVolumeRb(true);
		}
	}
}

void Window::SlotSkelVolumeRb(bool b)
{
	if (b)
	{
		if (skelVolume){
			//volumeRenderable->setVolume(skelVolume);
		}
		else{
			std::cout << "skelVolume not set!!" << std::endl;
			oriVolumeRb->setChecked(true);
			SlotOriVolumeRb(true);
		}
	}
}

void Window::SlotImmerRb(bool b)
{
	if (b){
		regularInteractor->isActive = false;
		immersiveInteractor->isActive = true;
		openGL->matrixMgr = matrixMgr;
		//matrixMgr->setDefaultForImmersiveMode();
	}
}

void Window::SlotNonImmerRb(bool b)
{
	if (b)
	{
		regularInteractor->isActive = true;
		immersiveInteractor->isActive = false;
		openGL->matrixMgr = matrixMgrMini;
	}
}

void Window::doTourBtnClicked()
{
	//animationByMatrixProcessor->startAnimation();
}

void Window::saveScreenBtnClicked()
{
	openGL->saveCurrentImage();
}
