#include <window.h>
#include <iostream>

#include "myDefineRayCasting.h"
#include "GLWidget.h"
#include "Volume.h"
#include "RawVolumeReader.h"
#include "DataMgr.h"
#include "VecReader.h"
#include "GLMatrixManager.h"
#include "ScreenMarker.h"
#include "LabelVolumeProcessor.h"
#include "VolumeRenderableCUDA.h"
#include "VolumeRenderableImmerCUDA.h"
#include "mouse/RegularInteractor.h"
#include "mouse/ImmersiveInteractor.h"
#include "mouse/ScreenBrushInteractor.h"
#include "LabelVolumeProcessor.h"
#include "ViewpointEvaluator.h"
#include "GLWidgetQtDrawing.h"
#include "AnimationByMatrixProcessor.h"
#include "Particle.h"

#include "PositionBasedDeformProcessor.h"
#include "MatrixMgrRenderable.h"
#include "InfoGuideRenderable.h"
#include "BinaryTuplesReader.h"
#include "DeformFrameRenderable.h"
#include "SphereRenderable.h"
#include "PolyRenderable.h"
#include "PolyMesh.h"

#include "PlyVTKReader.h"

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


Window::Window()
{
	setWindowTitle(tr("Egocentric VR Volume Visualization"));

	////////////////////////////////prepare data////////////////////////////////
	//////////////////Volume and RayCastingParameters
	std::shared_ptr<DataMgr> dataMgr;
	dataMgr = std::make_shared<DataMgr>();
	const std::string dataPath = dataMgr->GetConfig("VOLUME_DATA_PATH");

	rcp = std::make_shared<RayCastingParameters>();

	std::string subfolder;
	
	//Volume::rawFileInfo(dataPath, dims, spacing, rcp, subfolder);
	//rcp->useColor = false;

	dims = make_int3(160, 224, 64);
	spacing = make_float3(1, 1, 1);
	rcp = std::make_shared<RayCastingParameters>(0.6, 0.0, 0.0, 0.29, 0.0, 0.35, 512, 0.25f, 1.0, false); //for 181
	subfolder = "beetle";
	rcp->use2DInteg = false;

	std::shared_ptr<RayCastingParameters> rcpMini = rcp;// std::make_shared<RayCastingParameters>(1.8, 1.0, 1.5, 1.0, 0.3, 2.6, 512, 0.25f, 1.0, false);

	inputVolume = std::make_shared<Volume>(true);
	
	if (std::string(dataPath).find(".vec") != std::string::npos){
		std::shared_ptr<VecReader> reader;
		reader = std::make_shared<VecReader>(dataPath.c_str());
		reader->OutputToVolumeByNormalizedVecMag(inputVolume);
		//reader->OutputToVolumeByNormalizedVecDownSample(inputVolume,2);
		//reader->OutputToVolumeByNormalizedVecUpSample(inputVolume, 2);
		//reader->OutputToVolumeByNormalizedVecMagWithPadding(inputVolume,10);
		reader.reset();
	}
	else{
		std::shared_ptr<RawVolumeReader> reader;
		if (std::string(dataPath).find("engine") != std::string::npos || std::string(dataPath).find("knee") != std::string::npos || std::string(dataPath).find("181") != std::string::npos || std::string(dataPath).find("Bucky") != std::string::npos || std::string(dataPath).find("bloodCell") != std::string::npos || std::string(dataPath).find("Lobster") != std::string::npos || std::string(dataPath).find("Orange") != std::string::npos){
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
	

	if (rcp->use2DInteg){
		inputVolume->computeGradient();
		rcp->secondCutOffLow = 0.19f;
		rcp->secondCutOffHigh = 0.72f;
		rcp->secondNormalizationCoeff = inputVolume->maxGadientLength;
	}

	bool channelSkelViewReady = false;
	if (channelSkelViewReady){
		channelVolume = std::make_shared<Volume>(true);
		std::shared_ptr<RawVolumeReader> reader2 = std::make_shared<RawVolumeReader>((subfolder + "/cleanedChannel.raw").c_str(), dims, RawVolumeReader::dtFloat32);
		reader2->OutputToVolumeByNormalizedValue(channelVolume);
		channelVolume->initVolumeCuda();
		reader2.reset();

		skelVolume = std::make_shared<Volume>();
		std::shared_ptr<RawVolumeReader> reader4 = std::make_shared<RawVolumeReader>((subfolder + "/skel.raw").c_str(), dims, RawVolumeReader::dtFloat32);
		reader4->OutputToVolumeByNormalizedValue(skelVolume);
		skelVolume->initVolumeCuda();
		reader4.reset();
	}

	//labelVolCUDA = std::make_shared<VolumeCUDA>();
	//labelVolCUDA->VolumeCUDA_init(dims, (unsigned short *)0, 1, 1);
	//labelVolLocal = new unsigned short[dims.x*dims.y*dims.z];
	//memset(labelVolLocal, 0, sizeof(unsigned short)*dims.x*dims.y*dims.z);

	////////////////matrix manager
	float3 posMin, posMax;
	inputVolume->GetPosRange(posMin, posMax);
	matrixMgr = std::make_shared<GLMatrixManager>(posMin, posMax);
	matrixMgr->setDefaultForImmersiveMode();		
	if (std::string(dataPath).find("engine") != std::string::npos){
		matrixMgr->moveEyeInLocalByModeMat(make_float3(70, -20, 60));
	}

	matrixMgrMini = std::make_shared<GLMatrixManager>(posMin, posMax);





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
		positionBasedDeformProcessor = std::make_shared<PositionBasedDeformProcessor>(inputVolume, matrixMgr, channelVolume);
		openGL->AddProcessor("1positionBasedDeformProcessor", positionBasedDeformProcessor.get());

		animationByMatrixProcessor = std::make_shared<AnimationByMatrixProcessor>(matrixMgr);
		animationByMatrixProcessor->setViews(views);
		openGL->AddProcessor("animationByMatrixProcessor", animationByMatrixProcessor.get());
	}

	//lvProcessor = std::make_shared<LabelVolumeProcessor>(labelVolCUDA);
	//lvProcessor->setScreenMarker(sm);
	//lvProcessor->rcp = rcp;
	//openGL->AddProcessor("screenMarkerLabelVolumeProcessor", lvProcessor.get());


	//////////////////////////////// Renderable ////////////////////////////////	
	//volumeRenderable = std::make_shared<VolumeRenderableImmerCUDA>(inputVolume, labelVolCUDA, positionBasedDeformProcessor);
	//volumeRenderable->rcp = rcp;
	//openGL->AddRenderable("1volume", volumeRenderable.get());
	////volumeRenderable->setScreenMarker(sm);

	//deformFrameRenderable = std::make_shared<DeformFrameRenderable>(matrixMgr, positionBasedDeformProcessor);
	//openGL->AddRenderable("0deform", deformFrameRenderable.get()); 
	//volumeRenderable->setBlending(true); //only when needed when want the deformFrameRenderable

	matrixMgrRenderable = std::make_shared<MatrixMgrRenderable>(matrixMgr);
	openGL->AddRenderable("2volume", matrixMgrRenderable.get()); 


	const std::string polyDataPath = dataMgr->GetConfig("POLY_DATA_PATH");

	polyMesh = std::make_shared<PolyMesh>();

	PlyVTKReader plyVTKReader;
	plyVTKReader.readPLYByVTK(polyDataPath.c_str(), polyMesh.get());
	
	polyRenderable = std::make_shared<PolyRenderable>(polyMesh);
	
	openGL->AddRenderable("mm", polyRenderable.get());



	//////////////////////////////// Interactor ////////////////////////////////
	immersiveInteractor = std::make_shared<ImmersiveInteractor>();
	immersiveInteractor->setMatrixMgr(matrixMgr);
	regularInteractor = std::make_shared<RegularInteractor>();
	regularInteractor->setMatrixMgr(matrixMgrMini);
	regularInteractor->isActive = false;
	immersiveInteractor->isActive = true;

	if (channelSkelViewReady){
		immersiveInteractor->noMoveMode = true;
	}
	openGL->AddInteractor("1modelImmer", immersiveInteractor.get());
	openGL->AddInteractor("2modelReg", regularInteractor.get());
	
	//sbInteractor = std::make_shared<ScreenBrushInteractor>();
	////sbInteractor->setScreenMarker(sm);
	//openGL->AddInteractor("3screenMarker", sbInteractor.get());
	

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
	std::cout << posMin.x << " " << posMin.y << " " << posMin.z << std::endl;
	std::cout << posMax.x << " " << posMax.y << " " << posMax.z << std::endl;
	controlLayout->addWidget(saveStateBtn.get());
	controlLayout->addWidget(loadStateBtn.get());

	if (channelSkelViewReady){
		QCheckBox* isDeformEnabled = new QCheckBox("Enable Deform", this);
		isDeformEnabled->setChecked(positionBasedDeformProcessor->isActive);
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
	vrWidget->AddRenderable("2info", infoGuideRenderable.get());
	
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

void Window::transFuncP1LabelSliderValueChanged(int v)
{
	rcp->transFuncP1 = 1.0*v / 100;
	transFuncP1Label->setText(QString::number(1.0*v / 100));
}
void Window::transFuncP2LabelSliderValueChanged(int v)
{
	rcp->transFuncP2 = 1.0*v / 100;
	transFuncP2Label->setText(QString::number(1.0*v / 100));
}

void Window::transFuncP1SecondLabelSliderValueChanged(int v)
{
	rcp->secondCutOffHigh = 1.0*v / 100;
	transFuncP1SecondLabel->setText(QString::number(1.0*v / 100));
}
void Window::transFuncP2SecondLabelSliderValueChanged(int v)
{
	rcp->secondCutOffLow = 1.0*v / 100;
	transFuncP2SecondLabel->setText(QString::number(1.0*v / 100));
}


void Window::brSliderValueChanged(int v)
{
	rcp->brightness = v*1.0 / 20.0;
	brLabel->setText(QString::number(rcp->brightness));
}

void Window::dsSliderValueChanged(int v)
{
	rcp->density = v*1.0 / 20.0;
	dsLabel->setText(QString::number(rcp->density));
}

void Window::laSliderValueChanged(int v)
{
	rcp->la = 1.0*v / 10;
	laLabel->setText(QString::number(1.0*v / 10));

}
void Window::ldSliderValueChanged(int v)
{
	rcp->ld = 1.0*v / 10;
	ldLabel->setText(QString::number(1.0*v / 10));
}
void Window::lsSliderValueChanged(int v)
{
	rcp->ls = 1.0*v / 10;
	lsLabel->setText(QString::number(1.0*v / 10));
}

void Window::isDeformEnabledClicked(bool b)
{
	if (b){
		positionBasedDeformProcessor->isActive = true;
		positionBasedDeformProcessor->reset();
	}
	else{
		positionBasedDeformProcessor->isActive = false;
		inputVolume->reset();
		channelVolume->reset();
	}
}

void Window::isBrushingClicked()
{
	sbInteractor->isActive = !sbInteractor->isActive;
}

void Window::moveToOptimalBtnClicked()
{

}

void Window::SlotOriVolumeRb(bool b)
{
	if (b)
		volumeRenderable->setVolume(inputVolume);
}

void Window::SlotChannelVolumeRb(bool b)
{
	if (b)
	{
		if (channelVolume){
			volumeRenderable->setVolume(channelVolume);
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
			volumeRenderable->setVolume(skelVolume);
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

void Window::zSliderValueChanged(int v)
{
	helper.z = v;
}

void Window::updateLabelVolBtnClicked()
{

}

void Window::findGeneralOptimalBtnClicked()
{

}


void Window::turnOffGlobalGuideBtnClicked()
{
	infoGuideRenderable->changeWhetherGlobalGuideMode(false);
}

void Window::redrawBtnClicked()
{

}

void Window::doTourBtnClicked()
{
	animationByMatrixProcessor->startAnimation();
}

void Window::saveScreenBtnClicked()
{
	openGL->saveCurrentImage();
}
