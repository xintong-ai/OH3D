#include <window.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>

#include "myDefineRayCasting.h"
#include "GLWidget.h"
#include "Volume.h"
#include "RawVolumeReader.h"
#include "DataMgr.h"
#include "VecReader.h"
#include "GLMatrixManager.h"
#include "mouse/RegularInteractor.h"
#include "mouse/ImmersiveInteractor.h"
#include "AnimationByMatrixProcessor.h"
#include "Particle.h"

#include "MatrixMgrRenderable.h"
#include "BinaryTuplesReader.h"
#include "DeformFrameRenderable.h"
#include "SphereRenderable.h"
#include "PolyRenderable.h"
#include "PolyMesh.h"

#include "PlyVTKReader.h"
#include "VTPReader.h"


#include "PositionBasedDeformProcessor.h"
#include "PositionBasedDeformProcessorForTV.h"


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


//assume the tuple file is always ready, or cannot run
bool channelSkelViewReady = true;

void fileInfo(std::string dataPath, DataType & channelVolDataType, float3 &shift, int3 & dims, std::string & subfolder)
{
	if (std::string(dataPath).find("rbcs") != std::string::npos){
		shift = make_float3(5, 3, 0);

		//dims = make_int3(65, 225, 161) + make_int3(shift.x, shift.y, shift.z);
		dims = make_int3(70, 121, 161);

		//spacing = make_float3(1, 1, 1);
		//rcp = std::make_shared<RayCastingParameters>(1.8, 1.0, 1.5, 0.9, 0.3, 2.6, 256, 0.25f, 1.0, false);
		subfolder = "bloodCell";
		subfolder = "D:/Data/Lin/reducedBloodCell";
		//channelVolDataType = RawVolumeReader::dtUint8;
		channelVolDataType = RawVolumeReader::dtInt32;	//note!! for time varying case, the channel volume is not defined in the same way

	}
	else{
		std::cout << "file name not defined in fileInfo()" << std::endl;
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
	
	std::shared_ptr<RayCastingParameters> rcpForChannelSkel = std::make_shared<RayCastingParameters>(1.8, 1.0, 1.5, 1.0, 0.3, 2.6, 512, 0.25f, 1.0, false);
	std::string subfolder;
	float3 shift;
	const std::string polyDataPath = dataMgr->GetConfig("POLY_DATA_PATH");

	DataType channelVolDataType;
	fileInfo(polyDataPath, channelVolDataType, shift, dims, subfolder);
	spacing = make_float3(1, 1, 1);


	positionBasedDeformProcessorForTV = std::make_shared<PositionBasedDeformProcessorForTV>();
		
	/*
	//single time step
	polyMesh = std::make_shared<PolyMesh>();
	if (std::string(polyDataPath).find(".ply") != std::string::npos){
		PlyVTKReader plyVTKReader;
		plyVTKReader.readPLYByVTK(polyDataPath.c_str(), polyMesh.get());
	}
	else{
		VTPReader reader;
		reader.readFile(polyDataPath.c_str(), polyMesh.get());
	}
	polyMesh->setAssisParticle((subfolder+"/polyMeshRegions.mytup").c_str());
	//polyMesh->setVertexCoordsOri(); //not needed when vertex coords need not to change
	
	//normally when a particle already used valTuple, we do not need to use val
	//but here use val to record the amount of shiftness?
	polyMesh->particle->val.resize(polyMesh->particle->numParticles, 0);
	polyMesh->particle->valMax = 0;
	polyMesh->particle->valMin = 0;

	polyMesh->doShift(shift); //note the order of shift, and processing assistParticle and cleanedChannel volume
	*/


	for (int i = positionBasedDeformProcessorForTV->timeStart; i <= positionBasedDeformProcessorForTV->timeEnd; i++){
		std::cout << "reading data of timestep " << i << std::endl;
		//single time step
		std::shared_ptr<PolyMesh> polyMesh = std::make_shared<PolyMesh>();
		std::stringstream ss;
		ss << std::setw(4) << std::setfill('0') << i;
		std::string s = ss.str();
		std::string fname = subfolder + "/marked-reduced-rbcs-" + s + ".vtp";

		VTPReader reader;
		reader.readFile(fname.c_str(), polyMesh.get());
		
		polyMesh->setAssisParticle((subfolder + "/marked-reduced-rbcs-" + s + "-polyMeshRegions.mytup").c_str());
		//polyMesh->setVertexCoordsOri(); //not needed when vertex coords need not to change

		//normally when a particle already used valTuple, we do not need to use val
		//but here use val to record the amount of shiftness after deformation
		polyMesh->particle->val.resize(polyMesh->particle->numParticles, 0);
		polyMesh->particle->valMax = 0;
		polyMesh->particle->valMin = 0;

		polyMesh->doShift(shift); //note the order of shift, and processing assistParticle and cleanedChannel volume

		//create the cellMaps
		if (i > positionBasedDeformProcessorForTV->timeStart){
			std::shared_ptr<Particle> lastParticle = (positionBasedDeformProcessorForTV->polyMeshes.back())->particle;
			int n = lastParticle->numParticles;
			std::vector<int> cellMap(n, -1);

			int m = polyMesh->particle->numParticles;
			int tupleCount = polyMesh->particle->tupleCount; //should be 11
			for (int i = 0; i < n; i++){
				int lastLabel = lastParticle->valTuple[i * tupleCount + 10];
				bool notFound = true;

				for (int j = 0; j < m && notFound; j++){
					if (lastLabel == polyMesh->particle->valTuple[tupleCount * j + 10]){
						cellMap[i] = j;
						notFound = false;
					}
				}
			}
			positionBasedDeformProcessorForTV->cellMaps.push_back(cellMap);
		}
		positionBasedDeformProcessorForTV->polyMeshes.push_back(polyMesh);

		int n = polyMesh->particle->numParticles;
		int tupleCount = polyMesh->particle->tupleCount; //should be 11
		for (int i = 0; i < n; i++){
			int label = polyMesh->particle->valTuple[i * tupleCount + 10];
			positionBasedDeformProcessorForTV->maxLabel = max(positionBasedDeformProcessorForTV->maxLabel, label);
		}
	}
	positionBasedDeformProcessorForTV->saveOriginalCopyOfMeshes();

	polyMesh = positionBasedDeformProcessorForTV->polyMeshes[0];
	//std::cout << "maxLabel: " << maxLabel << std::endl; 

	//read wall
	polyMeshWall = std::make_shared<PolyMesh>();
	VTPReader reader;
	reader.readFile((subfolder + "/reduced-wall.vtp").c_str(), polyMeshWall.get());
	//polyMeshWall->setVertexCoordsOri(); //not needed when vertex coords need not to change
	polyMeshWall->doShift(shift); //note the order of shift, and processing assistParticle and cleanedChannel volume



	////////////////matrix manager
	float3 posMin, posMax;
	polyMesh->GetPosRange(posMin, posMax);
	std::cout << "posMin: " << posMin.x << " " << posMin.y << " " << posMin.z << std::endl;
	std::cout << "posMax: " << posMax.x << " " << posMax.y << " " << posMax.z << std::endl;
	matrixMgr = std::make_shared<GLMatrixManager>(posMin, posMax);
	matrixMgr->setDefaultForImmersiveMode();		
	matrixMgrExocentric = std::make_shared<GLMatrixManager>(posMin, posMax);

	matrixMgr->moveEyeInLocalByModeMat(make_float3(matrixMgr->getEyeInLocal().x, -20, matrixMgr->getEyeInLocal().z));

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
		positionBasedDeformProcessor = std::make_shared<PositionBasedDeformProcessor>(polyMesh->particle, matrixMgr);
		positionBasedDeformProcessor->disThr = 4.1;
		positionBasedDeformProcessor->minPos = -shift;
		positionBasedDeformProcessor->maxPos = make_float3(dims.x, dims.y, dims.z);



		openGL->AddProcessor("1ForTV", positionBasedDeformProcessorForTV.get());
		openGL->AddProcessor("2positionBasedDeformProcessor", positionBasedDeformProcessor.get());

		positionBasedDeformProcessor->deformationScale = 10;
		positionBasedDeformProcessor->deformationScaleVertical = 14;
	}


	//////////////////////////////// Renderable ////////////////////////////////	


	deformFrameRenderable = std::make_shared<DeformFrameRenderable>(matrixMgr, positionBasedDeformProcessor);
	openGL->AddRenderable("0deform", deformFrameRenderable.get()); 


	if (channelSkelViewReady){
	}

	matrixMgrRenderable = std::make_shared<MatrixMgrRenderable>(matrixMgr);
	openGL->AddRenderable("3matrix", matrixMgrRenderable.get()); 

	polyRenderable = std::make_shared<PolyRenderable>(polyMesh);
	openGL->AddRenderable("1poly", polyRenderable.get());
	polyRenderable->setCenterBasedRendering();
	if (channelSkelViewReady){
		polyRenderable->positionBasedDeformProcessor = positionBasedDeformProcessor;
	}


	polyWallRenderable = std::make_shared<PolyRenderable>(polyMeshWall);
	openGL->AddRenderable("5polyWall", polyWallRenderable.get());
	polyMeshWall->opacity = 0.5;

	//////////////////////////////// Interactor ////////////////////////////////
	immersiveInteractor = std::make_shared<ImmersiveInteractor>();
	immersiveInteractor->setMatrixMgr(matrixMgr);
	regularInteractor = std::make_shared<RegularInteractor>();
	regularInteractor->setMatrixMgr(matrixMgrExocentric);
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
		isDeformEnabled->setChecked(positionBasedDeformProcessor->isActive);
		controlLayout->addWidget(isDeformEnabled);
		connect(isDeformEnabled, SIGNAL(clicked(bool)), this, SLOT(isDeformEnabledClicked(bool)));

		QCheckBox* isDeformColoringEnabled = new QCheckBox("Color Deformed Part", this);
		isDeformColoringEnabled->setChecked(positionBasedDeformProcessor->isColoringDeformedPart);
		controlLayout->addWidget(isDeformColoringEnabled);
		connect(isDeformColoringEnabled, SIGNAL(clicked(bool)), this, SLOT(isDeformColoringEnabledClicked(bool)));
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


	QPushButton *startTVBtn = new QPushButton("Start Time Variant");
	controlLayout->addWidget(startTVBtn);
	connect(startTVBtn, SIGNAL(clicked()), this, SLOT(startTVBtnClicked()));


	QPushButton *backToFirstTimestepBtn = new QPushButton("Back To the First Time Step");
	controlLayout->addWidget(backToFirstTimestepBtn);
	connect(backToFirstTimestepBtn, SIGNAL(clicked()), this, SLOT(backToFirstTimestepBtnClicked()));


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




	polyRenderable->positionBasedDeformProcessorForTV = positionBasedDeformProcessorForTV;

	positionBasedDeformProcessorForTV->polyMesh = polyMesh;
	positionBasedDeformProcessorForTV->isActive = false;

	positionBasedDeformProcessorForTV->positionBasedDeformProcessor = positionBasedDeformProcessor;
	//positionBasedDeformProcessorForTV->polyRenderable = polyRenderable;

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
	}
}

void Window::isDeformColoringEnabledClicked(bool b)
{
	if (b){
		positionBasedDeformProcessor->isColoringDeformedPart = true;
	}
	else{
		positionBasedDeformProcessor->isColoringDeformedPart = false;
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
		openGL->matrixMgr = matrixMgrExocentric;
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

void Window::startTVBtnClicked()
{
	positionBasedDeformProcessorForTV->turnActive();
}


void Window::backToFirstTimestepBtnClicked()
{
	positionBasedDeformProcessorForTV->resetPolyMeshes();
	
	polyMesh = positionBasedDeformProcessorForTV->polyMeshes[0];
	positionBasedDeformProcessorForTV->polyMesh = polyMesh;
	polyRenderable->polyMesh = polyMesh;
	polyRenderable->dataChange();

	positionBasedDeformProcessor->updateParticleData(polyMesh->particle);
}	
