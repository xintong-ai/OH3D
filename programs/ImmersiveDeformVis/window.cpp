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
	DataType volDataType = RawVolumeReader::dtUint16;

	Volume::rawFileInfo(dataPath, dims, spacing, rcp, subfolder);
	RawVolumeReader::rawFileReadingInfo(dataPath, volDataType, labelFromFile);
	
	rcpForChannelSkel = std::make_shared<RayCastingParameters>(1.8, 1.0, 1.5, 1.0, 0.3, 2.6, 1024, 0.25f, 1.0, false);

	//dims = make_int3(256, 256, 64);
	//dims = make_int3(301, 324, 56);
	//dims = make_int3(64, 64, 64);
	////dims = make_int3(160, 224, 64);
	//spacing = make_float3(1, 1, 1);
	//rcp = std::make_shared<RayCastingParameters>(1.1, 0.0, 0.0, 0.29, 0.0, 1.0, 512, 0.25f/2, 1.0, false); //for 181
	//subfolder = "beetle";
	//rcp->use2DInteg = false;


	//std::shared_ptr<RayCastingParameters> rcpMini = std::make_shared<RayCastingParameters>(1.8, 1.0, 1.5, 0.5, 0.11, 0.6, 512, 0.25f, 1.0, false);
	std::shared_ptr<RayCastingParameters> rcpMini = rcp;// = std::make_shared<RayCastingParameters>(1.8, 1.0, 1.5, 1.0, 0.3, 0.6, 512, 0.25f, 1.0, false); //for 181

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
		reader = std::make_shared<RawVolumeReader>(dataPath.c_str(), dims, volDataType);	
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

	bool channelSkelViewReady = true;
	if (channelSkelViewReady){
		channelVolume = std::make_shared<Volume>(true);
		std::shared_ptr<RawVolumeReader> reader2 = std::make_shared<RawVolumeReader>((subfolder + "/cleanedChannel.raw").c_str(), dims, RawVolumeReader::dtFloat32);
		reader2->OutputToVolumeByNormalizedValue(channelVolume);
		channelVolume->spacing = spacing;
		channelVolume->initVolumeCuda();
		reader2.reset();

		skelVolume = std::make_shared<Volume>();
		std::shared_ptr<RawVolumeReader> reader4 = std::make_shared<RawVolumeReader>((subfolder + "/skel.raw").c_str(), dims, RawVolumeReader::dtFloat32);
		reader4->OutputToVolumeByNormalizedValue(skelVolume);
		skelVolume->spacing = spacing;
		skelVolume->initVolumeCuda();
		reader4.reset();
	}

	int maxLabel = 1;
	if(labelFromFile){
		unsigned char* labelVolRes = new unsigned char[dims.x*dims.y*dims.z];
		FILE * fp = fopen(dataMgr->GetConfig("FEATURE_PATH").c_str(), "rb");
		fread(labelVolRes, sizeof(unsigned char), dims.x*dims.y*dims.z, fp);
		fclose(fp);
		unsigned short *temp = new unsigned short[dims.x*dims.y*dims.z];
		for (int i = 0; i < dims.x*dims.y*dims.z; i++){
			//specific processing only for Baseline data
			if (labelVolRes[i] >2)
				temp[i] = 2;
			else if (labelVolRes[i] >1)
				temp[i] = 1;
			else 
				temp[i] = 0;
		}
		maxLabel = 2;

		//for (int i = 0; i < dims.x*dims.y*dims.z; i++){
		//	//specific processing only for Baseline data
		//	if (labelVolRes[i] >2)
		//		temp[i] = 1;
		//	else
		//		temp[i] = 0;
		//}
		//maxLabel = 1;

		labelVolCUDA = std::make_shared<VolumeCUDA>();
		labelVolCUDA->VolumeCUDA_init(dims, temp, 0, 1); //currently if from file, do not allow change

		delete[] labelVolRes;
		delete[] temp;
	}
	else{
		labelVolCUDA = std::make_shared<VolumeCUDA>();
		labelVolCUDA->VolumeCUDA_init(dims, (unsigned short *)0, 1, 1);
		labelVolLocal = new unsigned short[dims.x*dims.y*dims.z];
		memset(labelVolLocal, 0, sizeof(unsigned short)*dims.x*dims.y*dims.z);
	}
	

	////////////////matrix manager
	float3 posMin, posMax;
	inputVolume->GetPosRange(posMin, posMax);
	matrixMgr = std::make_shared<GLMatrixManager>(posMin, posMax);
	matrixMgr->setDefaultForImmersiveMode();		
	if (std::string(dataPath).find("engine") != std::string::npos){
		matrixMgr->moveEyeInLocalByModeMat(make_float3(70, -20, 60));
	}
	else if (std::string(dataPath).find("Tomato") != std::string::npos){
		matrixMgr->moveEyeInLocalByModeMat(make_float3(182, 78, 33)*spacing);
	}
	else if (std::string(dataPath).find("181") != std::string::npos){
		matrixMgr->moveEyeInLocalByModeMat(make_float3(64, 109, 107)*spacing);
	}
	else if (std::string(dataPath).find("Baseline") != std::string::npos){
		matrixMgr->moveEyeInLocalByModeMat(make_float3(145, 114, 22)*spacing);
	}
	matrixMgrMini = std::make_shared<GLMatrixManager>(posMin, posMax);


	//////////////ScreenMarker, ViewpointEvaluator
	std::shared_ptr<ScreenMarker> sm = std::make_shared<ScreenMarker>();
	ve = std::make_shared<ViewpointEvaluator>(rcp, inputVolume);
	//ve->initDownSampledResultVolume(make_int3(40, 40, 40));
	ve->dataFolder = subfolder;
	ve->setSpherePoints();
	ve->setLabel(labelVolCUDA);
	ve->maxLabel = maxLabel;

	if (channelSkelViewReady){
		std::shared_ptr<BinaryTuplesReader> reader3 = std::make_shared<BinaryTuplesReader>((subfolder + "/views.mytup").c_str());
		reader3->OutputToParticleDataArrays(ve->skelViews);
		reader3.reset();
	}


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
	volumeRenderable = std::make_shared<VolumeRenderableImmerCUDA>(inputVolume, labelVolCUDA, positionBasedDeformProcessor);
	volumeRenderable->rcp = rcp;
	openGL->AddRenderable("2volume", volumeRenderable.get());
	volumeRenderable->setScreenMarker(sm);

	//ve->createOneParticleFormOfViewSamples();
	//ve->allViewSamples->initForRendering(10, 1);
	//glyphRenderable = std::make_shared<SphereRenderable>(ve->allViewSamples);
	//openGL->AddRenderable("2glyphOfViews", glyphRenderable.get());

	//deformFrameRenderable = std::make_shared<DeformFrameRenderable>(matrixMgr, positionBasedDeformProcessor);
	//openGL->AddRenderable("0deform", deformFrameRenderable.get()); 
	//volumeRenderable->setBlending(true); //only when needed when want the deformFrameRenderable

	//matrixMgrRenderable = std::make_shared<MatrixMgrRenderable>(matrixMgr);
	//openGL->AddRenderable("3matrixMgr", matrixMgrRenderable.get()); 

	if (channelSkelViewReady){
		infoGuideRenderable = std::make_shared<InfoGuideRenderable>(ve, matrixMgr);
		openGL->AddRenderable("4infoGuide", infoGuideRenderable.get());
	}


	//////////////////////////////// Interactor ////////////////////////////////
	immersiveInteractor = std::make_shared<ImmersiveInteractor>();
	immersiveInteractor->setMatrixMgr(matrixMgr);
	regularInteractor = std::make_shared<RegularInteractor>();
	regularInteractor->setMatrixMgr(matrixMgrMini);
	regularInteractor->isActive = false;
	immersiveInteractor->isActive = true;


	openGL->AddInteractor("1modelImmer", immersiveInteractor.get());
	openGL->AddInteractor("2modelReg", regularInteractor.get());
	
	sbInteractor = std::make_shared<ScreenBrushInteractor>();
	sbInteractor->setScreenMarker(sm);
	openGL->AddInteractor("3screenMarker", sbInteractor.get());
	

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
	
	QPushButton *alwaysLocalGuideBtn = new QPushButton("Always Compute Local Guide");
	controlLayout->addWidget(alwaysLocalGuideBtn);
	connect(alwaysLocalGuideBtn, SIGNAL(clicked()), this, SLOT(alwaysLocalGuideBtnClicked()));


	///////////////ray casting settings
	QLabel *transFuncP1SliderLabelLit = new QLabel("Transfer Function Higher Cut Off");
	//controlLayout->addWidget(transFuncP1SliderLabelLit);
	QSlider *transFuncP1LabelSlider = new QSlider(Qt::Horizontal);
	transFuncP1LabelSlider->setRange(0, 100);
	transFuncP1LabelSlider->setValue(rcp->transFuncP1 * 100);
	connect(transFuncP1LabelSlider, SIGNAL(valueChanged(int)), this, SLOT(transFuncP1LabelSliderValueChanged(int)));
	transFuncP1Label = new QLabel(QString::number(rcp->transFuncP1));
	QHBoxLayout *transFuncP1Layout = new QHBoxLayout;
	transFuncP1Layout->addWidget(transFuncP1LabelSlider);
	transFuncP1Layout->addWidget(transFuncP1Label);
	//controlLayout->addLayout(transFuncP1Layout);

	QLabel *transFuncP2SliderLabelLit = new QLabel("Transfer Function Lower Cut Off");
	//controlLayout->addWidget(transFuncP2SliderLabelLit);
	QSlider *transFuncP2LabelSlider = new QSlider(Qt::Horizontal);
	transFuncP2LabelSlider->setRange(0, 100);
	transFuncP2LabelSlider->setValue(rcp->transFuncP2 * 100);
	connect(transFuncP2LabelSlider, SIGNAL(valueChanged(int)), this, SLOT(transFuncP2LabelSliderValueChanged(int)));
	transFuncP2Label = new QLabel(QString::number(rcp->transFuncP2));
	QHBoxLayout *transFuncP2Layout = new QHBoxLayout;
	transFuncP2Layout->addWidget(transFuncP2LabelSlider);
	transFuncP2Layout->addWidget(transFuncP2Label);
	//controlLayout->addLayout(transFuncP2Layout);

	QLabel *brLabelLit = new QLabel("Brightness of the volume: ");
	//controlLayout->addWidget(brLabelLit);
	QSlider* brSlider = new QSlider(Qt::Horizontal);
	brSlider->setRange(0, 40);
	brSlider->setValue(rcp->brightness * 20);
	connect(brSlider, SIGNAL(valueChanged(int)), this, SLOT(brSliderValueChanged(int)));
	brLabel = new QLabel(QString::number(rcp->brightness));
	QHBoxLayout *brLayout = new QHBoxLayout;
	brLayout->addWidget(brSlider);
	brLayout->addWidget(brLabel);
	//controlLayout->addLayout(brLayout);

	QLabel *dsLabelLit = new QLabel("Density of the volume: ");
	//controlLayout->addWidget(dsLabelLit);
	QSlider* dsSlider = new QSlider(Qt::Horizontal);
	dsSlider->setRange(0, 100);
	dsSlider->setValue(rcp->density * 20);
	connect(dsSlider, SIGNAL(valueChanged(int)), this, SLOT(dsSliderValueChanged(int)));
	dsLabel = new QLabel(QString::number(rcp->density));
	QHBoxLayout *dsLayout = new QHBoxLayout;
	dsLayout->addWidget(dsSlider);
	dsLayout->addWidget(dsLabel);
	//controlLayout->addLayout(dsLayout);

	QLabel *laSliderLabelLit = new QLabel("Coefficient for Ambient Lighting: ");
	//controlLayout->addWidget(laSliderLabelLit);
	QSlider* laSlider = new QSlider(Qt::Horizontal);
	laSlider->setRange(0, 50);
	laSlider->setValue(rcp->la * 10);
	connect(laSlider, SIGNAL(valueChanged(int)), this, SLOT(laSliderValueChanged(int)));
	laLabel = new QLabel(QString::number(rcp->la));
	QHBoxLayout *laLayout = new QHBoxLayout;
	laLayout->addWidget(laSlider);
	laLayout->addWidget(laLabel);
	//controlLayout->addLayout(laLayout);

	QLabel *ldSliderLabelLit = new QLabel("Coefficient for Diffusial Lighting: ");
	//controlLayout->addWidget(ldSliderLabelLit);
	QSlider* ldSlider = new QSlider(Qt::Horizontal);
	ldSlider->setRange(0, 50);
	ldSlider->setValue(rcp->ld * 10);
	connect(ldSlider, SIGNAL(valueChanged(int)), this, SLOT(ldSliderValueChanged(int)));
	ldLabel = new QLabel(QString::number(rcp->ld));
	QHBoxLayout *ldLayout = new QHBoxLayout;
	ldLayout->addWidget(ldSlider);
	ldLayout->addWidget(ldLabel);
	//controlLayout->addLayout(ldLayout);

	QLabel *lsSliderLabelLit = new QLabel("Coefficient for Specular Lighting: ");
	//controlLayout->addWidget(lsSliderLabelLit);
	QSlider* lsSlider = new QSlider(Qt::Horizontal);
	lsSlider->setRange(0, 50);
	lsSlider->setValue(rcp->ls * 10);
	connect(lsSlider, SIGNAL(valueChanged(int)), this, SLOT(lsSliderValueChanged(int)));
	lsLabel = new QLabel(QString::number(rcp->ls));
	QHBoxLayout *lsLayout = new QHBoxLayout;
	lsLayout->addWidget(lsSlider);
	lsLayout->addWidget(lsLabel);
	//controlLayout->addLayout(lsLayout);

	QLabel *transFuncP1SecondSliderLabelLit = new QLabel("Second Higher Cut Off");
	QSlider *transFuncP1SecondLabelSlider = new QSlider(Qt::Horizontal);
	transFuncP1SecondLabelSlider->setRange(0, 100);
	transFuncP1SecondLabelSlider->setValue(rcp->secondCutOffHigh * 100);
	connect(transFuncP1SecondLabelSlider, SIGNAL(valueChanged(int)), this, SLOT(transFuncP1SecondLabelSliderValueChanged(int)));
	transFuncP1SecondLabel = new QLabel(QString::number(rcp->secondCutOffHigh));
	QHBoxLayout *transFuncP1SecondLayout = new QHBoxLayout;
	transFuncP1SecondLayout->addWidget(transFuncP1SecondLabelSlider);
	transFuncP1SecondLayout->addWidget(transFuncP1SecondLabel);

	QLabel *transFuncP2SecondSliderLabelLit = new QLabel("Transfer Function Lower Cut Off");
	QSlider *transFuncP2SecondLabelSlider = new QSlider(Qt::Horizontal);
	transFuncP2SecondLabelSlider->setRange(0, 100);
	transFuncP2SecondLabelSlider->setValue(rcp->secondCutOffLow* 100);
	connect(transFuncP2SecondLabelSlider, SIGNAL(valueChanged(int)), this, SLOT(transFuncP2SecondLabelSliderValueChanged(int)));
	transFuncP2SecondLabel = new QLabel(QString::number(rcp->secondCutOffLow));
	QHBoxLayout *transFuncP2SecondLayout = new QHBoxLayout;
	transFuncP2SecondLayout->addWidget(transFuncP2SecondLabelSlider);
	transFuncP2SecondLayout->addWidget(transFuncP2SecondLabel);

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
	rcLayout->addWidget(transFuncP1SecondSliderLabelLit);
	rcLayout->addLayout(transFuncP1SecondLayout);
	rcLayout->addWidget(transFuncP2SecondSliderLabelLit);
	rcLayout->addLayout(transFuncP2SecondLayout);

	controlLayout->addWidget(rcGroupBox);

	controlLayout->addStretch();

	connect(saveStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotSaveState()));
	connect(loadStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotLoadState()));
	connect(eyePosBtn, SIGNAL(clicked()), this, SLOT(applyEyePos()));



	//////////////////////////miniature
	QVBoxLayout *assistLayout = new QVBoxLayout;
	QLabel *miniatureLabel = new QLabel("miniature");
	//assistLayout->addWidget(miniatureLabel);

	openGLMini = std::make_shared<GLWidget>(matrixMgrMini);

	QSurfaceFormat format2;
	format2.setDepthBufferSize(24);
	format2.setStencilBufferSize(8);
	format2.setVersion(2, 0);
	format2.setProfile(QSurfaceFormat::CoreProfile);
	openGLMini->setFormat(format2);

	//matrixMgrRenderableMini = std::make_shared<MatrixMgrRenderable>(matrixMgr);
	//matrixMgrRenderableMini->renderPart = 2;
	//openGLMini->AddRenderable("3center", matrixMgrRenderableMini.get());
	//ve->createOneParticleFormOfViewSamples(); 
	//ve->allViewSamples->initForRendering(50, 1);
	//glyphRenderable = std::make_shared<SphereRenderable>(ve->allViewSamples);
	//openGLMini->AddRenderable("2glyphRenderable", glyphRenderable.get());
	//volumeRenderableMini = std::make_shared<VolumeRenderableCUDA>(inputVolume);
	//volumeRenderableMini->rcp = rcpMini; 
	//volumeRenderableMini->setBlending(true, 50);
	//openGLMini->AddRenderable("4volume", volumeRenderableMini.get());
	//regularInteractorMini = std::make_shared<RegularInteractor>();
	//regularInteractorMini->setMatrixMgr(matrixMgrMini);
	//openGLMini->AddInteractor("1regular", regularInteractorMini.get());
	//assistLayout->addWidget(openGLMini.get(), 3);



	////////////////////2D slice view
	if (labelVolLocal!=0)
		helper.setData(inputVolume, labelVolLocal);
	else{
		labelVolLocal = new unsigned short[dims.x*dims.y*dims.z]; //should remove this part later
		helper.setData(inputVolume, labelVolLocal);
	}
	GLWidgetQtDrawing *openGL2D = new GLWidgetQtDrawing(&helper, this);
	assistLayout->addWidget(openGL2D, 0);
	QTimer *timer = new QTimer(this);
	connect(timer, &QTimer::timeout, openGL2D, &GLWidgetQtDrawing::animate);
	timer->start(5);


	QLabel *zSliderLabelLit = new QLabel("Z index: ");
	QSlider *zSlider = new QSlider(Qt::Horizontal);
	zSlider->setRange(0, inputVolume->size.z);
	zSlider->setValue(helper.z);
	connect(zSlider, SIGNAL(valueChanged(int)), this, SLOT(zSliderValueChanged(int)));
	QHBoxLayout *zLayout = new QHBoxLayout;
	zLayout->addWidget(zSliderLabelLit);
	zLayout->addWidget(zSlider);
	assistLayout->addLayout(zLayout);

	QPushButton *redrawBtn = new QPushButton("Redraw the Label");
	assistLayout->addWidget(redrawBtn);
	connect(redrawBtn, SIGNAL(clicked()), this, SLOT(redrawBtnClicked()));

	QPushButton *updateLabelVolBtn = new QPushButton("Find optimal for Label");
	assistLayout->addWidget(updateLabelVolBtn);
	connect(updateLabelVolBtn, SIGNAL(clicked()), this, SLOT(updateLabelVolBtnClicked()));

	QPushButton *findGeneralOptimalBtn = new QPushButton("Find general optimal");
	assistLayout->addWidget(findGeneralOptimalBtn);
	connect(findGeneralOptimalBtn, SIGNAL(clicked()), this, SLOT(findGeneralOptimalBtnClicked()));

	//QCheckBox* isBrushingCb = new QCheckBox("Brush", this);
	//isBrushingCb->setChecked(sbInteractor->isActive);
	//assistLayout->addWidget(isBrushingCb);
	//connect(isBrushingCb, SIGNAL(clicked()), this, SLOT(isBrushingClicked()));

	QPushButton *moveToOptimalBtn = new QPushButton("Move to the Optimal Viewpoint");
	assistLayout->addWidget(moveToOptimalBtn);
	connect(moveToOptimalBtn, SIGNAL(clicked()), this, SLOT(moveToOptimalBtnClicked()));

	//QPushButton *doTourBtn = new QPushButton("Do the Animation Tour");
	//assistLayout->addWidget(doTourBtn);
	//connect(doTourBtn, SIGNAL(clicked()), this, SLOT(doTourBtnClicked()));

	QPushButton *turnOffGlobalGuideBtn = new QPushButton("Turn Off GLobal Guide");
	assistLayout->addWidget(turnOffGlobalGuideBtn);
	connect(turnOffGlobalGuideBtn, SIGNAL(clicked()), this, SLOT(turnOffGlobalGuideBtnClicked()));

	mainLayout->addLayout(assistLayout, 1);
	openGL->setFixedSize(576, 648); //in accordance to 960x1080 of OSVR
//openGLMini->setFixedSize(300, 300);

	mainLayout->addWidget(openGL.get(), 5);
	mainLayout->addLayout(controlLayout, 1);
	setLayout(mainLayout);


#ifdef USE_OSVR
	vrWidget = std::make_shared<VRWidget>(matrixMgr);
	vrWidget->setWindowFlags(Qt::Window);
	vrVolumeRenderable = std::make_shared<VRVolumeRenderableCUDA>(inputVolume);
	vrVolumeRenderable->sm = sm;

	vrWidget->AddRenderable("1volume", vrVolumeRenderable.get());
	if (channelSkelViewReady){
		immersiveInteractor->noMoveMode = true;
		vrWidget->AddRenderable("2info", infoGuideRenderable.get());
	}

	openGL->SetVRWidget(vrWidget.get());
	vrVolumeRenderable->rcp = rcp;
#endif

}






Window::~Window()
{
	if (labelVolLocal)
		delete[]labelVolLocal;
}

void Window::init()
{
#ifdef USE_OSVR
	vrWidget->show();
#endif
}

void Window::SlotSaveState()
{
	matrixMgr->SaveState("state.txt");
}

void Window::SlotLoadState()
{
	matrixMgr->LoadState("state.txt");
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
	//ve->compute_UniformSampling(VPMethod::JS06Sphere);
	matrixMgr->moveEyeInLocalByModeMat(make_float3(ve->optimalEyeInLocal.x, ve->optimalEyeInLocal.y, ve->optimalEyeInLocal.z));
	//ve->saveResultVol("labelEntro.raw");
}

void Window::SlotOriVolumeRb(bool b)
{
	if (b){
		volumeRenderable->setVolume(inputVolume);
		volumeRenderable->rcp = rcp;
	}
}

void Window::SlotChannelVolumeRb(bool b)
{
	if (b)
	{
		if (channelVolume){
			volumeRenderable->setVolume(channelVolume);
			volumeRenderable->rcp = rcpForChannelSkel;
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
			volumeRenderable->rcp = rcpForChannelSkel;
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
	if (!labelFromFile){
		labelVolCUDA->VolumeCUDA_contentUpdate(labelVolLocal, 1, 1);
		std::cout << std::endl << "The lable volume has been updated from drawing" << std::endl << std::endl;
	}

	ve->currentMethod = VPMethod::LabelVisibility;
	ve->compute_SkelSampling(VPMethod::LabelVisibility);
	std::cout << std::endl << "The optimal view point has been computed" << std::endl << "max entropy: " << ve->maxEntropy << std::endl;
	std::cout << "The optimal view point: " << ve->optimalEyeInLocal.x << " " << ve->optimalEyeInLocal.y << " " << ve->optimalEyeInLocal.z << std::endl << "The optimal view point in voxel: " << ve->optimalEyeInLocal.x / spacing.x << " " << ve->optimalEyeInLocal.y / spacing.y << " " << ve->optimalEyeInLocal.z / spacing.z << std::endl;
	infoGuideRenderable->changeWhetherGlobalGuideMode(true);
}

void Window::findGeneralOptimalBtnClicked()
{
	ve->currentMethod = VPMethod::Tao09Detail;
	ve->compute_SkelSampling(VPMethod::Tao09Detail);
	std::cout << std::endl << "The optimal view point has been computed" << std::endl << "max entropy: " << ve->maxEntropy<< std::endl;
	std::cout << "The optimal view point: " << ve->optimalEyeInLocal.x << " " << ve->optimalEyeInLocal.y << " "<< ve->optimalEyeInLocal.z << std::endl << "The optimal view point in voxel: " << ve->optimalEyeInLocal.x / spacing.x << " " << ve->optimalEyeInLocal.y / spacing.y << " " << ve->optimalEyeInLocal.z / spacing.z << std::endl;
	infoGuideRenderable->changeWhetherGlobalGuideMode(true);
}


void Window::turnOffGlobalGuideBtnClicked()
{
	infoGuideRenderable->changeWhetherGlobalGuideMode(false);
}

void Window::redrawBtnClicked()
{
	if (labelVolLocal){
		memset(labelVolLocal, 0, sizeof(unsigned short)*dims.x*dims.y*dims.z);
		labelVolCUDA->VolumeCUDA_contentUpdate(labelVolLocal, 1, 1);
	}
	helper.valSet = false;
}

void Window::doTourBtnClicked()
{
	animationByMatrixProcessor->startAnimation();
}

void Window::saveScreenBtnClicked()
{
	openGL->saveCurrentImage();
}
void Window::alwaysLocalGuideBtnClicked()
{
	infoGuideRenderable->isAlwaysLocalGuide = true;
}
