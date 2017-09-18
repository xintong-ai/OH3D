#include <window.h>
#include <iostream>

#include "myDefineRayCasting.h"
#include "GLWidget.h"
#include "Volume.h"
#include "RawVolumeReader.h"
#include "DataMgr.h"
#include "VecReader.h"
#include "GLMatrixManager.h"
#include "VolumeRenderableCUDA.h"
#include "VolumeRenderableImmerCUDA.h"
#include "mouse/RegularInteractor.h"
#include "mouse/ImmersiveInteractor.h"
#include "mouse/ScreenBrushInteractor.h"
#include "GLWidgetQtDrawing.h"
#include "AnimationByMatrixProcessor.h"
#include "Particle.h"

#include "PositionBasedDeformProcessor.h"
#include "MatrixMgrRenderable.h"
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

#include <thrust/device_vector.h>
#include "VTIReader.h"

//

void positionBasedDeformerConfigure(std::string dataPath, float & densityThr, int & checkRadius)
{
	if (std::string(dataPath).find("181") != std::string::npos){
		densityThr = 0.01;
		checkRadius = 1;
	}
	else if (std::string(dataPath).find("moortgat") != std::string::npos){
		densityThr = 0.0007;
		checkRadius = 1;
	}
	else{
		std::cout << "volume data name not recognized" << std::endl;
		exit(0);
	}
};

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
	bool hasLabelFromFile;
	float densityThr;
	int checkRadius;

	Volume::rawFileInfo(dataPath, dims, spacing, rcp, subfolder);
	RawVolumeReader::rawFileReadingInfo(dataPath, volDataType, hasLabelFromFile);
	positionBasedDeformerConfigure(dataPath, densityThr, checkRadius);

	//rcp->tstep = 1;  //this is actually a mistake in the VIS submission version, since rcp will be changed in the construction function of ViewpointEvaluator, which sets the tstep as 1.0
	//use larger step size in testing phases

	//rcp = std::make_shared<RayCastingParameters>(0.4, 0.9, 1.2, 0.3, 0.18, 0.1, 512, 0.125f, 1.0, true); //for moortgat data, non cubic spline interpolation. Note this data is not suitable to use cubic spline interpolation. also the Shininess for specular light for this data is 5


	inputVolume = std::make_shared<Volume>(true);
	if (std::string(dataPath).find(".vti") != std::string::npos){
		VTIReader vtiReader(dataPath.c_str(), inputVolume);
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
	else if (std::string(dataPath).find("colon") != std::string::npos){
		matrixMgr->moveEyeInLocalByModeMat(make_float3(177, 152, 32)*spacing);
	}
	else if (std::string(dataPath).find("moortgat") != std::string::npos){
		matrixMgr->moveEyeInLocalByModeMat(make_float3(matrixMgr->getEyeInLocal().x - 10, -20, matrixMgr->getEyeInLocal().z));

	}
	matrixMgrExocentric = std::make_shared<GLMatrixManager>(posMin, posMax);


	/********GL widget******/
	openGL = std::make_shared<GLWidget>(matrixMgr);
	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setStencilBufferSize(8);
	format.setVersion(2, 0);
	format.setProfile(QSurfaceFormat::CoreProfile);
	openGL->setFormat(format); // must be called before the widget or its parent window gets shown


	//////////////////////////////// Processor ////////////////////////////////		
	positionBasedDeformProcessor = std::make_shared<PositionBasedDeformProcessor>(inputVolume, matrixMgr);
	openGL->AddProcessor("1positionBasedDeformProcessor", positionBasedDeformProcessor.get());
	positionBasedDeformProcessor->rcp = rcp;
	positionBasedDeformProcessor->densityThr = densityThr;
	positionBasedDeformProcessor->checkRadius = checkRadius;
	positionBasedDeformProcessor->setDeformationScale(6);

	positionBasedDeformProcessor->radius = 7;
	//animationByMatrixProcessor = std::make_shared<AnimationByMatrixProcessor>(matrixMgr);
	//animationByMatrixProcessor->setViews(views);
	//openGL->AddProcessor("animationByMatrixProcessor", animationByMatrixProcessor.get());


	//////////////////////////////// Renderable ////////////////////////////////	
	volumeRenderable = std::make_shared<VolumeRenderableImmerCUDA>(inputVolume, positionBasedDeformProcessor);
	volumeRenderable->rcp = rcp;
	volumeRenderable->updateColorTable(); //must be called once beforehand

	openGL->AddRenderable("2volume", volumeRenderable.get());
	//volumeRenderable->setPreIntegrate(true);


	//deformFrameRenderable = std::make_shared<DeformFrameRenderable>(matrixMgr, positionBasedDeformProcessor);
	//openGL->AddRenderable("0deform", deformFrameRenderable.get());
	//volumeRenderable->setBlending(true); //only when needed when want the deformFrameRenderable
	//!!!!! once turned on blending, some render technique is not implemented yet !!!!!!

	//matrixMgrRenderable = std::make_shared<MatrixMgrRenderable>(matrixMgr);
	//openGL->AddRenderable("3matrixMgr", matrixMgrRenderable.get()); 


	//////////////////////////////// Interactor ////////////////////////////////
	immersiveInteractor = std::make_shared<ImmersiveInteractor>();
	immersiveInteractor->setMatrixMgr(matrixMgr);
	regularInteractor = std::make_shared<RegularInteractor>();
	regularInteractor->setMatrixMgr(matrixMgrExocentric);
	regularInteractor->isActive = false;
	immersiveInteractor->isActive = true;


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
	std::cout << posMin.x << " " << posMin.y << " " << posMin.z << std::endl;
	std::cout << posMax.x << " " << posMax.y << " " << posMax.z << std::endl;
	controlLayout->addWidget(saveStateBtn.get());
	controlLayout->addWidget(loadStateBtn.get());

	QCheckBox* isDeformEnabled = new QCheckBox("Enable Deform", this);
	isDeformEnabled->setChecked(positionBasedDeformProcessor->isActive);
	controlLayout->addWidget(isDeformEnabled);
	connect(isDeformEnabled, SIGNAL(clicked(bool)), this, SLOT(isDeformEnabledClicked(bool)));

	QCheckBox* isForceDeformEnabled = new QCheckBox("Force Deform", this);
	isForceDeformEnabled->setChecked(positionBasedDeformProcessor->isForceDeform);
	controlLayout->addWidget(isForceDeformEnabled);
	connect(isForceDeformEnabled, SIGNAL(clicked(bool)), this, SLOT(isForceDeformEnabledClicked(bool)));

	QCheckBox* isDeformColoringEnabled = new QCheckBox("Color Deformed Part (when preintegrate)", this);
	isDeformColoringEnabled->setChecked(positionBasedDeformProcessor->isColoringDeformedPart);
	controlLayout->addWidget(isDeformColoringEnabled);
	connect(isDeformColoringEnabled, SIGNAL(clicked(bool)), this, SLOT(isDeformColoringEnabledClicked(bool)));




	QGroupBox *groupBoxShapeModes = new QGroupBox(tr("deformation shape modes"));
	QHBoxLayout *shapeModeLayout = new QHBoxLayout;
	circularRb = std::make_shared<QRadioButton>(tr("&Circular"));
	cuboidRb = std::make_shared<QRadioButton>(tr("&Cuboid"));
	physicallyRb = std::make_shared<QRadioButton>(tr("&Physically"));

	cuboidRb->setChecked(true);
	shapeModeLayout->addWidget(circularRb.get());
	shapeModeLayout->addWidget(cuboidRb.get());
	shapeModeLayout->addWidget(physicallyRb.get());
	groupBoxShapeModes->setLayout(shapeModeLayout);
	controlLayout->addWidget(groupBoxShapeModes);
	connect(circularRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotCircularRb(bool)));
	connect(cuboidRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotCuboidRb(bool)));
	connect(physicallyRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotPhysicallyRb(bool)));


	QGroupBox *groupBoxORModes = new QGroupBox(tr("occlusion removal modes"));
	QHBoxLayout *orModeLayout = new QHBoxLayout;
	originalRb = std::make_shared<QRadioButton>(tr("&original"));
	deformRb = std::make_shared<QRadioButton>(tr("&deform"));
	clipRb = std::make_shared<QRadioButton>(tr("&clip"));
	transpRb = std::make_shared<QRadioButton>(tr("&transparent"));

	deformRb->setChecked(true);
	orModeLayout->addWidget(originalRb.get());
	orModeLayout->addWidget(deformRb.get());
	orModeLayout->addWidget(clipRb.get());
	orModeLayout->addWidget(transpRb.get());
	groupBoxORModes->setLayout(orModeLayout);
	controlLayout->addWidget(groupBoxORModes);
	connect(originalRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotOriginalRb(bool)));
	connect(deformRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotDeformRb(bool)));
	connect(clipRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotClipRb(bool)));
	connect(transpRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotTranspRb(bool)));




	QPushButton *saveScreenBtn = new QPushButton("Save the current screen");
	controlLayout->addWidget(saveScreenBtn);
	connect(saveScreenBtn, SIGNAL(clicked()), this, SLOT(saveScreenBtnClicked()));



	QCheckBox* usePreIntCB = new QCheckBox("Use Pre-Integrate", this);
	usePreIntCB->setChecked(volumeRenderable->usePreInt);
	controlLayout->addWidget(usePreIntCB);
	connect(usePreIntCB, SIGNAL(clicked(bool)), this, SLOT(usePreIntCBClicked(bool)));

	QCheckBox* useSplineInterpolationCB = new QCheckBox("Use cubic spline interpolation (when preintegrate)", this);
	useSplineInterpolationCB->setChecked(volumeRenderable->useSplineInterpolation);
	controlLayout->addWidget(useSplineInterpolationCB);
	connect(useSplineInterpolationCB, SIGNAL(clicked(bool)), this, SLOT(useSplineInterpolationCBClicked(bool)));

	QGroupBox *rcGroupBox = new QGroupBox(tr("Ray Casting setting"));
	addRayCastingInterfaces(rcGroupBox);
	controlLayout->addWidget(rcGroupBox);

	controlLayout->addStretch();

	connect(saveStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotSaveState()));
	connect(loadStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotLoadState()));



	//////////////////////////miniature
	QVBoxLayout *assistLayout = new QVBoxLayout;
	QLabel *miniatureLabel = new QLabel("miniature");
	//assistLayout->addWidget(miniatureLabel);

	openGLMini = std::make_shared<GLWidget>(matrixMgrExocentric);

	QSurfaceFormat format2;
	format2.setDepthBufferSize(24);
	format2.setStencilBufferSize(8);
	format2.setVersion(2, 0);
	format2.setProfile(QSurfaceFormat::CoreProfile);
	openGLMini->setFormat(format2);

	////////////////////2D slice view
	helper.setData(inputVolume, 0);

	GLWidgetQtDrawing *openGL2D = new GLWidgetQtDrawing(&helper, this);
	assistLayout->addWidget(openGL2D, 0);
	QTimer *timer = new QTimer(this);
	connect(timer, &QTimer::timeout, openGL2D, &GLWidgetQtDrawing::animate);
	timer->start(5);

	QLabel *zSliderLabelLit = new QLabel("Z index: ");
	QSlider *zSlider = new QSlider(Qt::Horizontal);
	zSlider->setRange(0, inputVolume->size.z - 1);
	zSlider->setValue(helper.z);
	connect(zSlider, SIGNAL(valueChanged(int)), this, SLOT(zSliderValueChanged(int)));
	QHBoxLayout *zLayout = new QHBoxLayout;
	zLayout->addWidget(zSliderLabelLit);
	zLayout->addWidget(zSlider);
	assistLayout->addLayout(zLayout);

	QPushButton *doTourBtn = new QPushButton("Do the Animation Tour");
	assistLayout->addWidget(doTourBtn);
	connect(doTourBtn, SIGNAL(clicked()), this, SLOT(doTourBtnClicked()));

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
	assistLayout->addWidget(eyePosGroup);
	connect(eyePosBtn, SIGNAL(clicked()), this, SLOT(applyEyePos()));


	QGroupBox *groupBox = new QGroupBox(tr("volume selection"));
	QHBoxLayout *deformModeLayout = new QHBoxLayout;
	oriVolumeRb = std::make_shared<QRadioButton>(tr("&original"));
	surfaceRb = std::make_shared<QRadioButton>(tr("&surface"));
	oriVolumeRb->setChecked(true);
	deformModeLayout->addWidget(oriVolumeRb.get());
	deformModeLayout->addWidget(surfaceRb.get());
	groupBox->setLayout(deformModeLayout);
	assistLayout->addWidget(groupBox);
	connect(oriVolumeRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotOriVolumeRb(bool)));
	connect(surfaceRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotSurfaceRb(bool)));

	oriVolumeRb->setDisabled(true);
	surfaceRb->setDisabled(true);


	QGroupBox *groupBox2 = new QGroupBox(tr("volume selection"));
	QHBoxLayout *deformModeLayout2 = new QHBoxLayout;
	immerRb = std::make_shared<QRadioButton>(tr("&immersive mode"));
	nonImmerRb = std::make_shared<QRadioButton>(tr("&non immer"));
	immerRb->setChecked(true);
	deformModeLayout2->addWidget(immerRb.get());
	deformModeLayout2->addWidget(nonImmerRb.get());
	groupBox2->setLayout(deformModeLayout2);
	assistLayout->addWidget(groupBox2);
	connect(immerRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotImmerRb(bool)));
	connect(nonImmerRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotNonImmerRb(bool)));


	mainLayout->addLayout(assistLayout, 1);
	//openGL->setFixedSize(576, 648); //in accordance to 960x1080 of OSVR



	//openGL->setFixedSize(1000, 1000);
	//openGLMini->setFixedSize(300, 300);
	openGL->setFixedSize(600, 600);
	//openGL->setFixedSize(750, 900);

	mainLayout->addWidget(openGL.get(), 5);
	mainLayout->addLayout(controlLayout, 1);
	setLayout(mainLayout);


#ifdef USE_OSVR
	vrWidget = std::make_shared<VRWidget>(matrixMgr);
	vrWidget->setWindowFlags(Qt::Window);
	vrVolumeRenderable = std::make_shared<VRVolumeRenderableCUDA>(inputVolume);

	vrWidget->AddRenderable("1volume", vrVolumeRenderable.get());
	immersiveInteractor->noMoveMode = true;


	openGL->SetVRWidget(vrWidget.get());
	vrVolumeRenderable->rcp = rcp;
#endif

}


void Window::addRayCastingInterfaces(QGroupBox *rcGroupBox)
{
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
	dsSlider->setRange(0, 200);
	dsSlider->setValue(rcp->density * 50);
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
	transFuncP2SecondLabelSlider->setValue(rcp->secondCutOffLow * 100);
	connect(transFuncP2SecondLabelSlider, SIGNAL(valueChanged(int)), this, SLOT(transFuncP2SecondLabelSliderValueChanged(int)));
	transFuncP2SecondLabel = new QLabel(QString::number(rcp->secondCutOffLow));
	QHBoxLayout *transFuncP2SecondLayout = new QHBoxLayout;
	transFuncP2SecondLayout->addWidget(transFuncP2SecondLabelSlider);
	transFuncP2SecondLayout->addWidget(transFuncP2SecondLabel);

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
}



Window::~Window()
{
	if (rcp->d_transferFunc != 0){
		cudaFreeArray(rcp->d_transferFunc);
	}
	rcp->d_transferFunc = 0;
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

void Window::usePreIntCBClicked(bool b)
{
	volumeRenderable->setPreIntegrate(b);
}

void Window::useSplineInterpolationCBClicked(bool b)
{
	volumeRenderable->setSplineInterpolation(b);
}

void Window::transFuncP1LabelSliderValueChanged(int v)
{
	rcp->transFuncP1 = 1.0*v / 100;
	transFuncP1Label->setText(QString::number(1.0*v / 100));
	volumeRenderable->preIntTableNeedUpdate();
}
void Window::transFuncP2LabelSliderValueChanged(int v)
{
	rcp->transFuncP2 = 1.0*v / 100;
	transFuncP2Label->setText(QString::number(1.0*v / 100));
	volumeRenderable->preIntTableNeedUpdate();
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
	rcp->density = v*1.0 / 50.0;
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
	}
}

void Window::isForceDeformEnabledClicked(bool b)
{
	if (b){
		positionBasedDeformProcessor->isForceDeform = true;
	}
	else{
		positionBasedDeformProcessor->isForceDeform = false;
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


void Window::SlotOriginalRb(bool b)
{
	if (b){
		positionBasedDeformProcessor->isActive = false;
		inputVolume->reset();
		volumeRenderable->endClipRendering();
	}
	else{
	}
}

void  Window::SlotDeformRb(bool b)
{
	if (b){
		positionBasedDeformProcessor->isActive = true;
		positionBasedDeformProcessor->deformData = true;
		positionBasedDeformProcessor->reset();
		volumeRenderable->endClipRendering();
	}
	else{
	}
}

void  Window::SlotClipRb(bool b)
{
	if (b){
		positionBasedDeformProcessor->isActive = true;
		positionBasedDeformProcessor->deformData = false;
		inputVolume->reset();
		//volumeRenderable->startClipRendering(volume);
	}
	else{
	}
}
void  Window::SlotTranspRb(bool b)
{
	//to be done later
}


void  Window::SlotCircularRb(bool b)
{
	if (positionBasedDeformProcessor->setShapeModel(SHAPE_MODEL::CIRCLE)){
	}
	else{
		if (positionBasedDeformProcessor->getShapeModel() == SHAPE_MODEL::CUBOID){
			cuboidRb->setChecked(true);
		}
		else if (positionBasedDeformProcessor->getShapeModel() == SHAPE_MODEL::PHYSICALLY){
			physicallyRb->setChecked(true);
		}
	}
}
void  Window::SlotCuboidRb(bool b)
{
	if (positionBasedDeformProcessor->setShapeModel(SHAPE_MODEL::CUBOID)){
	}
	else{
		if (positionBasedDeformProcessor->getShapeModel() == SHAPE_MODEL::CIRCLE){
			circularRb->setChecked(true);
		}
		else if (positionBasedDeformProcessor->getShapeModel() == SHAPE_MODEL::PHYSICALLY){
			physicallyRb->setChecked(true);
		}
	}
}
void  Window::SlotPhysicallyRb(bool b)
{
	//to be done later
}


void Window::SlotOriVolumeRb(bool b)
{
	if (b){
		volumeRenderable->setVolume(inputVolume);
		volumeRenderable->rcp = rcp;
		volumeRenderable->SetVisibility(true);
	}
}



void Window::SlotSurfaceRb(bool b)
{
	if (b)
	{
		std::cout << "polymesh not set!!" << std::endl;
		oriVolumeRb->setChecked(true);
		SlotOriVolumeRb(true);
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

void Window::zSliderValueChanged(int v)
{
	helper.z = v;
}

void Window::doTourBtnClicked()
{
	animationByMatrixProcessor->startAnimation();
}

void Window::saveScreenBtnClicked()
{
	openGL->saveCurrentImage();
}
