#include "window.h"
#include "GLWidget.h"
#include <iostream>

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

#ifdef USE_OSVR
#include "VRWidget.h"
#include "VRVolumeRenderableCUDA.h"
#endif



void computeChannelVolume(std::shared_ptr<Volume> v, std::shared_ptr<Volume> channelV, std::shared_ptr<RayCastingParameters> rcp)
{
	std::cout << "computing channel volume..." << std::endl;

	int3 dataSizes = v->size;

	for (int k = 0; k < dataSizes.z; k++)
	{
		for (int j = 0; j < dataSizes.y; j++)
		{
			for (int i = 0; i < dataSizes.x; i++)
			{
				int ind = k*dataSizes.y * dataSizes.x + j*dataSizes.x + i;
				if (v->values[ind] < rcp->transFuncP2){
					channelV->values[ind] = 1;
				}
				else{
					channelV->values[ind] = 0;
				}
			}
		}
	}
	channelV->initVolumeCuda();
	std::cout << "finish computing channel volume..." << std::endl;
	return;
}


void Window::computeSkel()
{
	std::cout << "computing skeletion volume..." << std::endl;

	const unsigned int numberOfPixels = dims.x * dims.y * dims.z;
	PixelType * localBuffer = new PixelType[numberOfPixels];

	for (int i = 0; i < numberOfPixels; i++){
		if (channelVolume->values[i] > 0.5){
			localBuffer[i] = 1;
		}
		else{
			localBuffer[i] = 0;
		}
	}

	importFilter->SetImportPointer(localBuffer, numberOfPixels, importImageFilterWillOwnTheBuffer);

	thinningFilter->SetInput(importFilter->GetOutput());
	thinningFilter->Update();

	skelVolume->values = thinningFilter->GetOutput()->GetBufferPointer(); //caution! care for segmentation fault

	skelVolume->initVolumeCuda();

	std::cout << "finish computing skeletion volume..." << std::endl;

}

Window::Window()
{
    setWindowTitle(tr("Interactive Glyph Visualization"));

////////////////data
	std::shared_ptr<DataMgr> dataMgr;
	dataMgr = std::make_shared<DataMgr>();
	const std::string dataPath = dataMgr->GetConfig("VOLUME_DATA_PATH");

	std::shared_ptr<RayCastingParameters> rcp = std::make_shared<RayCastingParameters>();

	if (std::string(dataPath).find("MGHT2") != std::string::npos){
		dims = make_int3(320, 320, 256);
		spacing = make_float3(0.7, 0.7, 0.7);
	}
	else if (std::string(dataPath).find("MGHT1") != std::string::npos){
		dims = make_int3(256, 256, 176);
		spacing = make_float3(1.0, 1.0, 1.0);
		rcp = std::make_shared<RayCastingParameters>(1.0, 0.2, 0.7, 0.44, 0.29, 1.25, 512, 0.25f, 1.3, false);
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
	else if (std::string(dataPath).find("brat") != std::string::npos){
		dims = make_int3(160, 216, 176);
		spacing = make_float3(1, 1, 1);
		rcp = std::make_shared<RayCastingParameters>(1.0, 0.2, 0.7, 0.44, 0.25, 1.25, 512, 0.25f, 1.3, false); //for brat
	}
	else if (std::string(dataPath).find("engine") != std::string::npos){
		dims = make_int3(149, 208, 110);
		spacing = make_float3(1, 1, 1);
		rcp = std::make_shared<RayCastingParameters>(0.8, 0.4, 1.2, 1.0, 0.05, 1.25, 512, 0.25f, 1.0, false);
	}
	else if (std::string(dataPath).find("knee") != std::string::npos){
		dims = make_int3(379, 229, 305);
		spacing = make_float3(1, 1, 1);
	}
	else if (std::string(dataPath).find("181") != std::string::npos){
		dims = make_int3(181, 217, 181);
		spacing = make_float3(1, 1, 1);
		rcp = std::make_shared<RayCastingParameters>(1.8, 1.0, 1.5, 1.0, 0.3, 2.6, 512, 0.25f, 1.0, false); //for 181
	}
	else{
		std::cout << "volume data name not recognized" << std::endl;
		exit(0);
	}

	inputVolume = std::make_shared<Volume>();
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

	channelVolume = std::make_shared<Volume>();
	channelVolume->setSize(inputVolume->size);
	channelVolume->dataOrigin = inputVolume->dataOrigin;
	channelVolume->spacing = inputVolume->spacing;

	computeChannelVolume(inputVolume, channelVolume, rcp);


	
	skelVolume = std::make_shared<Volume>();
	skelVolume->setSize(inputVolume->size);
	skelVolume->dataOrigin = inputVolume->dataOrigin;
	skelVolume->spacing = inputVolume->spacing;
	//initITK(); 
	//computeSkel();



	std::shared_ptr<ScreenMarker> sm = std::make_shared<ScreenMarker>();

	ve = std::make_shared<ViewpointEvaluator>(rcp, inputVolume);
	ve->initDownSampledResultVolume(make_int3(40, 40, 40));


	useLabel = true;
	labelVol = 0;
	if (useLabel){
		//std::shared_ptr<RawVolumeReader> reader;
		//const std::string labelDataPath = dataMgr->GetConfig("FEATURE_PATH");
		//reader = std::make_shared<RawVolumeReader>(labelDataPath.c_str(), dims);
		//labelVol = std::make_shared<VolumeCUDA>();
		//reader->OutputToVolumeCUDAUnsignedShort(labelVol);
		//reader.reset();

		labelVol = std::make_shared<VolumeCUDA>();
		labelVol->VolumeCUDA_init(dims, (unsigned short *)0, 1, 1);

		lvProcessor = std::make_shared<LabelVolumeProcessor>(labelVol);
		lvProcessor->setScreenMarker(sm);
		lvProcessor->rcp = rcp;

		ve->setLabel(labelVol);
	}



	/********GL widget******/

#ifdef USE_OSVR
	matrixMgr = std::make_shared<GLMatrixManager>(true);
#else
	matrixMgr = std::make_shared<GLMatrixManager>(false);
#endif

	bool isImmersive = true;
	if (isImmersive){
		matrixMgr->SetImmersiveMode();
	}

	openGL = std::make_shared<GLWidget>(matrixMgr);


	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setStencilBufferSize(8);
	format.setVersion(2, 0);
	format.setProfile(QSurfaceFormat::CoreProfile);
	openGL->setFormat(format); // must be called before the widget or its parent window gets shown


	float3 posMin, posMax;
	inputVolume->GetPosRange(posMin, posMax);
	matrixMgr->SetVol(posMin, posMax);
	

	matrixMgrMini = std::make_shared<GLMatrixManager>(false);
	matrixMgrMini->SetVol(posMin, posMax);
	

	volumeRenderable = std::make_shared<VolumeRenderableImmerCUDA>(inputVolume, labelVol);
	volumeRenderable->rcp = rcp;
	openGL->AddRenderable("1volume", volumeRenderable.get()); //make sure the volume is rendered first since it does not use depth test
	volumeRenderable->setScreenMarker(sm);


	//if (isImmersive){
	//	immersiveInteractor = std::make_shared<ImmersiveInteractor>();
	//	immersiveInteractor->setMatrixMgr(matrixMgr);
	//	openGL->AddInteractor("model", immersiveInteractor.get());
	//}
	//else{
	//	regularInteractor = std::make_shared<RegularInteractor>();
	//	regularInteractor->setMatrixMgr(matrixMgr);
	//	openGL->AddInteractor("model", regularInteractor.get());
	//}

	immersiveInteractor = std::make_shared<ImmersiveInteractor>();
	immersiveInteractor->setMatrixMgr(matrixMgr);
	openGL->AddInteractor("model", immersiveInteractor.get());

	regularInteractor = std::make_shared<RegularInteractor>();
	regularInteractor->setMatrixMgr(matrixMgr);
	openGL->AddInteractor("modelReg", regularInteractor.get());

	if (isImmersive){
		regularInteractor->isActive = false;
		immersiveInteractor->isActive = true;
	}
	else{
		regularInteractor->isActive = true;
		immersiveInteractor->isActive = false;
	}

	sbInteractor = std::make_shared<ScreenBrushInteractor>();
	sbInteractor->setScreenMarker(sm);
	openGL->AddInteractor("screenMarker", sbInteractor.get());


	if (useLabel){
		openGL->AddProcessor("screenMarkerVolumeProcessor", lvProcessor.get());
	}


	///********controls******/
	QHBoxLayout *mainLayout = new QHBoxLayout;


	saveStateBtn = std::make_shared<QPushButton>("Save State");
	loadStateBtn = std::make_shared<QPushButton>("Load State");
std::cout << posMin.x << " " << posMin.y << " " << posMin.z << std::endl;
std::cout << posMax.x << " " << posMax.y << " " << posMax.z << std::endl;

	QVBoxLayout *controlLayout = new QVBoxLayout;

	controlLayout->addWidget(saveStateBtn.get());
	controlLayout->addWidget(loadStateBtn.get());


	QCheckBox* isBrushingCb = new QCheckBox("Brush", this);
	isBrushingCb->setChecked(sbInteractor->isActive);
	controlLayout->addWidget(isBrushingCb);
	connect(isBrushingCb, SIGNAL(clicked()), this, SLOT(isBrushingClicked()));

	QPushButton *moveToOptimalBtn = new QPushButton("Move to the Optimal Viewpoint");
	controlLayout->addWidget(moveToOptimalBtn);
	connect(moveToOptimalBtn, SIGNAL(clicked()), this, SLOT(moveToOptimalBtnClicked()));

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
	oriVolumeRb = std::make_shared<QRadioButton>(tr("&original volume"));
	channelVolumeRb = std::make_shared<QRadioButton>(tr("&channel volume"));
	oriVolumeRb->setChecked(true);
	deformModeLayout->addWidget(oriVolumeRb.get());
	deformModeLayout->addWidget(channelVolumeRb.get());
	groupBox->setLayout(deformModeLayout);
	controlLayout->addWidget(groupBox);
	connect(oriVolumeRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotOriVolumeRb(bool)));
	connect(channelVolumeRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotChannelVolumeRb(bool)));


	QGroupBox *groupBox2 = new QGroupBox(tr("volume selection"));
	QHBoxLayout *deformModeLayout2= new QHBoxLayout;
	immerRb = std::make_shared<QRadioButton>(tr("&immersive mode"));
	nonImmerRb = std::make_shared<QRadioButton>(tr("&non immer"));
	immerRb->setChecked(true);
	deformModeLayout2->addWidget(immerRb.get());
	deformModeLayout2->addWidget(nonImmerRb.get());
	groupBox2->setLayout(deformModeLayout2);
	controlLayout->addWidget(groupBox2);
	connect(immerRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotImmerRb(bool)));
	connect(nonImmerRb.get(), SIGNAL(clicked(bool)), this, SLOT(SlotNonImmerRb(bool)));

	




	QLabel *transFuncP1SliderLabelLit = new QLabel("Transfer Function Higher Cut Off");
	//controlLayout->addWidget(transFuncP1SliderLabelLit);
	QSlider *transFuncP1LabelSlider = new QSlider(Qt::Horizontal);
	transFuncP1LabelSlider->setRange(0, 100);
	transFuncP1LabelSlider->setValue(volumeRenderable->rcp->transFuncP1 * 100);
	connect(transFuncP1LabelSlider, SIGNAL(valueChanged(int)), this, SLOT(transFuncP1LabelSliderValueChanged(int)));
	transFuncP1Label = new QLabel(QString::number(volumeRenderable->rcp->transFuncP1));
	QHBoxLayout *transFuncP1Layout = new QHBoxLayout;
	transFuncP1Layout->addWidget(transFuncP1LabelSlider);
	transFuncP1Layout->addWidget(transFuncP1Label);
	//controlLayout->addLayout(transFuncP1Layout);

	QLabel *transFuncP2SliderLabelLit = new QLabel("Transfer Function Lower Cut Off");
	//controlLayout->addWidget(transFuncP2SliderLabelLit);
	QSlider *transFuncP2LabelSlider = new QSlider(Qt::Horizontal);
	transFuncP2LabelSlider->setRange(0, 100);
	transFuncP2LabelSlider->setValue(volumeRenderable->rcp->transFuncP2 * 100);
	connect(transFuncP2LabelSlider, SIGNAL(valueChanged(int)), this, SLOT(transFuncP2LabelSliderValueChanged(int)));
	transFuncP2Label = new QLabel(QString::number(volumeRenderable->rcp->transFuncP2));
	QHBoxLayout *transFuncP2Layout = new QHBoxLayout;
	transFuncP2Layout->addWidget(transFuncP2LabelSlider);
	transFuncP2Layout->addWidget(transFuncP2Label);
	//controlLayout->addLayout(transFuncP2Layout);

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
	//controlLayout->addLayout(brLayout);

	QLabel *dsLabelLit = new QLabel("Density of the volume: ");
	//controlLayout->addWidget(dsLabelLit);
	QSlider* dsSlider = new QSlider(Qt::Horizontal);
	dsSlider->setRange(0, 100);
	dsSlider->setValue(volumeRenderable->rcp->density * 20);
	connect(dsSlider, SIGNAL(valueChanged(int)), this, SLOT(dsSliderValueChanged(int)));
	dsLabel = new QLabel(QString::number(volumeRenderable->rcp->density));
	QHBoxLayout *dsLayout = new QHBoxLayout;
	dsLayout->addWidget(dsSlider);
	dsLayout->addWidget(dsLabel);
	//controlLayout->addLayout(dsLayout);


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
	//controlLayout->addLayout(laLayout);

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
	//controlLayout->addLayout(ldLayout);

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

	controlLayout->addWidget(rcGroupBox);

	controlLayout->addStretch();

	connect(saveStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotSaveState()));
	connect(loadStateBtn.get(), SIGNAL(clicked()), this, SLOT(SlotLoadState()));
	connect(eyePosBtn, SIGNAL(clicked()), this, SLOT(applyEyePos()));

	QVBoxLayout *assistLayout = new QVBoxLayout;
	QLabel *miniatureLabel = new QLabel("miniature");
	assistLayout->addWidget(miniatureLabel);

	//matrixMgrMini = std::make_shared<GLMatrixManager>(false);
	//matrixMgrMini->SetVol(posMin, posMax);
	openGLMini = std::make_shared<GLWidget>(matrixMgrMini);
	openGLMini->setFormat(format);
	if (isImmersive){
		volumeRenderableMini = std::make_shared<VolumeRenderableCUDA>(inputVolume);
		//volumeRenderableMini->rcp = rcp;
		volumeRenderableMini->rcp = std::make_shared<RayCastingParameters>(0.8, 2.0, 2.0, 0.9, 0.1, 0.05, 512, 0.25f, 0.6, false);
		openGLMini->AddRenderable("1volume", volumeRenderableMini.get()); //make sure the volume is rendered first since it does not use depth test
		assistLayout->addWidget(openGLMini.get(), 3);
		regularInteractorMini = std::make_shared<RegularInteractor>();
		regularInteractorMini->setMatrixMgr(matrixMgrMini);
		openGLMini->AddInteractor("regular", regularInteractorMini.get());
	}
	//openGLMini->setFixedSize(200, 200);

	mainLayout->addLayout(assistLayout,1);
	openGL->setFixedSize(600, 600);
	mainLayout->addWidget(openGL.get(), 5);
	mainLayout->addLayout(controlLayout,1);
	setLayout(mainLayout);
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


}

void Window::SlotLoadState()
{

}

void Window::applyEyePos()
{
	QString s = eyePosLineEdit->text();
	QStringList sl = s.split(QRegExp("[\\s,]+"));
	matrixMgr->moveEyeInLocalTo(QVector3D(sl[0].toFloat(), sl[1].toFloat(), sl[2].toFloat()));
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
	volumeRenderable->rcp->density = v*1.0 / 20.0;
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


void Window::isBrushingClicked()
{
	sbInteractor->isActive = !sbInteractor->isActive;
}

void Window::moveToOptimalBtnClicked()
{
	ve->compute(VPMethod::JS06Sphere);
	
	matrixMgr->moveEyeInLocalTo(QVector3D(ve->optimalEyeInLocal.x, ve->optimalEyeInLocal.y, ve->optimalEyeInLocal.z));

	ve->saveResultVol("labelEntro.raw");
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
		//volumeRenderable->setVolume(skelVolume);
		volumeRenderable->setVolume(channelVolume);
	}
}

void Window::SlotImmerRb(bool b)
{
	if (b){
		regularInteractor->isActive = false;
		immersiveInteractor->isActive = true;

		matrixMgr->SetImmersiveMode();
	}
}

void Window::SlotNonImmerRb(bool b)
{
	if (b)
	{
		regularInteractor->isActive = true;
		immersiveInteractor->isActive = false;

		matrixMgr->SetNonImmersiveMode();		
	}
}