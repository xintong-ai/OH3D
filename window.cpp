#include "window.h"
#include "glwidget.h"
//#include "VecReader.h"
#include "BoxRenderable.h"
#include "ParticleReader.h"
#include "DTIVolumeReader.h"
#include "SphereRenderable.h"
#include "SQRenderable.h"
#include "LensRenderable.h"
#include "GridRenderable.h"
#include "Displace.h"

#include "VecReader.h"
#include "ArrowRenderable.h"
#include "DataMgr.h"
#include "VRWidget.h"
#include "VRGlyphRenderable.h"
#include "ModelGridRenderable.h"

#include "GLMatrixManager.h"


#include <LeapListener.h>
#include <Leap.h>

enum DATA_TYPE
{
	TYPE_PARTICLE,
	TYPE_TENSOR,
	TYPE_VECTOR,
};

QSlider* CreateSlider()
{
	QSlider* slider = new QSlider(Qt::Horizontal);
	slider->setRange(0, 10);
	slider->setValue(5);
	return slider;
}

class GLTextureCube;
Window::Window()
{
    setWindowTitle(tr("Interactive Glyph Visualization"));
	QHBoxLayout *mainLayout = new QHBoxLayout;

	dataMgr = std::make_unique<DataMgr>();
	
	//QSizePolicy fixedPolicy(QSizePolicy::Policy::Fixed, QSizePolicy::Policy::Fixed);
	std::unique_ptr<Reader> reader;

	DATA_TYPE dataType = DATA_TYPE::TYPE_PARTICLE; //DATA_TYPE::TYPE_VECTOR;//DATA_TYPE::TYPE_TENSOR; //
	if ("TYPE_PARTICLE" == dataMgr->GetConfig("DATA_TYPE")){
		dataType = DATA_TYPE::TYPE_PARTICLE;
	}
	else if ("TYPE_VECTOR" == dataMgr->GetConfig("DATA_TYPE")){
		dataType = DATA_TYPE::TYPE_VECTOR;
	}
	else if ("TYPE_TENSOR" == dataMgr->GetConfig("DATA_TYPE")){
		dataType = DATA_TYPE::TYPE_TENSOR;
	}
	const std::string dataPath = dataMgr->GetConfig("DATA_PATH");
	if (DATA_TYPE::TYPE_PARTICLE == dataType) {
		reader = std::make_unique<ParticleReader>(dataPath.c_str());
		glyphRenderable = std::make_unique<SphereRenderable>(
			((ParticleReader*)reader.get())->GetPos(),
			((ParticleReader*)reader.get())->GetVal());
	}
	else if (DATA_TYPE::TYPE_TENSOR == dataType) {
		reader = std::make_unique<DTIVolumeReader>(dataPath.c_str());
		std::vector<float4> pos;
		std::vector<float> val;
		((DTIVolumeReader*)reader.get())->GetSamples(pos, val);
		glyphRenderable = std::make_unique<SQRenderable>(pos, val);
	}
	else if (DATA_TYPE::TYPE_VECTOR == dataType) {
		reader = std::make_unique<VecReader>(dataPath.c_str());
		std::vector<float4> pos;
		std::vector<float3> vec;
		std::vector<float> val;
		((VecReader*)reader.get())->GetSamples(pos, vec, val);
		glyphRenderable = std::make_unique<ArrowRenderable>(pos, vec, val);
	}
	std::cout << "number of rendered glyphs: " << glyphRenderable->GetNumOfGlyphs() << std::endl;

	/********GL widget******/
	matrixMgr = std::make_shared<GLMatrixManager>();
	openGL = std::make_unique<GLWidget>(matrixMgr);
	lensRenderable = std::make_unique<LensRenderable>();
	lensRenderable->SetDrawScreenSpace(false);
	if ("ON" == dataMgr->GetConfig("VR_SUPPORT")){
		vrWidget = std::make_unique<VRWidget>(matrixMgr, openGL.get());
		vrWidget->setWindowFlags(Qt::Window);
		vrGlyphRenderable = std::make_unique<VRGlyphRenderable>(glyphRenderable.get());
		vrWidget->AddRenderable("glyph", vrGlyphRenderable.get());
		vrWidget->AddRenderable("lens", lensRenderable.get());
		openGL->SetVRWidget(vrWidget.get());
	}
	QSurfaceFormat format;
	format.setDepthBufferSize(24);
	format.setStencilBufferSize(8);
	format.setVersion(2, 0);
	format.setProfile(QSurfaceFormat::CoreProfile);
	openGL->setFormat(format); // must be called before the widget or its parent window gets shown


	float3 posMin, posMax;
	reader->GetPosRange(posMin, posMax);
	gridRenderable = std::make_unique<GridRenderable>(64);
	matrixMgr->SetVol(posMin, posMax);// cubemap->GetInnerDim());
	modelGridRenderable = std::make_unique<ModelGridRenderable>(&posMin.x, &posMax.x, 15);
	//openGL->AddRenderable("bbox", bbox);
	openGL->AddRenderable("glyph", glyphRenderable.get());
	openGL->AddRenderable("lenses", lensRenderable.get());
	openGL->AddRenderable("grid", gridRenderable.get());
	openGL->AddRenderable("model", modelGridRenderable.get());

	///********controls******/
	addLensBtn = new QPushButton("Add Circle Lens");
	addLineLensBtn = new QPushButton("Add Line Lens");
	addCurveBLensBtn = new QPushButton("Add CurveB Lens");
	delLensBtn = std::make_unique<QPushButton>("Delete a Lens");
	addCurveBLensBtn = new QPushButton("Add CurveB Lens");
	adjustOffsetBtn = new QPushButton("Adjust Offset");
	QCheckBox* gridCheck = new QCheckBox("Grid", this);
	QLabel* transSizeLabel = new QLabel("Transition region size:", this);
	QSlider* transSizeSlider = CreateSlider();
	QLabel* sideSizeLabel = new QLabel("Lens side size:", this);
	QSlider* sideSizeSlider = CreateSlider();
	QLabel* glyphSizeAdjustLabel = new QLabel("Glyph size adjust:", this);
	QSlider* glyphSizeAdjustSlider = CreateSlider();
	refineBoundaryBtn = new QPushButton("Refine Lens Boundary Line");
	listener = new LeapListener();
	controller = new Leap::Controller();
	controller->setPolicyFlags(Leap::Controller::PolicyFlag::POLICY_OPTIMIZE_HMD);
	controller->addListener(*listener);

	QVBoxLayout *controlLayout = new QVBoxLayout;
	controlLayout->addWidget(addLensBtn);
	controlLayout->addWidget(addLineLensBtn);

	controlLayout->addWidget(addCurveBLensBtn);
	controlLayout->addWidget(delLensBtn.get());
	controlLayout->addWidget(gridCheck);
	controlLayout->addWidget(transSizeLabel);
	controlLayout->addWidget(transSizeSlider);
	controlLayout->addWidget(sideSizeLabel);
	controlLayout->addWidget(sideSizeSlider);
	controlLayout->addWidget(glyphSizeAdjustLabel);
	controlLayout->addWidget(glyphSizeAdjustSlider);
	controlLayout->addWidget(adjustOffsetBtn);
	controlLayout->addWidget(refineBoundaryBtn);
	controlLayout->addStretch();

	connect(addLensBtn, SIGNAL(clicked()), this, SLOT(AddLens()));
	connect(addLineLensBtn, SIGNAL(clicked()), this, SLOT(AddLineLens()));
	connect(addCurveBLensBtn, SIGNAL(clicked()), this, SLOT(AddCurveBLens()));
	connect(delLensBtn.get(), SIGNAL(clicked()), lensRenderable.get(), SLOT(SlotDelLens()));


	adjustOffsetBtn = new QPushButton("Adjust Offset");
	controlLayout->addWidget(adjustOffsetBtn);
	connect(addCurveBLensBtn, SIGNAL(clicked()), this, SLOT(AddCurveBLens()));
	connect(adjustOffsetBtn, SIGNAL(clicked()), this, SLOT(adjustOffset()));
	connect(gridCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleGrid(bool)));
	connect(transSizeSlider, SIGNAL(valueChanged(int)), lensRenderable.get(), SLOT(SlotFocusSizeChanged(int)));
	connect(sideSizeSlider, SIGNAL(valueChanged(int)), lensRenderable.get(), SLOT(SlotSideSizeChanged(int)));
	connect(glyphSizeAdjustSlider, SIGNAL(valueChanged(int)), glyphRenderable.get(), SLOT(SlotGlyphSizeAdjustChanged(int)));
	connect(listener, SIGNAL(UpdateRightHand(QVector3D, QVector3D, QVector3D)),
		this, SLOT(UpdateRightHand(QVector3D, QVector3D, QVector3D)));
	connect(refineBoundaryBtn, SIGNAL(clicked()), this, SLOT(RefineLensBoundary()));

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
	lensRenderable->AddLineLens();
}


void Window::AddCurveBLens()
{
	lensRenderable->AddCurveBLens();
}

void Window::adjustOffset()
{
	lensRenderable->adjustOffset();
}

void Window::RefineLensBoundary()
{
	lensRenderable->RefineLensBoundary();
}

//void Window::SlotSliceOrieChanged(bool clicked)
//{
//	if (radioX->isChecked()){
//		sliceSlider->setRange(0, cubemap->GetInnerDim(0)/*vecReader->GetVolumeDim().z*/ - 1);
//		glyphRenderable->SlotSetSliceOrie(0);
//	}	else if (radioY->isChecked()){
//		sliceSlider->setRange(0, cubemap->GetInnerDim(1)/*vecReader->GetVolumeDim().z*/ - 1);
//		glyphRenderable->SlotSetSliceOrie(1);
//	}	else if (radioZ->isChecked()){
//		sliceSlider->setRange(0, cubemap->GetInnerDim(2)/*vecReader->GetVolumeDim().z*/ - 1);
//		glyphRenderable->SlotSetSliceOrie(2);
//	}
//	sliceSlider->setValue(0);
//}
//
//void Window::animate()
//{
//	//int v = heightScaleSlider->value();
//	//v = (v + 1) % nHeightScale;
//	//heightScaleSlider->setValue(v);
//	openGL->animate();
//}
//

void Window::SlotToggleGrid(bool b)
{
	gridRenderable->SetVisibility(b);
}

Window::~Window() {
}

void Window::init()
{
	if ("ON" == dataMgr->GetConfig("VR_SUPPORT")){
		vrWidget->show();
	}
}


void Window::UpdateRightHand(QVector3D thumbTip, QVector3D indexTip, QVector3D indexDir)
{
	//std::cout << indexTip.x() << "," << indexTip.y() << "," << indexTip.z() << std::endl;
	lensRenderable->SlotLensCenterChanged(make_float3(indexTip.x(), indexTip.y(), indexTip.z()));
}