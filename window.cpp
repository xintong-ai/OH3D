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

enum DATA_TYPE
{
	TYPE_PARTICLE,
	TYPE_TENSOR,
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

	//DataMgr dataMgr;
	//QSizePolicy fixedPolicy(QSizePolicy::Policy::Fixed, QSizePolicy::Policy::Fixed);

	//this->move(100, 100);

	/********GL widget******/
	openGL = std::make_unique<GLWidget>();
	QSurfaceFormat format;
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setVersion(2, 0);
    format.setProfile(QSurfaceFormat::CoreProfile);
    openGL->setFormat(format); // must be called before the widget or its parent window gets shown

	//VecReader* vecReader = new VecReader("D:/data/plume/15plume3d421-504x504x2048.vec");
	//VecReader* vecReader = new VecReader("D:/OneDrive/data/plume/15plume3d421.vec");
	//VecReader* vecReader = new VecReader("D:/OneDrive/data/nek/nek_512.vec");
	//VecReader* vecReader = new VecReader("data/nek.vec");
	
	 
	//VecReader* vecReader = new VecReader("E:/OSU-files/HaloWorkNew/vechist_data/15plume3d421.vec");
	//VecReader* vecReader = new VecReader("D:/OneDrive/data/plume/15plume3d421-504x504x2048.vec");
	//VecReader* vecReader = new VecReader("D:/OneDrive/data/isabel/UVWf01.vec");
	//VecReader* vecReader = new VecReader("D:/OneDrive/data/tornado/1.vec");

	//Streamline* streamline = new Streamline(vecReader->GetFileName().c_str());
	//LineRenderable* lineRenderable = new LineRenderable(streamline);
	//openGL->AddRenderable("streamlines", lineRenderable);

	//cubemap = new Cubemap(vecReader);
	//cubemap = new Cubemap("D:/Dropbox/hist/VecHist/python/crystal/data/universe_hist.bin");
	//cubemap->GenCubeMap(55, 55, 300, 10, 10, 10);
	//glyphRenderable = new GlyphRenderable(cubemap);
	//int3 innerDim = cubemap->GetInnerDim();
	//glyphRenderable->SetVolumeDim(innerDim.x, innerDim.y, innerDim.z);
	//openGL->AddRenderable("glyphs", glyphRenderable);
	std::unique_ptr<Reader> reader;

	const DATA_TYPE dataType = DATA_TYPE::TYPE_PARTICLE;// DATA_TYPE::TYPE_PARTICLE;//
	if (DATA_TYPE::TYPE_PARTICLE == dataType) {
		reader = std::make_unique<ParticleReader>
			("D:/Data/FPM/smoothinglength_0.44/run15/099.vtu");
		//ParticleReader* particleReader = new ParticleReader("D:/Data/VISContest2016/099.vtu");
		//Displace* displace = new Displace();
		glyphRenderable = std::make_unique<SphereRenderable>(
			((ParticleReader*)reader.get())->GetPos(),
			((ParticleReader*)reader.get())->GetVal());
	}
	else if (DATA_TYPE::TYPE_TENSOR == dataType) {
		//reader = std::make_unique<DTIVolumeReader>
		//	("D:/onedrive/data/WhiteMatterExplorationData/DTIVolume.nhdr");
		reader = std::make_unique<DTIVolumeReader>
			("D:/Data/dti-challenge-2015/patient1_dti/patient1_dti.nhdr");
		std::vector<float4> pos;
		std::vector<float> val;
		((DTIVolumeReader*)reader.get())->GetSamples(pos, val);
		glyphRenderable = std::make_unique<SQRenderable>(pos, val);
		//glyphRenderable = std::make_unique<SQRenderable>(
		//	);
		//	((ParticleReader*)reader.get())->GetPos(),
		//	((ParticleReader*)reader.get())->GetVal());
	}

	float3 posMin, posMax;
	reader->GetPosRange(posMin, posMax);
	lensRenderable = std::make_unique<LensRenderable>();

	gridRenderable = std::make_unique<GridRenderable>(64);
	//sphereRenderable->SetVolRange(posMin, posMax);
	//BoxRenderable* bbox = new BoxRenderable(vol);// cubemap->GetInnerDim());
	//bbox->SetVisibility(true);
	openGL->SetVol(posMin, posMax);// cubemap->GetInnerDim());
	//openGL->AddRenderable("bbox", bbox);
	openGL->AddRenderable("glyph", glyphRenderable.get());
	openGL->AddRenderable("lenses", lensRenderable.get());
	openGL->AddRenderable("grid", gridRenderable.get());

	///********controls******/
	QVBoxLayout *controlLayout = new QVBoxLayout;

	//QGroupBox *groupBox = new QGroupBox(tr("Slice Orientation"));

	addLensBtn = new QPushButton("Add Circle Lens");
	addLineLensBtn = new QPushButton("Add Line Lens");
	addPolyLineLensBtn = new QPushButton("Add Poly Line Lens");
	addCurveLensBtn = new QPushButton("Add Curve Line Lens");
	//QHBoxLayout* addThingsLayout = new QHBoxLayout;
	//addThingsLayout->addWidget(addLensBtn);
	//addThingsLayout->addWidget(addLineLensBtn);

	QCheckBox* gridCheck = new QCheckBox("Grid", this);
	connect(gridCheck, SIGNAL(clicked(bool)), this, SLOT(SlotToggleGrid(bool)));

	QLabel* transSizeLabel = new QLabel("Transition region size:", this);
	QSlider* transSizeSlider = CreateSlider();
	connect(transSizeSlider, SIGNAL(valueChanged(int)), lensRenderable.get(), SLOT(SlotFocusSizeChanged(int)));

	QLabel* sideSizeLabel = new QLabel("Lens side size:", this);
	QSlider* sideSizeSlider = CreateSlider();
	connect(sideSizeSlider, SIGNAL(valueChanged(int)), lensRenderable.get(), SLOT(SlotSideSizeChanged(int)));

	QLabel* glyphSizeAdjustLabel = new QLabel("Glyph size adjust:", this);
	QSlider* glyphSizeAdjustSlider = CreateSlider();
	connect(glyphSizeAdjustSlider, SIGNAL(valueChanged(int)), glyphRenderable.get(), SLOT(SlotGlyphSizeAdjustChanged(int)));

	//radioX = new QRadioButton(tr("&X"));
	//radioY = new QRadioButton(tr("&Y"));
	//radioZ = new QRadioButton(tr("&Z"));
	//radioX->setChecked(true);
	//QHBoxLayout *sliceOrieLayout = new QHBoxLayout;
	//sliceOrieLayout->addWidget(radioX);
	//sliceOrieLayout->addWidget(radioY);
	//sliceOrieLayout->addWidget(radioZ);
	//sliceOrieLayout->addStretch();
	//groupBox->setLayout(sliceOrieLayout);
	//controlLayout->addWidget(groupBox);
	controlLayout->addWidget(addLensBtn);
	controlLayout->addWidget(addLineLensBtn);
	controlLayout->addWidget(addPolyLineLensBtn);
	controlLayout->addWidget(addCurveLensBtn);

	controlLayout->addWidget(gridCheck);
	controlLayout->addWidget(transSizeLabel);
	controlLayout->addWidget(transSizeSlider);
	controlLayout->addWidget(sideSizeLabel);
	controlLayout->addWidget(sideSizeSlider);
	controlLayout->addWidget(glyphSizeAdjustLabel);
	controlLayout->addWidget(glyphSizeAdjustSlider);
	
	controlLayout->addStretch();


	connect(addLensBtn, SIGNAL(clicked()), this, SLOT(AddLens()));
	connect(addLineLensBtn, SIGNAL(clicked()), this, SLOT(AddLineLens()));
	connect(addPolyLineLensBtn, SIGNAL(clicked()), this, SLOT(AddPolyLineLens()));
	connect(addCurveLensBtn, SIGNAL(clicked()), this, SLOT(AddCurveLens()));


	//connect(radioX, SIGNAL(clicked(bool)), this, SLOT(SlotSliceOrieChanged(bool)));
	//connect(radioY, SIGNAL(clicked(bool)), this, SLOT(SlotSliceOrieChanged(bool)));
	//connect(radioZ, SIGNAL(clicked(bool)), this, SLOT(SlotSliceOrieChanged(bool)));

	//QCheckBox* blockBBoxCheck = new QCheckBox("Block bounding box", this);
	//connect(blockBBoxCheck, SIGNAL(clicked(bool)), glyphRenderable, SLOT(SlotSetCubesVisible(bool)));

	////statusLabel = new QLabel("status: Navigation");
	//QLabel* sliceLabel = new QLabel("Slice number:", this);
	//sliceSlider = new QSlider(Qt::Horizontal);
	////sliceSlider->setFixedSize(120, 30);
	//sliceSlider->setRange(0, cubemap->GetInnerDim( glyphRenderable->GetSliceDimIdx())/*vecReader->GetVolumeDim().z*/ - 1);
	//sliceSlider->setValue(0);



	//QLabel* heightScaleLabel = new QLabel("Glyph Height Scale:", this);
	//heightScaleSlider = new QSlider(Qt::Horizontal);
	////numPartSlider->setFixedSize(120, 30);
	//heightScaleSlider->setRange(0, nScale);
	//heightScaleSlider->setValue(glyphRenderable->GetHeightScale());

	//QLabel* sizeScaleLabel = new QLabel("Glyph Size Scale:", this);
	//sizeScaleSlider = new QSlider(Qt::Horizontal);
	////numPartSlider->setFixedSize(120, 30);
	//sizeScaleSlider->setRange(1, nScale);
	//sizeScaleSlider->setValue(glyphRenderable->GetSizeScale());
	////glyphRenderable->SetSizeScale(5);

	//QLabel* expScaleLabel = new QLabel("Exponential Scale:", this);
	//QComboBox *expScaleCombo = new QComboBox();
	//expScaleCombo->addItem("1", 0);
	//expScaleCombo->addItem("2", 1);
	//expScaleCombo->addItem("3", 2);
	//expScaleCombo->setCurrentIndex(1);

	//QVBoxLayout *interactLayout = new QVBoxLayout;
	//QGroupBox *interactGrpBox = new QGroupBox(tr("Interactions"));
	//interactLayout->addWidget(blockBBoxCheck);
	//interactLayout->addWidget(sliceLabel);
	//interactLayout->addWidget(sliceSlider);
	//interactLayout->addWidget(sliceThicknessLabel);
	//interactLayout->addWidget(numPartSlider);
	//interactLayout->addWidget(heightScaleLabel);
	//interactLayout->addWidget(heightScaleSlider);
	//interactLayout->addWidget(sizeScaleLabel);
	//interactLayout->addWidget(sizeScaleSlider);
	//interactLayout->addWidget(expScaleLabel);
	//interactLayout->addWidget(expScaleCombo);
	//interactGrpBox->setLayout(interactLayout);
	//controlLayout->addWidget(interactGrpBox);
	//connect(sliceSlider, SIGNAL(valueChanged(int)), glyphRenderable, SLOT(SlotSliceNumChanged(int)));
	//connect(numPartSlider, SIGNAL(valueChanged(int)), glyphRenderable, SLOT(SlotNumPartChanged(int)));
	//connect(heightScaleSlider, SIGNAL(valueChanged(int)), glyphRenderable, SLOT(SlotHeightScaleChanged(int)));
	//connect(sizeScaleSlider, SIGNAL(valueChanged(int)), glyphRenderable, SLOT(SlotSizeScaleChanged(int)));
	//connect(expScaleCombo, SIGNAL(currentIndexChanged(int)), glyphRenderable, SLOT(SlotExpScaleChanged(int)));

	//GL2DProjWidget* gl2DProjWidget = new GL2DProjWidget(this);
	//controlLayout->addWidget(gl2DProjWidget);
	////gl2DProjWidget->SetCubeTexture(glyphRenderable->GetCubeTexture(0));
	//connect(glyphRenderable, SIGNAL(SigChangeTex(GLTextureCube*, Cube*)), 
	//	gl2DProjWidget, SLOT(SlotSetCubeTexture(GLTextureCube*, Cube*)));

	//aTimer = new QTimer;
	//connect(aTimer,SIGNAL(timeout()),SLOT(animate()));
	//aTimer->start(33);
	//aTimer->stop();
	//

	//interactLayout->addWidget(animationCheck);
	//controlLayout->addStretch();
	mainLayout->addWidget(openGL.get(),3);
	mainLayout->addLayout(controlLayout,1);
	setLayout(mainLayout);
}

void Window::AddLens()
{
	//sphereRenderable->AddCircleLens();
	lensRenderable->AddCircleLens();
	//openGL->AddLens();
	//lensWidSlider->setValue(((GLLensTracer*)lensRenderable)->GetLensWidth() * 10);
	//UpdateStatusLabel();
}

void Window::AddLineLens()
{
	lensRenderable->AddLineLens();
}

void Window::AddPolyLineLens()
{
	lensRenderable->AddPolyLineLens();
}

void Window::AddCurveLens()
{
	lensRenderable->AddCurveLens();
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
	//TOOO:
//	delete ;
}
