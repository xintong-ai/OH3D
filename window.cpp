#include "window.h"
#include "glwidget.h"
//#include "VecReader.h"
#include "BoxRenderable.h"
#include "ParticleReader.h"
#include "SphereRenderable.h"
#include "LensRenderable.h"
#include "Displace.h"

class GLTextureCube;
Window::Window()
{
    setWindowTitle(tr("Interactive Glyph Visualization"));
	QHBoxLayout *mainLayout = new QHBoxLayout;

	//DataMgr dataMgr;
	//QSizePolicy fixedPolicy(QSizePolicy::Policy::Fixed, QSizePolicy::Policy::Fixed);

	//this->move(100, 100);

	/********GL widget******/
	openGL = new GLWidget;
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

	ParticleReader* particleReader = new ParticleReader("D:/onedrive/data/particle/smoothinglength_0.44/run15/099.vtu");
	//Displace* displace = new Displace();
	sphereRenderable = new SphereRenderable(particleReader->GetPos(), particleReader->GetNum(), particleReader->GetVal());
	float3 posMin, posMax;
	particleReader->GetDataRange(posMin, posMax);
	lensRenderable = new LensRenderable();
	//sphereRenderable->SetVolRange(posMin, posMax);
	//BoxRenderable* bbox = new BoxRenderable(vol);// cubemap->GetInnerDim());
	//bbox->SetVisibility(true);
	openGL->SetVol(posMin, posMax);// cubemap->GetInnerDim());
	//openGL->AddRenderable("bbox", bbox);
	openGL->AddRenderable("spheres", sphereRenderable);
	openGL->AddRenderable("lenses", lensRenderable);

	///********controls******/
	QVBoxLayout *controlLayout = new QVBoxLayout;

	//QGroupBox *groupBox = new QGroupBox(tr("Slice Orientation"));

	addLensBtn = new QPushButton("Add Circle Lens");
	QHBoxLayout* addThingsLayout = new QHBoxLayout;
	addThingsLayout->addWidget(addLensBtn);
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
	controlLayout->addStretch();


	connect(addLensBtn, SIGNAL(clicked()), this, SLOT(AddLens()));
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

	//QLabel* sliceThicknessLabel = new QLabel("Layer thickness:", this);
	//QSlider* numPartSlider = new QSlider(Qt::Horizontal);
	////numPartSlider->setFixedSize(120, 30);
	//numPartSlider->setRange(1, cubemap->GetInnerDim(glyphRenderable->GetSliceDimIdx()));
	//numPartSlider->setValue(1);

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
	//QCheckBox* animationCheck = new QCheckBox("Animation", this);
	//connect(animationCheck, SIGNAL(clicked(bool)), this, SLOT(SlotSetAnimation(bool)));
	//interactLayout->addWidget(animationCheck);
	//controlLayout->addStretch();
	mainLayout->addWidget(openGL,3);
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
//void Window::SlotSetAnimation(bool doAnimation)
//{
//	if (doAnimation){
//		aTimer->start();
//		glyphRenderable->SetAnimationOn(true);
//	}
//	else {
//		aTimer->stop();
//		glyphRenderable->SetAnimationOn(false);
//	}
//}

Window::~Window() {
	//TOOO:
//	delete ;
}
