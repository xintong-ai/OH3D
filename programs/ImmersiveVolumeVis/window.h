#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <QVector3D>
#include <memory>
#include "CMakeConfig.h"
#include "myDefine.h"

//itk
#include <itkImage.h>
#include <itkImportImageFilter.h>
#include "itkBinaryThinningImageFilter3D.h"


#include "GLWidgetQtDrawing.h"


class DataMgr;
class GLWidget;
class MarchingCubes;
class QPushButton;
class QSlider;
class Renderable;
class QCheckBox;
class QLabel;
class QRadioButton;
class QTimer;
class GLMatrixManager;
class Volume;
class VolumeCUDA;
class VolumeRenderableImmerCUDA;
class VolumeRenderableCUDA;
class ImmersiveInteractor;
class QLineEdit;
class ScreenBrushInteractor;
class RegularInteractor;
class LabelVolumeProcessor;
class ViewpointEvaluator;

#ifdef USE_OSVR
class VRWidget;
class VRVolumeRenderableCUDA;
#endif

class Window : public QWidget
{
	Q_OBJECT	//without this line, the slot does not work
public:
	Window();
	~Window();

	void init();

private:
	int3 dims;
	float3 spacing;

	RayCastingParameters rcp;
	std::shared_ptr<Volume> inputVolume;

	std::shared_ptr<Volume> channelVolume = 0; //only render when need to test
	std::shared_ptr<Volume> skelVolume = 0; //only render when need to test
	
	std::shared_ptr<VolumeCUDA> labelVolCUDA;
	unsigned short* labelVolLocal = 0;

	bool useLabel;

	std::shared_ptr<GLWidget> openGL;
	
	std::shared_ptr<ImmersiveInteractor> immersiveInteractor;
	std::shared_ptr<ScreenBrushInteractor> sbInteractor;
		
	std::shared_ptr<LabelVolumeProcessor> lvProcessor;

	std::shared_ptr<VolumeRenderableImmerCUDA> volumeRenderable;

	std::shared_ptr<GLMatrixManager> matrixMgr;

	std::shared_ptr<QPushButton> saveStateBtn;
	std::shared_ptr<QPushButton> loadStateBtn;
	QLabel *laLabel, *ldLabel, *lsLabel;
	QLabel *transFuncP1Label, *transFuncP2Label, *brLabel, *dsLabel;
	QLabel *deformForceLabel;
	QLabel *meshResLabel;
	QLineEdit *eyePosLineEdit;

	std::shared_ptr<QRadioButton> oriVolumeRb;
	std::shared_ptr<QRadioButton> channelVolumeRb;
	
	std::shared_ptr<QRadioButton> immerRb;
	std::shared_ptr<QRadioButton> nonImmerRb;

	std::shared_ptr<ViewpointEvaluator> ve;

	QTimer *aTimer;

	//for miniature
	std::shared_ptr<GLWidget> openGLMini;
	std::shared_ptr<VolumeRenderableCUDA> volumeRenderableMini;
	std::shared_ptr<GLMatrixManager> matrixMgrMini;
	std::shared_ptr<RegularInteractor> regularInteractor;
	std::shared_ptr<RegularInteractor> regularInteractorMini;

#ifdef USE_OSVR
	std::shared_ptr<VRWidget> vrWidget;
	std::shared_ptr<VRVolumeRenderableCUDA> vrVolumeRenderable;
#endif


	//for itk
	//typedef unsigned char PixelType;
	typedef float PixelType; //use float when need to render. otherwise should use less storage

	typedef itk::Image< PixelType, 3 > ImageType;
	typedef itk::ImportImageFilter< PixelType, 3 > ImportFilterType;
	const bool importImageFilterWillOwnTheBuffer = false; //probably can change to true for faster speed?
	typedef itk::BinaryThinningImageFilter3D< ImageType, ImageType > ThinningFilterType;
	ImportFilterType::Pointer importFilter;
	ThinningFilterType::Pointer thinningFilter;

	void initITK() //can only be performed AFTER dims and spacing is set
	{
		importFilter = ImportFilterType::New();
		thinningFilter = ThinningFilterType::New();

		ImageType::SizeType imsize;
		imsize[0] = dims.x;
		imsize[1] = dims.y;
		imsize[2] = dims.z;
		ImportFilterType::IndexType start;
		start.Fill(0);
		ImportFilterType::RegionType region;
		region.SetIndex(start);
		region.SetSize(imsize);
		importFilter->SetRegion(region);

		const itk::SpacePrecisionType origin[3] = { imsize[0], imsize[1], imsize[2] };
		importFilter->SetOrigin(origin);
		const itk::SpacePrecisionType _spacing[3] = { spacing.x, spacing.y, spacing.z };
		importFilter->SetSpacing(_spacing);
	};
	void computeSkel();

	Helper helper;

private slots:
	
	void SlotSaveState();
	void SlotLoadState();
	void applyEyePos();

	void transFuncP1LabelSliderValueChanged(int);
	void transFuncP2LabelSliderValueChanged(int); 
	void brSliderValueChanged(int v);
	void dsSliderValueChanged(int v);
	void laSliderValueChanged(int);
	void ldSliderValueChanged(int);
	void lsSliderValueChanged(int);

	void isBrushingClicked();
	void moveToOptimalBtnClicked();

	void SlotOriVolumeRb(bool);
	void SlotChannelVolumeRb(bool);

	void SlotImmerRb(bool);
	void SlotNonImmerRb(bool);

	void zSliderValueChanged(int v);
	void updateLabelVolBtnClicked();
	
};

#endif
