#include <time.h>
#include "VolumeRenderableCUDA.h"

#include <vector>

#include <QOpenGLVertexArrayObject>
#include <QOpenGLFunctions_1_2>
#include <QOpenGLFunctions_4_3_Core>
// removing the following lines will cause runtime error
#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()

#include <memory>
#include "DeformGLWidget.h"
#include "helper_math.h"
#include <ModelGrid.h>
#include <cuda_gl_interop.h>

#include "VolumeRenderableCUDAKernel.h"
#include "modelVolumeDeformer.h"
#include "Lens.h"
#include "ModelGrid.h"
#include <TransformFunc.h>



VolumeRenderableCUDA::VolumeRenderableCUDA(std::shared_ptr<Volume> _volume)
{
	volume = _volume;
}

VolumeRenderableCUDA::~VolumeRenderableCUDA()
{
	
	VolumeRender_deinit();

	deinitTextureAndCudaArrayOfScreen();

	//cudaDeviceReset();
};

void VolumeRenderableCUDA::init()
{
	VolumeRender_init();

	initTextureAndCudaArrayOfScreen();
}



void VolumeRenderableCUDA::draw(float modelview[16], float projection[16])
{
	if (!visible)
		return;

	RecordMatrix(modelview, projection);

	int winWidth, winHeight;
	actor->GetWindowSize(winWidth, winHeight);

	QMatrix4x4 q_modelview = QMatrix4x4(modelview).transposed();
	QMatrix4x4 q_invMV = q_modelview.inverted();
	QVector4D q_eye4 = q_invMV.map(QVector4D(0, 0, 0, 1));
	float3 eyeInWorld = make_float3(q_eye4[0], q_eye4[1], q_eye4[2]);

	QMatrix4x4 q_projection = QMatrix4x4(projection).transposed();
	QMatrix4x4 q_mvp = q_projection*q_modelview;
	QMatrix4x4 q_invMVP = q_mvp.inverted();

	q_invMV.copyDataTo(invMVMatrix);
	q_invMVP.copyDataTo(invMVPMatrix); //copyDataTo() automatically copy in row-major order
	q_mvp.copyDataTo(MVPMatrix);
	q_modelview.copyDataTo(MVMatrix);
	q_modelview.normalMatrix().copyDataTo(NMatrix);
	bool isCutaway = vis_method == VIS_METHOD::CUTAWAY;
	VolumeRender_setConstants(MVMatrix, MVPMatrix, invMVMatrix, invMVPMatrix, NMatrix, &isCutaway, &transFuncP1, &transFuncP2, &la, &ld, &ls, &(volume->spacing));
	if (!isFixed){
		recordFixInfo(q_mvp, q_modelview);
	}

	//prepare the storage for output
	uint *d_output;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
		cuda_pbo_resource));
	checkCudaErrors(cudaMemset(d_output, 0, winWidth*winHeight * 4));

	if (lenses != 0 && lenses->size() > 0){
		ComputeDisplace(modelview, projection);
		VolumeRender_setVolume(&(modelVolumeDeformer->volumeCUDADeformed));
	}
	else{
		VolumeRender_setVolume(&(modelVolumeDeformer->originalVolume->volumeCuda));
	}

	VolumeRender_setGradient(&(modelVolumeDeformer->volumeCUDAGradient));

	//compute the dvr
	VolumeRender_render(d_output, winWidth, winHeight, density, brightness, eyeInWorld, volume->size, maxSteps, tstep, useColor);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

	// display results
	glClear(GL_COLOR_BUFFER_BIT);

	// draw image from PBO
	glDisable(GL_DEPTH_TEST);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	// draw using texture
	// copy from pbo to texture
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	qgl->glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, volumeTex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, winWidth, winHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);


	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// draw textured quad

	auto functions12 = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_1_2>();
	functions12->glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0);
	glVertex2f(-1, -1);
	glTexCoord2f(1, 0);
	glVertex2f(1, -1);
	glTexCoord2f(1, 1);
	glVertex2f(1, 1);
	glTexCoord2f(0, 1);
	glVertex2f(-1, 1);
	glEnd();

	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	glEnable(GL_DEPTH_TEST);
}


void VolumeRenderableCUDA::ComputeDisplace(float _mv[16], float _pj[16])
{
	if (lenses!=0 && lenses->size() > 0){
		Lens *l = lenses->back();
		if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE && l->type == TYPE_LINE && l->isConstructing == false && modelGrid->gridType == LINESPLIT_UNIFORM_GRID){
			float focusRatio = l->focusRatio;

			//convert the camera location from camera space to object space
			//https://www.opengl.org/archives/resources/faq/technical/viewing.htm
			QMatrix4x4 q_modelview = QMatrix4x4(_mv);
			q_modelview = q_modelview.transposed();
			QMatrix4x4 q_inv_modelview = q_modelview.inverted();

			QVector4D cameraObj = q_inv_modelview * QVector4D(0, 0, 0, 1);
			cameraObj = cameraObj / cameraObj.w();
			float3 lensCen = l->c;

			float3 lensDir = make_float3(
				cameraObj.x() - lensCen.x,
				cameraObj.y() - lensCen.y,
				cameraObj.z() - lensCen.z);
			lensDir = normalize(lensDir);

			//LineLens has info on the screen space. need to compute info on the world space
			//the following computation code is temporarily placed here. a better choice is to put them in lens.cpp, but more design of Lens.h is needed

			//screen length to object
			int winWidth, winHeight;
			actor->GetWindowSize(winWidth, winHeight);

			//transfer the end points of the major and minor axis to global space
			float2 centerScreen = l->GetCenterScreenPos(_mv, _pj, winWidth, winHeight);
			float2 endPointSemiMajorAxisScreen = centerScreen + ((LineLens*)l)->lSemiMajorAxis*((LineLens*)l)->direction;
			float2 endPointSemiMinorAxisScreen = centerScreen + ((LineLens*)l)->lSemiMinorAxis*make_float2(-((LineLens*)l)->direction.y, ((LineLens*)l)->direction.x);

			float4 centerInClip = Object2Clip(make_float4(l->c, 1), _mv, _pj);

			//_prj means the point's projection on the plane that has the same z clip-space coord with the lens center
			float3 endPointSemiMajorAxisClip_prj = make_float3(Screen2Clip(endPointSemiMajorAxisScreen, winWidth, winHeight), centerInClip.z);
			float3 endPointSemiMinorAxisClip_prj = make_float3(Screen2Clip(endPointSemiMinorAxisScreen, winWidth, winHeight), centerInClip.z);

			QMatrix4x4 q_projection = QMatrix4x4(_pj);
			q_projection = q_projection.transposed();
			QMatrix4x4 q_inv_projection = q_projection.inverted();

			float *_invmv = q_inv_modelview.data();
			float *_inpj = q_inv_projection.data();
			float3 endPointSemiMajorAxisGlobal_prj = make_float3(Clip2ObjectGlobal(make_float4(endPointSemiMajorAxisClip_prj, 1), _invmv, _inpj));
			float3 endPointSemiMinorAxisGlobal_prj = make_float3(Clip2ObjectGlobal(make_float4(endPointSemiMinorAxisClip_prj, 1), _invmv, _inpj));


			//using the end points of the major and minor axis in global space, to compute the length and direction of major and minor axis in global space
			float lSemiMajorAxisGlobal_prj = length(endPointSemiMajorAxisGlobal_prj - l->c);
			float lSemiMinorAxisGlobal_prj = length(endPointSemiMinorAxisGlobal_prj - l->c);
			float3 majorAxisGlobal_prj = normalize(endPointSemiMajorAxisGlobal_prj - l->c);
			float3 minorAxisGlobal_prj = normalize(endPointSemiMinorAxisGlobal_prj - l->c);

			float3 minorAxisGlobal = normalize(cross(lensDir, majorAxisGlobal_prj));
			float3 majorAxisGlobal = normalize(cross(minorAxisGlobal, lensDir));
			float lSemiMajorAxisGlobal = lSemiMajorAxisGlobal_prj / dot(majorAxisGlobal, majorAxisGlobal_prj);
			float lSemiMinorAxisGlobal = lSemiMinorAxisGlobal_prj / dot(minorAxisGlobal, minorAxisGlobal_prj);

			//if (l->justChanged){
				modelGrid->ReinitiateMesh(l->c, lSemiMajorAxisGlobal, lSemiMinorAxisGlobal, majorAxisGlobal, ((LineLens*)l)->focusRatio, lensDir, 0, 0, volume.get());
			//}

			modelGrid->Update(&lensCen.x, &lensDir.x, lSemiMajorAxisGlobal, lSemiMinorAxisGlobal, focusRatio, majorAxisGlobal);

			modelVolumeDeformer->deformByModelGrid(modelGrid->GetLensSpaceOrigin(), majorAxisGlobal, lensDir, modelGrid->GetNumSteps(), modelGrid->GetStep());
			modelVolumeDeformer->computeGradient();

		}
	}

}


void VolumeRenderableCUDA::resetVolume()
{
	//VolumeCUDA_deinit(&volumeCUDACur);

	//int winWidth, winHeight;
	//actor->GetWindowSize(winWidth, winHeight);

	//cudaExtent volumeSize = make_cudaExtent(volume->size[0], volume->size[1], volume->size[2]);
	//VolumeCUDA_init(&volumeCUDACur, volumeSize, volume, 1);

	//isFixed = false;
	//curDeformDegree = 0;
}




void VolumeRenderableCUDA::mousePress(int x, int y, int modifier)
{
	lastPt = make_int2(x, y);
}

void VolumeRenderableCUDA::mouseRelease(int x, int y, int modifier)
{

}

void VolumeRenderableCUDA::mouseMove(int x, int y, int modifier)
{

}

bool VolumeRenderableCUDA::MouseWheel(int x, int y, int modifier, int delta)
{

	return false;
}


void VolumeRenderableCUDA::initTextureAndCudaArrayOfScreen()
{
	int winWidth, winHeight;
	actor->GetWindowSize(winWidth, winHeight);

	qgl->glGenBuffers(1, &pbo);
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	qgl->glBufferData(GL_PIXEL_UNPACK_BUFFER, winWidth*winHeight*sizeof(GLubyte)* 4, 0, GL_STREAM_DRAW);
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

	// create texture for display
	qgl->glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &volumeTex);
	glBindTexture(GL_TEXTURE_2D, volumeTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, winWidth, winHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void VolumeRenderableCUDA::deinitTextureAndCudaArrayOfScreen()
{
	if (cuda_pbo_resource != 0)
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
	if (pbo != 0)
		qgl->glDeleteBuffers(1, &pbo);
	if (volumeTex != 0)
		glDeleteTextures(1, &volumeTex);


}

void VolumeRenderableCUDA::resize(int width, int height)
{
	visible = false;
	deinitTextureAndCudaArrayOfScreen();
	initTextureAndCudaArrayOfScreen();
	visible = true;
}

