#include <GLMatrixManager.h>
#include <Trackball.h>
#include <Rotation.h>
#include <qmatrix4x4.h>
#include <helper_math.h>
#include <fstream>
#include <iostream>
//
//void Perspective(float fovyInDegrees, float aspectRatio,
//	float znear, float zfar)
//{
//	float ymax, xmax;
//	ymax = znear * tanf(fovyInDegrees * M_PI / 360.0);
//	xmax = ymax * aspectRatio;
//	glFrustum(-xmax, xmax, -ymax, ymax, znear, zfar);
//}

GLMatrixManager::GLMatrixManager()
{
	trackball = new Trackball();
	rot = new Rotation();
	transRot.setToIdentity();
}

void GLMatrixManager::Rotate(float fromX, float fromY, float toX, float toY)
{
	*rot = trackball->rotate(fromX, fromY,
		toX, toY);
	float m[16];
	rot->matrix(m);
	QMatrix4x4 qm = QMatrix4x4(m).transposed();
	transRot = qm * transRot;
}

void GLMatrixManager::Translate(float x, float y)
{
	transVec[0] += x;
	transVec[1] += y;
}

void GLMatrixManager::GetProjection(float ret[16], float width, float height)
{
	QMatrix4x4 m;
	//m.perspective(96.73, (float)width / height, (float)0.1, (float)100);// for VR
	m.perspective(30, (float)width / height, (float)0.1, (float)100);// for VR
	m = m.transposed();
	m.copyDataTo(ret); //this copy is row major, so we need to transpose it first
}

void GLMatrixManager::Scale(float v)
{
	transScale *= exp(v * -0.001);
}

void GLMatrixManager::FinishedScale()
{
	transScale *= currentTransScale;
	currentTransScale = 1;
}


void GLMatrixManager::GetModelMatrix(float ret[16])
{
	float3 dataCenter = (dataMin + dataMax) * 0.5;
	float3 dataWidth = dataMax - dataMin;
	float dataMaxWidth = std::max(std::max(dataWidth.x, dataWidth.y), dataWidth.z);
	float scale = 2.0f / dataMaxWidth;
	QMatrix4x4 m;
	m.setToIdentity();	
	m.translate(transVec);
	m = m * transRot;
	m.scale(scale * transScale * currentTransScale);
	m.translate(-dataCenter.x, -dataCenter.y, -dataCenter.z);
	m = m.transposed();
	m.copyDataTo(ret); //this copy is row major, so we need to transpose it first

	//glTranslatef(transVec[0], transVec[1], transVec[2]);
	//glMultMatrixf(transRot.data());
	//glScalef(transScale * currentTransScale, transScale* currentTransScale, transScale* currentTransScale);

	//glScalef(scale, scale, scale);
	//glTranslatef(-dataCenter.x, -dataCenter.y, -dataCenter.z);


	//glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
}

void GLMatrixManager::GetModelView(float mv[16], float _viewMat[16])
{
	QMatrix4x4 m;
	GetModelMatrix(m.data());
	QMatrix4x4 vm(_viewMat);
	vm = vm.transposed();
	m = vm * m;
	m = m.transposed();
	m.copyDataTo(mv);
}

void GLMatrixManager::GetModelView(float mv[16])
{
	GetModelView(mv, viewMat.data());
}


void GLMatrixManager::SetVol(int3 dim)
{
	dataMin = make_float3(0, 0, 0);
	dataMax = make_float3(dim.x - 1, dim.y - 1, dim.z - 1);
}

void GLMatrixManager::SetVol(float3 posMin, float3 posMax)
{
	dataMin = posMin;
	dataMax = posMax;
}


float3 GLMatrixManager::DataCenter()
{
	return (dataMin + dataMax) * 0.5;
}

void GLMatrixManager::SaveState(const char* filename)
{
	std::ofstream myfile;
	myfile.open(filename);
	for (int i = 0; i < 16; i++){
		myfile << transRot.constData()[i] << " ";
	}
	myfile << std::endl;

	for (int i = 0; i < 3; i++) {
		myfile << transVec[i] << " ";
	}
	myfile << std::endl;

	myfile << transScale << std::endl;
	myfile.close();
}

void GLMatrixManager::LoadState(const char* filename)
{
	std::ifstream ifs(filename, std::ifstream::in);

	for (int i = 0; i < 16; i++) {
		ifs >> transRot.data()[i];
	}

	ifs >> transVec[0];
	ifs >> transVec[1];
	ifs >> transVec[2];

	ifs >> transScale;

	ifs.close();

}
