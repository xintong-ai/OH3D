#include <GLMatrixManager.h>
#include <mouse/trackball.h>
#include <mouse/Rotation.h>
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
GLMatrixManager::GLMatrixManager(bool _vrMode) :vrMode(_vrMode)
{
	trackball = new Trackball();
	rot = new Rotation();
	rotMat.setToIdentity();

	if (vrMode) {
		transVec = QVector3D(0.0f, 0.0f, -3.0f);//move it towards the front of the camera
		transScale = 2;
		float3 dataCen = (dataMin + dataMax) * 0.5;
		cofLocal = QVector3D(dataCen.x, dataCen.y, dataCen.z);
	}
	else if (immersiveMode)
	{
		transVec = QVector3D(0.0f, 0.0f, 0.0f);//not using this in immersiveMode

		eyeInWorld = QVector3D(0.0f, 0.0f, 1.01f);
		viewMat.lookAt(eyeInWorld, QVector3D(0.0f, 0.0f, 0.0f), QVector3D(0.0f, 1.0f, 0.0f));// always use (0,0,0) as the world coordinate of the view focus

		//rotMat.rotate(90, QVector3D(-1.0f, 0.0f, 0.0f));
		rotMat.rotate(90, QVector3D(0.0f, 1.0f, 0.0f));

		transScale = 2;
		cofLocal = QVector3D(70, 70, 128);

	}
	else{
		transVec = QVector3D(0.0f, 0.0f, -5.0f);//move it towards the front of the camera
		transScale = 1;
		float3 dataCen = (dataMin + dataMax) * 0.5;
		cofLocal = QVector3D(dataCen.x, dataCen.y, dataCen.z);
	}

}

void GLMatrixManager::SetImmersiveMode()
{
	immersiveMode = true;

	transVec = QVector3D(0.0f, 0.0f, 0.0f);//not using this in immersiveMode

	eyeInWorld = QVector3D(-1.01f, 0.0f, 0.0f);
	viewMat.setToIdentity();
	viewMat.lookAt(eyeInWorld, QVector3D(0.0f, 0.0f, 0.0f), QVector3D(0.0f, 0.0f, 1.0f));// always use (0,0,0) as the world coordinate of the view focus

	rotMat.setToIdentity();

	transScale = 2;
	cofLocal = QVector3D(70, 70, 128);

}


void GLMatrixManager::Rotate(float fromX, float fromY, float toX, float toY)
{
	*rot = trackball->rotate(fromX, fromY,
		toX, toY);
	float m[16];
	rot->matrix(m);
	QMatrix4x4 qm = QMatrix4x4(m).transposed();
	rotMat = qm * rotMat;
}


void GLMatrixManager::GetProjection(float ret[16], float width, float height)
{
	QMatrix4x4 m;
	if (vrMode)
		m.perspective(96.73, (float)width / height, (float)0.01, (float)100);// for VR
	else if (immersiveMode){
		m.perspective(55 / ((float)width / height), (float)width / height, (float)0.01, (float)100);
	}
	else{
		m.perspective(30, (float)width / height, (float)0.1, (float)100);
	}

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


void GLMatrixManager::GetModelMatrix(QMatrix4x4 &m)
{
	m.setToIdentity();
	m.translate(transVec);
	m = m * rotMat;
	m.scale(volScale * transScale * currentTransScale);
	m.translate(-cofLocal);
}

void GLMatrixManager::GetModelMatrix(float ret[16])
{
	QMatrix4x4 m;
	GetModelMatrix(m);

	m = m.transposed();
	m.copyDataTo(ret); //this copy is row major, so we need to transpose it first
}

void GLMatrixManager::GetModelViewMatrix(float mv[16], float _viewMat[16])
{
	QMatrix4x4 m;
	GetModelMatrix(m.data());
	QMatrix4x4 vm(_viewMat);
	vm = vm.transposed();
	m = vm * m;
	m = m.transposed();
	m.copyDataTo(mv);
}

void GLMatrixManager::GetModelViewMatrix(float mv[16])
{
	GetModelViewMatrix(mv, viewMat.data());
}

void GLMatrixManager::GetModelViewMatrix(QMatrix4x4 &mv)
{
	QMatrix4x4 m;
	GetModelMatrix(m);
	mv = viewMat*m;
}

void GLMatrixManager::SetVol(float3 posMin, float3 posMax)
{
	dataMin = posMin;
	dataMax = posMax;

	float3 dataWidth = dataMax - dataMin;
	float dataMaxWidth = std::max(std::max(dataWidth.x, dataWidth.y), dataWidth.z);
	volScale = 2.0f / dataMaxWidth;
	if (!immersiveMode){
		float3 dataCen = (dataMin + dataMax) * 0.5;
		cofLocal = QVector3D(dataCen.x, dataCen.y, dataCen.z);
	}

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
		myfile << rotMat.constData()[i] << " ";
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
		ifs >> rotMat.data()[i];
	}

	ifs >> transVec[0];
	ifs >> transVec[1];
	ifs >> transVec[2];

	ifs >> transScale;

	ifs.close();

}
