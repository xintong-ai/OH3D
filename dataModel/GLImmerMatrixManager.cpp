#include <GLImmerMatrixManager.h>

#include <qmatrix4x4.h>
#include <helper_math.h>
#include <fstream>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>






void GLImmerMatrixManager::Rotate(float fromX, float fromY, float toX, float toY)
{
	*rot = trackball->rotate(fromX, fromY,
		toX, toY);
	float m[16];
	rot->matrix(m);
	
	QMatrix4x4 qm = QMatrix4x4(m).transposed();
	//rotMat = qm.inverted() * rotMat;
	QMatrix4x4 temprotMat = qm.inverted() * rotMat;
	
	upVecInLocal; //when rotated, assume the view vec is changed, but we keep the up direction
	viewVecInLocal = normalize(qvec3ToFloat3(temprotMat.inverted()*originalViewVecInLocal));
	
	if (dot(viewVecInLocal, upVecInLocal) > 0.95)
		return;

	//rotMat = temprotMat;
	//resetModelMat();
	//return;


	//std::cout << "the new viewVecInLocal: " << viewVecInLocal.x() << " " << viewVecInLocal.y() << " " << viewVecInLocal.z() << std::endl;

	QMatrix4x4 hypoModel; //hypothetatic model matrix, if not using immersive mode
	hypoModel.translate(-QVector3D(eyeInLocal.x, eyeInLocal.y, eyeInLocal.z));
	QMatrix4x4 hypoView; //hypothetatic view matrix, if not using immersive mode
	hypoView.lookAt(QVector3D(0, 0, 0), float3ToQvec3(make_float3(0,0,0)+viewVecInLocal), float3ToQvec3(upVecInLocal)); //here actually should use viewVecInWorld and upVecInWorld, but since we know hypoModel is just translation, so the vector will not change



	QMatrix4x4 mv = hypoView*hypoModel;
	modelMat = viewMat.inverted()*mv;

	rotMat = modelMat*hypoModel.inverted();
	viewVecInLocal = qvec3ToFloat3(rotMat.inverted()*originalViewVecInLocal);//further adjust


//	std::cout << "the new upVecInLocal: " << upVecInLocal.x << " " << upVecInLocal.y << " " << upVecInLocal.z << std::endl;

	int y = 0;
	return;

}