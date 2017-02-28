#ifndef MATRIXINTERACTOR_H
#define MATRIXINTERACTOR_H

#include "mouse/trackball.h"
#include "mouse/Rotation.h"

#include "Interactor.h"
class GLMatrixManager;
class MatrixInteractor :public Interactor
{
protected:
	std::shared_ptr<GLMatrixManager> matrixMgr;
	Trackball *trackball;
	Rotation *rot;
public:
	MatrixInteractor(){
		trackball = new Trackball();
		rot = new Rotation();
	};
	~MatrixInteractor(){
		delete trackball;
		delete rot;
	};

	void setMatrixMgr(std::shared_ptr<GLMatrixManager> m){ matrixMgr = m; };

};
#endif