

#ifndef MATRIXINTERACTOR_H
#define MATRIXINTERACTOR_H

#include "Interactor.h"

class GLMatrixManager;
class MatrixInteractor :public Interactor
{
protected:
	std::shared_ptr<GLMatrixManager> matrixMgr;

public:
	MatrixInteractor(){};
	~MatrixInteractor(){};

	void setMatrixMgr(std::shared_ptr<GLMatrixManager> m){ matrixMgr = m; };

};
#endif