#ifndef MATRIX_MANAGER_H
#define MATRIX_MANAGER_H
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>

class MatrixManager{ //to separate with Qt
protected:
	float3 eyeInLocal;
	float3 viewVecInLocal;
	float3 upVecInLocal = make_float3(0.0f, 0.0f, 1.0f);
public:
	float3 getEyeInLocal(){
		return eyeInLocal;
	};
	float3 getViewVecInLocal(){
		return viewVecInLocal;
	};
	float3 getMoveVecFromViewAndUp(){
		return normalize(cross(upVecInLocal,cross(viewVecInLocal, upVecInLocal)));
	}

};
#endif