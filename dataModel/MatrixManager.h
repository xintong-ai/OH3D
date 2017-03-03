#ifndef MATRIX_MANAGER_H
#define MATRIX_MANAGER_H
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>

class MatrixManager //to separate with Qt, so it can be used by cuda code
{ 
protected:
	//float3 eyeInLocal;
	//float3 viewVecInLocal;
	//float3 upVecInLocal = make_float3(0.0f, 0.0f, 1.0f);
public:
	//virtual float3 getEyeInLocal(){
	//	return eyeInLocal;
	//};
	//virtual float3 getUpInLocal(){
	//	return upVecInLocal;
	//};
	//virtual float3 getViewVecInLocal(){
	//	return viewVecInLocal;
	//};
	//virtual float3 getHorizontalMoveVec(){
	//	return normalize(cross(upVecInLocal,cross(viewVecInLocal, upVecInLocal)));
	//}
	virtual float3 getEyeInLocal() = 0;
	virtual float3 getUpInLocal() = 0;
	virtual float3 getViewVecInLocal() = 0;
	virtual float3 getHorizontalMoveVec(float3 refUpInLocal) = 0; //the direction decided by view vec, but perpendicular to the refUpInLocal

	int recentChange = 0; 
	//0. no change is recorded; 1. changed by moving forward. 2. backward. 3. left; 4. right. 5. down; 6. up
};
#endif