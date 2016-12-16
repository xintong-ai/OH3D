#ifndef TRANSFORM_FUNC_H
#define TRANSFORM_FUNC_H
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>
//#include "VectorMatrix.h"
//#include "cutil_math.h"

//
//__device__ __host__ inline VECTOR2 GetXY(VECTOR4 pos)
//{
//	return VECTOR2(pos[0], pos[1]);
//}
#include "myMat.h"

inline int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

template <typename T>
__device__ __host__
inline bool invertMatrix(T m[16], T invOut[16])
{
	double inv[16], det;
	int i;

	inv[0] = m[5] * m[10] * m[15] -
		m[5] * m[11] * m[14] -
		m[9] * m[6] * m[15] +
		m[9] * m[7] * m[14] +
		m[13] * m[6] * m[11] -
		m[13] * m[7] * m[10];

	inv[4] = -m[4] * m[10] * m[15] +
		m[4] * m[11] * m[14] +
		m[8] * m[6] * m[15] -
		m[8] * m[7] * m[14] -
		m[12] * m[6] * m[11] +
		m[12] * m[7] * m[10];

	inv[8] = m[4] * m[9] * m[15] -
		m[4] * m[11] * m[13] -
		m[8] * m[5] * m[15] +
		m[8] * m[7] * m[13] +
		m[12] * m[5] * m[11] -
		m[12] * m[7] * m[9];

	inv[12] = -m[4] * m[9] * m[14] +
		m[4] * m[10] * m[13] +
		m[8] * m[5] * m[14] -
		m[8] * m[6] * m[13] -
		m[12] * m[5] * m[10] +
		m[12] * m[6] * m[9];

	inv[1] = -m[1] * m[10] * m[15] +
		m[1] * m[11] * m[14] +
		m[9] * m[2] * m[15] -
		m[9] * m[3] * m[14] -
		m[13] * m[2] * m[11] +
		m[13] * m[3] * m[10];

	inv[5] = m[0] * m[10] * m[15] -
		m[0] * m[11] * m[14] -
		m[8] * m[2] * m[15] +
		m[8] * m[3] * m[14] +
		m[12] * m[2] * m[11] -
		m[12] * m[3] * m[10];

	inv[9] = -m[0] * m[9] * m[15] +
		m[0] * m[11] * m[13] +
		m[8] * m[1] * m[15] -
		m[8] * m[3] * m[13] -
		m[12] * m[1] * m[11] +
		m[12] * m[3] * m[9];

	inv[13] = m[0] * m[9] * m[14] -
		m[0] * m[10] * m[13] -
		m[8] * m[1] * m[14] +
		m[8] * m[2] * m[13] +
		m[12] * m[1] * m[10] -
		m[12] * m[2] * m[9];

	inv[2] = m[1] * m[6] * m[15] -
		m[1] * m[7] * m[14] -
		m[5] * m[2] * m[15] +
		m[5] * m[3] * m[14] +
		m[13] * m[2] * m[7] -
		m[13] * m[3] * m[6];

	inv[6] = -m[0] * m[6] * m[15] +
		m[0] * m[7] * m[14] +
		m[4] * m[2] * m[15] -
		m[4] * m[3] * m[14] -
		m[12] * m[2] * m[7] +
		m[12] * m[3] * m[6];

	inv[10] = m[0] * m[5] * m[15] -
		m[0] * m[7] * m[13] -
		m[4] * m[1] * m[15] +
		m[4] * m[3] * m[13] +
		m[12] * m[1] * m[7] -
		m[12] * m[3] * m[5];

	inv[14] = -m[0] * m[5] * m[14] +
		m[0] * m[6] * m[13] +
		m[4] * m[1] * m[14] -
		m[4] * m[2] * m[13] -
		m[12] * m[1] * m[6] +
		m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] +
		m[1] * m[7] * m[10] +
		m[5] * m[2] * m[11] -
		m[5] * m[3] * m[10] -
		m[9] * m[2] * m[7] +
		m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] -
		m[0] * m[7] * m[10] -
		m[4] * m[2] * m[11] +
		m[4] * m[3] * m[10] +
		m[8] * m[2] * m[7] -
		m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] +
		m[0] * m[7] * m[9] +
		m[4] * m[1] * m[11] -
		m[4] * m[3] * m[9] -
		m[8] * m[1] * m[7] +
		m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] -
		m[0] * m[6] * m[9] -
		m[4] * m[1] * m[10] +
		m[4] * m[2] * m[9] +
		m[8] * m[1] * m[6] -
		m[8] * m[2] * m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	if (det == 0)
		return false;

	det = 1.0 / det;

	for (i = 0; i < 16; i++)
		invOut[i] = inv[i] * det;

	return true;
}


//bool invertMatrix(const float m[16], float invOut[16])
//{
//	float inv[16], det;
//	int i;
//
//	inv[0] = m[5] * m[10] * m[15] -
//		m[5] * m[11] * m[14] -
//		m[9] * m[6] * m[15] +
//		m[9] * m[7] * m[14] +
//		m[13] * m[6] * m[11] -
//		m[13] * m[7] * m[10];
//
//	inv[4] = -m[4] * m[10] * m[15] +
//		m[4] * m[11] * m[14] +
//		m[8] * m[6] * m[15] -
//		m[8] * m[7] * m[14] -
//		m[12] * m[6] * m[11] +
//		m[12] * m[7] * m[10];
//
//	inv[8] = m[4] * m[9] * m[15] -
//		m[4] * m[11] * m[13] -
//		m[8] * m[5] * m[15] +
//		m[8] * m[7] * m[13] +
//		m[12] * m[5] * m[11] -
//		m[12] * m[7] * m[9];
//
//	inv[12] = -m[4] * m[9] * m[14] +
//		m[4] * m[10] * m[13] +
//		m[8] * m[5] * m[14] -
//		m[8] * m[6] * m[13] -
//		m[12] * m[5] * m[10] +
//		m[12] * m[6] * m[9];
//
//	inv[1] = -m[1] * m[10] * m[15] +
//		m[1] * m[11] * m[14] +
//		m[9] * m[2] * m[15] -
//		m[9] * m[3] * m[14] -
//		m[13] * m[2] * m[11] +
//		m[13] * m[3] * m[10];
//
//	inv[5] = m[0] * m[10] * m[15] -
//		m[0] * m[11] * m[14] -
//		m[8] * m[2] * m[15] +
//		m[8] * m[3] * m[14] +
//		m[12] * m[2] * m[11] -
//		m[12] * m[3] * m[10];
//
//	inv[9] = -m[0] * m[9] * m[15] +
//		m[0] * m[11] * m[13] +
//		m[8] * m[1] * m[15] -
//		m[8] * m[3] * m[13] -
//		m[12] * m[1] * m[11] +
//		m[12] * m[3] * m[9];
//
//	inv[13] = m[0] * m[9] * m[14] -
//		m[0] * m[10] * m[13] -
//		m[8] * m[1] * m[14] +
//		m[8] * m[2] * m[13] +
//		m[12] * m[1] * m[10] -
//		m[12] * m[2] * m[9];
//
//	inv[2] = m[1] * m[6] * m[15] -
//		m[1] * m[7] * m[14] -
//		m[5] * m[2] * m[15] +
//		m[5] * m[3] * m[14] +
//		m[13] * m[2] * m[7] -
//		m[13] * m[3] * m[6];
//
//	inv[6] = -m[0] * m[6] * m[15] +
//		m[0] * m[7] * m[14] +
//		m[4] * m[2] * m[15] -
//		m[4] * m[3] * m[14] -
//		m[12] * m[2] * m[7] +
//		m[12] * m[3] * m[6];
//
//	inv[10] = m[0] * m[5] * m[15] -
//		m[0] * m[7] * m[13] -
//		m[4] * m[1] * m[15] +
//		m[4] * m[3] * m[13] +
//		m[12] * m[1] * m[7] -
//		m[12] * m[3] * m[5];
//
//	inv[14] = -m[0] * m[5] * m[14] +
//		m[0] * m[6] * m[13] +
//		m[4] * m[1] * m[14] -
//		m[4] * m[2] * m[13] -
//		m[12] * m[1] * m[6] +
//		m[12] * m[2] * m[5];
//
//	inv[3] = -m[1] * m[6] * m[11] +
//		m[1] * m[7] * m[10] +
//		m[5] * m[2] * m[11] -
//		m[5] * m[3] * m[10] -
//		m[9] * m[2] * m[7] +
//		m[9] * m[3] * m[6];
//
//	inv[7] = m[0] * m[6] * m[11] -
//		m[0] * m[7] * m[10] -
//		m[4] * m[2] * m[11] +
//		m[4] * m[3] * m[10] +
//		m[8] * m[2] * m[7] -
//		m[8] * m[3] * m[6];
//
//	inv[11] = -m[0] * m[5] * m[11] +
//		m[0] * m[7] * m[9] +
//		m[4] * m[1] * m[11] -
//		m[4] * m[3] * m[9] -
//		m[8] * m[1] * m[7] +
//		m[8] * m[3] * m[5];
//
//	inv[15] = m[0] * m[5] * m[10] -
//		m[0] * m[6] * m[9] -
//		m[4] * m[1] * m[10] +
//		m[4] * m[2] * m[9] +
//		m[8] * m[1] * m[6] -
//		m[8] * m[2] * m[5];
//
//	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
//
//	if (det == 0)
//		return false;
//
//	det = 1.0 / det;
//
//	for (i = 0; i < 16; i++)
//		invOut[i] = inv[i] * det;
//
//	return true;
//}

template <typename T>
__device__ __host__ 
inline float4 mat4mulvec4(T *a, float4 b)
{
	float4 c;
	c.x = a[0] * b.x + a[4] * b.y + a[8] * b.z + a[12] * b.w;
	c.y = a[1] * b.x + a[5] * b.y + a[9] * b.z + a[13] * b.w;
	c.z = a[2] * b.x + a[6] * b.y + a[10] * b.z + a[14] * b.w;
	c.w = a[3] * b.x + a[7] * b.y + a[11] * b.z + a[15] * b.w;
	return c;
}

inline __device__ __host__ float2 GetXY(float4 pos)
{
	return make_float2(pos.x, pos.y);
}

//template <typename T>
//__device__ __host__ 
//inline VECTOR4 mat4mulvec4(T *a, VECTOR4 b)
//{
//	VECTOR4 c;
//	c[0] = a[0] * b[0] + a[4] * b[1] + a[8] * b[2] + a[12] * b[3];
//	c[1] = a[1] * b[0] + a[5] * b[1] + a[9] * b[2] + a[13] * b[3];
//	c[2] = a[2] * b[0] + a[6] * b[1] + a[10] * b[2] + a[14] * b[3];
//	c[3] = a[3] * b[0] + a[7] * b[1] + a[11] * b[2] + a[15] * b[3];
//	return c;
//}

//Object space-->Camera space-->Clip space-->Screen space
//multiply projection and modelview matrix

//only multiply projection matrix
template <typename T>
__device__ __host__ 
inline float4 Camera2ClipGlobal(float4 pos, T* projection)
{
	float4 pos2;
	float4 v_screen = mat4mulvec4(projection, pos);//projection * modelview * v;
	pos2.x = v_screen.x / v_screen.w;
	pos2.y = v_screen.y / v_screen.w;
	pos2.z = v_screen.z / v_screen.w;
	pos2.w = 1.0;
	return pos2;
}

//template <typename T>
//__device__ __host__ 
//inline VECTOR4 Camera2Clip(VECTOR4 pos, T* projection)
//{
//	VECTOR4 pos2;
//	VECTOR4 v_screen = mat4mulvec4(projection, pos);//projection * modelview * v;
//	pos2[0] = v_screen[0] / v_screen[3];
//	pos2[1] = v_screen[1] / v_screen[3];
//	pos2[2] = v_screen[2] / v_screen[3];
//	pos2[3] = 1.0;
//	return pos2;
//}


//multiply modelview matrix
template <typename T>
__device__ __host__ inline float4 Object2CameraGlobal(float4 pos, T* modelView)//, float modelview[16], float projection[16])
{
	float4 pos2;
	float4 v_screen = mat4mulvec4(modelView, pos);//projection * modelview * v;
	pos2.x = v_screen.x / v_screen.w;
	pos2.y = v_screen.y / v_screen.w;
	pos2.z = v_screen.z / v_screen.w;
	pos2.w = 1.0;
	return pos2;
}

////multiply modelview matrix
//template <typename T>
//__device__ __host__ inline VECTOR4 Object2CameraGlobal(VECTOR4 pos, T* modelView)//, float modelview[16], float projection[16])
//{
//	VECTOR4 pos2;
//	VECTOR4 v_screen = mat4mulvec4(modelView, pos);//projection * modelview * v;
//	pos2[0] = v_screen[0] / v_screen[3];
//	pos2[1] = v_screen[1] / v_screen[3];
//	pos2[2] = v_screen[2] / v_screen[3];
//	pos2[3] = 1.0;
//	//return v_screen;
//	return pos2;
//}

////multiply modelview matrix
//template <typename T>
//__device__ __host__ inline VECTOR4 Camera2Object(VECTOR4 pos, T* invModelView)//, float modelview[16], float projection[16])
//{
//	VECTOR4 pos2;
//	VECTOR4 v_screen = mat4mulvec4(invModelView, pos);//projection * modelview * v;
//	pos2[0] = v_screen[0] / v_screen[3];
//	pos2[1] = v_screen[1] / v_screen[3];
//	pos2[2] = v_screen[2] / v_screen[3];
//	pos2[3] = 1.0;
//	//return v_screen;
//	return pos2;
//}

//multiply modelview matrix
template <typename T>
__device__ __host__ inline float4 Camera2Object(float4 pos, T* invModelView)//, float modelview[16], float projection[16])
{
	float4 pos2;
	float4 v_screen = mat4mulvec4(invModelView, pos);//projection * modelview * v;
	pos2.x = v_screen.x / v_screen.w;
	pos2.y = v_screen.y / v_screen.w;
	pos2.z = v_screen.z / v_screen.w;
	pos2.w = 1.0;
	//return v_screen;
	return pos2;
}
template <typename T>
__host__ __device__ inline float4 Clip2ObjectGlobal(float4 p, T* invModelView, T* invProjection)//, float modelview[16], float projection[16])
{
	p = mat4mulvec4(invModelView, mat4mulvec4(invProjection, p));
	p.x /= p.w;
	p.y /= p.w;
	p.z /= p.w;
	p.w = 1.0;
	return p;
}

template <typename T>
__host__ __device__ inline float4 Clip2Camera(float4 p, T* invProjection)//, float modelview[16], float projection[16])
{
	p = mat4mulvec4(invProjection, p);
	p.x /= p.w;
	p.y /= p.w;
	p.z /= p.w;
	p.w = 1.0;
	return p;
}

template <typename T>
__device__ __host__ inline float4 Object2Clip(float4 pos, T* modelView, T* projection)//, float modelview[16], float projection[16])
{
	float4 pos_clip;
	float4 v_screen = mat4mulvec4(projection, mat4mulvec4(modelView, pos));//projection * modelview * v;
	pos_clip.x = v_screen.x / v_screen.w;
	pos_clip.y = v_screen.y / v_screen.w;
	pos_clip.z = v_screen.z / v_screen.w;
	pos_clip.w = 1.0;
	return pos_clip;
}

__device__ __host__ inline float4 Object2Clip(float4 pos, matrix4x4 modelView, matrix4x4 projection)//, float modelview[16], float projection[16])
{
	float4 pos_clip;
	float4 v_screen = mat4mulvec4(&(projection.v[0].x), mat4mulvec4(&(modelView.v[0].x), pos));//projection * modelview * v;
	pos_clip.x = v_screen.x / v_screen.w;
	pos_clip.y = v_screen.y / v_screen.w;
	pos_clip.z = v_screen.z / v_screen.w;
	pos_clip.w = 1.0;
	return pos_clip;
}


//template <typename T>
//__device__ __host__ inline VECTOR4 Object2Clip(VECTOR4 pos, T* modelView, T* projection)//, float modelview[16], float projection[16])
//{
//	VECTOR4 pos_clip;
//	VECTOR4 v_screen = mat4mulvec4(projection, mat4mulvec4(modelView, pos));//projection * modelview * v;
//	pos_clip[0] = v_screen[0] / v_screen[3];
//	pos_clip[1] = v_screen[1] / v_screen[3];
//	pos_clip[2] = v_screen[2] / v_screen[3];
//	pos_clip[3] = 1.0;
//	return pos_clip;
//}

__device__ __host__ inline float2 Clip2ScreenGlobal(float2 p, int winWidth, int winHeight)
{
	float2 p2;
	p2.x = (p.x + 1) * winWidth / 2.0;
	p2.y = (p.y + 1) * winHeight / 2.0;
	return p2;
}

//template <typename T>
//__device__ __host__ inline VECTOR2 Clip2ScreenGlobal(T p, float winWidth, float winHeight)
//{
//	VECTOR2 p2;
//	p2[0] = (p[0] + 1) * winWidth / 2.0;
//	p2[1] = (p[1] + 1) * winHeight / 2.0;
//	return p2;
//}

__device__ __host__ inline float2 Screen2Clip(float2 p, float winWidth, float winHeight)
{
	float2 p2;
	p2.x = p.x / winWidth * 2.0 - 1.0;
	p2.y = p.y / winHeight * 2.0 - 1.0;
	return p2;
}

//__device__ __host__ inline VECTOR2 Screen2Clip(VECTOR2 p, float winWidth, float winHeight)
//{
//	VECTOR2 p2;
//	p2[0] = p[0] / winWidth * 2.0 - 1.0;
//	p2[1] = p[1] / winHeight * 2.0 - 1.0;
//	return p2;
//}


inline __device__ __host__ float2 Object2Screen(float4 p, matrix4x4 mv, matrix4x4 pj, int width, int height)
{
	return Clip2ScreenGlobal(GetXY(Object2Clip(p, &mv.v[0].x, &pj.v[0].x)), width, height);
}

template <typename T>
inline __device__ __host__ float2 Object2Screen(float4 p, T* mv, T* pj, int width, int height)
{
	return Clip2ScreenGlobal(GetXY(Object2Clip(p, mv, pj)), width, height);
}

template <typename T>
inline __device__ __host__ float2 Camera2Screen(float4 p, T* pj, int width, int height)
{
	return Clip2ScreenGlobal(GetXY(Camera2ClipGlobal(p, pj)), width, height);
}

__device__ inline bool within_device(float v)
{
	return v >= 0 && v <= 1;
}

inline bool within(float v)
{
	return v >= 0 && v <= 1;
}

__device__ __host__
inline float Determinant4x4(const float4& v0,
const float4& v1,
const float4& v2,
const float4& v3)
{
	float det = v0.w*v1.z*v2.y*v3.x - v0.z*v1.w*v2.y*v3.x -
		v0.w*v1.y*v2.z*v3.x + v0.y*v1.w*v2.z*v3.x +

		v0.z*v1.y*v2.w*v3.x - v0.y*v1.z*v2.w*v3.x -
		v0.w*v1.z*v2.x*v3.y + v0.z*v1.w*v2.x*v3.y +

		v0.w*v1.x*v2.z*v3.y - v0.x*v1.w*v2.z*v3.y -
		v0.z*v1.x*v2.w*v3.y + v0.x*v1.z*v2.w*v3.y +

		v0.w*v1.y*v2.x*v3.z - v0.y*v1.w*v2.x*v3.z -
		v0.w*v1.x*v2.y*v3.z + v0.x*v1.w*v2.y*v3.z +

		v0.y*v1.x*v2.w*v3.z - v0.x*v1.y*v2.w*v3.z -
		v0.z*v1.y*v2.x*v3.w + v0.y*v1.z*v2.x*v3.w +

		v0.z*v1.x*v2.y*v3.w - v0.x*v1.z*v2.y*v3.w -
		v0.y*v1.x*v2.z*v3.w + v0.x*v1.y*v2.z*v3.w;
	return det;
}

__device__ __host__
inline float4 GetBarycentricCoordinate(const float3& v0_,
const float3& v1_,
const float3& v2_,
const float3& v3_,
const float3& p0_)
{
	float4 v0 = make_float4(v0_, 1);
	float4 v1 = make_float4(v1_, 1);
	float4 v2 = make_float4(v2_, 1);
	float4 v3 = make_float4(v3_, 1);
	float4 p0 = make_float4(p0_, 1);
	float4 barycentricCoord = float4();
	const float det0 = Determinant4x4(v0, v1, v2, v3);
	const float det1 = Determinant4x4(p0, v1, v2, v3);
	const float det2 = Determinant4x4(v0, p0, v2, v3);
	const float det3 = Determinant4x4(v0, v1, p0, v3);
	const float det4 = Determinant4x4(v0, v1, v2, p0);
	barycentricCoord.x = (det1 / det0);
	barycentricCoord.y = (det2 / det0);
	barycentricCoord.z = (det3 / det0);
	barycentricCoord.w = (det4 / det0);
	return barycentricCoord;
}



//normalize the tet coord before computing
__device__ __host__
inline float4 GetBarycentricCoordinate2(const float3& v0_,
const float3& v1_,
const float3& v2_,
const float3& v3_,
const float3& p0_)
{
	float3 ave = (v0_ + v1_ + v2_ + v3_) / 4;
	float3 v0 = v0_ - ave;
	float3 v1 = v1_ - ave; 
	float3 v2 = v2_ - ave; 
	float3 v3 = v3_ - ave; 
	float3 p0 = p0_ - ave;

	//float matB2C[16] = { v0_.x, v0_.y, v0_.z, 1.0,
	//	v1_.x, v1_.y, v1_.z, 1.0,
	//	v2_.x, v2_.y, v2_.z, 1.0,
	//	v3_.x, v3_.y, v3_.z, 1.0 };//baricentric coord 2 cartisan coord
	float matB2C[16] = { v0.x, v0.y, v0.z, 1.0,
		v1.x, v1.y, v1.z, 1.0,
		v2.x, v2.y, v2.z, 1.0,
		v3.x, v3.y, v3.z, 1.0 };//baricentric coord 2 cartisan coord

	float matC2B[16];
	invertMatrix(matB2C, matC2B);
	//float4 barycentricCoord = mat4mulvec4(matC2B, make_float4(p0_, 1.0));
	float4 barycentricCoord = mat4mulvec4(matC2B, make_float4(p0, 1.0));
	
	return barycentricCoord;

}

#endif //TRANSFORM_FUNC_H
