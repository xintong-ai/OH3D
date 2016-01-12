#include <Displace.h>
#include <TransformFunc.h>

#include <vector_functions.h>
#include <Lens.h>

void Displace::LoadOrig(float4* v, int num)
{
	posOrig.assign(v, v + num);// , posOrig.begin());
}

struct functor_Object2Clip//: public thrust::unary_function<float,float>
{
	__device__ __host__ float4 operator() (float4 p)
	{
		return Object2Clip(p, &mv.v[0].x, &pj.v[0].x);
	}

	functor_Object2Clip(matrix4x4 _mv, matrix4x4 _pj) {
		mv = _mv;
		pj = _pj;
	}
	matrix4x4 mv, pj;

};

struct functor_Clip2Screen
{
	int w, h;
	__device__ float2 operator() (float4 p)
	{
		return Clip2ScreenGlobal(GetXY(p), w, h);
	}
	functor_Clip2Screen(int _w, int _h) { w = _w; h = _h; }
};

struct functor_Unproject
{
	matrix4x4 inv_mv, inv_pj;
	int w, h;
	__device__ float4 operator() (float4 pClip, float2 pScreen)
	{
		float2 clip = Screen2Clip(pScreen, w, h);
		float4 clip2 = make_float4(clip.x, clip.y, pClip.z, pClip.w);
		return Clip2ObjectGlobal(clip2, &inv_mv.v[0].x, &inv_pj.v[0].x);
	}
	functor_Unproject(matrix4x4 _inv_mv, matrix4x4 _inv_pj, int _w, int _h) {
		inv_mv = _inv_mv;
		inv_pj = _inv_pj;
		w = _w;
		h = _h;
	}

};

void Displace::Compute(float* modelview, float* projection, int winW, int winH, float4* ret)
{
	int size = posOrig.size();

	//clip coordiates of streamlines
	matrix4x4 mv(modelview);
	matrix4x4 pj(projection);

	//if (doRefresh) {

	thrust::device_vector<float4> d_vec_posClip(size);
	thrust::device_vector<float2> d_vec_posScreen(size);

	thrust::transform(posOrig.begin(), posOrig.end(), d_vec_posClip.begin(), functor_Object2Clip(mv, pj));

	thrust::transform(d_vec_posClip.begin(), d_vec_posClip.end(),
		d_vec_posScreen.begin(), functor_Clip2Screen(winW, winH));

	//posScreenTarget = d_vec_posScreen;
	//}

	matrix4x4 invMV;
	matrix4x4 invPJ;
	invertMatrix(&mv.v[0].x, &invMV.v[0].x);
	invertMatrix(&pj.v[0].x, &invPJ.v[0].x);

	thrust::device_vector<float4> d_vec_ret(size);
	thrust::transform(d_vec_posClip.begin(), d_vec_posClip.end(), d_vec_posScreen.begin(), d_vec_ret.begin(),
		functor_Unproject(invMV, invPJ, winW, winH));
	thrust::copy(d_vec_ret.begin(), d_vec_ret.end(), ret);
}

void Displace::AddSphereLens(int x, int y, int radius, float3 center)
{
	Lens* l = new CircleLens(x, y, radius, center);
	lenses.push_back(l);
}
