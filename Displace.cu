#include <Displace.h>
#include <TransformFunc.h>
#include <vector_functions.h>
#include <helper_math.h>
#include <Lens.h>
#include <math_constants.h>

void Displace::LoadOrig(float4* v, int num)
{
	posOrig.assign(v, v + num);// , posOrig.begin());
	d_vec_posScreenTarget.assign(num, make_float2(0, 0));
	//d_vec_Dist2LensBtm.assign(num, 0);
}

struct functor_Object2Clip//: public thrust::unary_function<float,float>
{
	__device__ __host__ float4 operator() (float4 p)
	{
		return Object2Clip(p, &mv.v[0].x, &pj.v[0].x);
	}
	matrix4x4 mv, pj;
	functor_Object2Clip(matrix4x4 _mv, matrix4x4 _pj) : mv(_mv), pj(_pj){}
};

struct functor_Clip2Screen
{
	int w, h;
	__device__ float2 operator() (float4 p)
	{
		return Clip2ScreenGlobal(GetXY(p), w, h);
	}
	functor_Clip2Screen(int _w, int _h) :w(_w), h(_h){}
};

__device__ inline float G(float x, float r)
{
	return pow((r - 1), 2) / (-r * r * x + r) + 2 - 1 / r;
}

struct functor_Displace
{
	int x, y, r;
	float d;
	template<typename Tuple>
	__device__ void operator() (Tuple t){//float2 screenPos, float4 clipPos) {
		float2 screenPos = thrust::get<0>(t);
		float4 clipPos = thrust::get<1>(t);
		float sideSize = 1.0;
		float2 ret = screenPos;
		if (clipPos.z < d) {
			float2 dir = screenPos - make_float2(x, y);
			float disOrig = length(dir);
			float ratio = 0.5;
			float rOut = (r + (d - clipPos.z) * r * 32 * sideSize) / ratio; //including the focus and transition region
			if (disOrig < rOut) {
				float disNew = G(disOrig / rOut, ratio) * rOut;
				ret = make_float2(x, y) + dir / disOrig * disNew;
			}
		}
		thrust::get<2>(t) = ret;
	}
	functor_Displace(int _x, int _y, int _r, float _d) : x(_x), y(_y), r(_r), d(_d){}
};

struct functor_ApproachTarget
{
	__device__ float2 operator() (float2 screenPos, float2 screenTarget) {
		float2 dir = screenTarget - screenPos;
		if (length(dir) < 0.5) 
			return screenTarget;
		else
			return screenPos + dir * 0.1;
	}
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
	functor_Unproject(matrix4x4 _inv_mv, matrix4x4 _inv_pj, int _w, int _h) :
		inv_mv(_inv_mv), inv_pj(_inv_pj), w(_w), h(_h){}
};

//struct func_CompDist2LensBtm{
//	float3 c;
//	matrix4x4 mv;
//	__device__ float operator() (float4 pObj){
//		float4 a = mat4mulvec4(&(mv.v[0].x), make_float4(c.x, c.y, c.z, 1.0f));
//		float4 b = mat4mulvec4(&(mv.v[0].x), pObj);
//		return abs(a.z / a.w - b.z / b.w);//projection * modelview * v;
//	}
//	func_CompDist2LensBtm(float3 _c, matrix4x4 _mv) : c(_c), mv(_mv){}
//};
//thrust::transform(posOrig.begin(), posOrig.end(), d_vec_Dist2LensBtm.begin(), (l->c, modelview));


void Displace::Compute(float* modelview, float* projection, int winW, int winH,
	std::vector<Lens*> lenses, float4* ret)
{
	if (lenses.size() <= 0)
		return;
	int size = posOrig.size();

	//clip coordiates of streamlines
	matrix4x4 mv(modelview);
	matrix4x4 pj(projection);

	thrust::device_vector<float4> d_vec_posClip(size);
	thrust::device_vector<float2> d_vec_posScreen(size);
	//thrust::counting_iterator < int > first(0);

	if (recomputeTarget) {
		thrust::transform(posOrig.begin(), posOrig.end(), d_vec_posClip.begin(), functor_Object2Clip(mv, pj));

		thrust::transform(d_vec_posClip.begin(), d_vec_posClip.end(),
			d_vec_posScreen.begin(), functor_Clip2Screen(winW, winH));


		for (int i = 0; i < lenses.size(); i++) {
			CircleLens* l = (CircleLens*)lenses[i];
			//thrust::transform(posOrig.begin(), posOrig.end(), 
			//	d_vec_Dist2LensBtm.begin(), func_CompDist2LensBtm(l->c, mv));
			thrust::for_each(
				thrust::make_zip_iterator(
				thrust::make_tuple(
				d_vec_posScreen.begin(), 
				d_vec_posClip.begin(), 
				d_vec_posScreenTarget.begin()
				)),
				thrust::make_zip_iterator(
				thrust::make_tuple(
				d_vec_posScreen.end(),
				d_vec_posClip.end(),
				d_vec_posScreenTarget.end()
				)),
				functor_Displace(l->x, l->y, l->radius, l->GetClipDepth(modelview, projection)));
		}
		

		recomputeTarget = false;
	}



	thrust::device_vector<float4> d_vec_posCur(size);
	thrust::copy(ret, ret + size, d_vec_posCur.begin());

	thrust::transform(d_vec_posCur.begin(), d_vec_posCur.end(), d_vec_posClip.begin(), functor_Object2Clip(mv, pj));
	thrust::transform(d_vec_posClip.begin(), d_vec_posClip.end(),
		d_vec_posScreen.begin(), functor_Clip2Screen(winW, winH));

	thrust::transform(d_vec_posScreen.begin(), d_vec_posScreen.end(),
		d_vec_posScreenTarget.begin(), d_vec_posScreen.begin(),
		functor_ApproachTarget());

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
