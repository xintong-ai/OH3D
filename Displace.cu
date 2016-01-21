#include <Displace.h>
#include <TransformFunc.h>

#include <vector_functions.h>
#include <helper_math.h>
#include <Lens.h>

void Displace::LoadOrig(float4* v, int num)
{
	posOrig.assign(v, v + num);// , posOrig.begin());
	d_vec_posScreenTarget.assign(num, make_float2(0, 0));
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

__device__ inline float G(float x, float r)
{
	return pow((r - 1), 2) / (-r * r * x + r) + 2 - 1 / r;
}

struct functor_Displace
{
	int x, y, r;
	float d;
	__device__ float2 operator() (float2 screenPos, float4 clipPos) {
		float2 ret = screenPos;

		if (clipPos.z < d) {
			float2 dir = screenPos - make_float2(x, y);
			float disOrig = length(dir);
			float ratio = 0.5;
			float rOut = r / ratio; //including the focus and transition region
			if (disOrig < rOut) {
				float disNew = G(disOrig / rOut, ratio) * rOut;
				ret = make_float2(x, y) + dir / disOrig * disNew;
			}
		}
		return ret;
	}
	functor_Displace(int _x, int _y, int _r, float _d) : x(_x), y(_y), r(_r), d(_d){}
};

struct functor_Displace_Line
{
	int x, y;
	float d;

	float lSemiMajorAxis, lSemiMinorAxis;
	float2 direction;

	__device__ float2 operator() (float2 screenPos, float4 clipPos) {
		float2 ret = screenPos;

		if (clipPos.z < d) {
			//sigmoid function: y=2*(1/(1+e^(-20*(x+1)))-0.5), x in [-1,0]
			//sigmoid function: y=2*(1/(1+e^(20*(x-1)))-0.5), x in [0,1]

			//dot product of (_x-x, _y-y) and direction

			float2 toPoint = screenPos - make_float2(x, y);
			float disMajor = toPoint.x*direction.x + toPoint.y*direction.y;
			if (abs(disMajor) < lSemiMajorAxis) {

				float2 minorDirection = make_float2(-direction.y, direction.x);
				//dot product of (_x-x, _y-y) and minorDirection
				float disMinor = toPoint.x*minorDirection.x + toPoint.y*minorDirection.y;


				float disMajorRatio = disMajor / lSemiMajorAxis;
				float disSigmoid; //always positive or 0
				if (disMajorRatio < 0){
					disSigmoid = 1 / (1 + exp(-40 * (disMajorRatio + 0.8)));
				}
				else {
					disSigmoid = 1 / (1 + exp(40 * (disMajorRatio - 0.8)));
				}

				float ratio = 0.5;
				if (abs(disMinor) < disSigmoid*lSemiMinorAxis / ratio){			
					float rOut = disSigmoid *lSemiMinorAxis / ratio; //including the focus and transition region

					float disMinorNewAbs = G(abs(disMinor) / rOut, ratio) * rOut;
					if (disMinor > 0){
						ret = make_float2(screenPos.x, screenPos.y) + minorDirection * (disMinorNewAbs - disMinor);
					}
					else {
						ret = make_float2(screenPos.x, screenPos.y) - minorDirection * (disMinorNewAbs + disMinor);
					}
				}
			}
		}
		return ret;
	}
	functor_Displace_Line(int _x, int _y, int _lSemiMajorAxis, int _lSemiMinorAxis, float2 _direction, float _d) :
		x(_x), y(_y), lSemiMajorAxis(_lSemiMajorAxis), lSemiMinorAxis(_lSemiMinorAxis), direction(_direction), d(_d){}
};


//thrust::transform(d_vec_posScreen.begin(), d_vec_posScreen.end(),
//	d_vec_posScreenTarget.begin(), d_vec_posScreen.begin(),
//	functor_ApproachTarget());

struct functor_ApproachTarget
{
	__device__ float2 operator() (float2 screenPos, float2 screenTarget) {
		float2 dir = screenTarget - screenPos;
		//float dis = length(dir);
		return screenPos + dir * 0.1;
	}

	functor_ApproachTarget(){}
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

	if (recomputeTarget) {
		thrust::transform(posOrig.begin(), posOrig.end(), d_vec_posClip.begin(), functor_Object2Clip(mv, pj));

		thrust::transform(d_vec_posClip.begin(), d_vec_posClip.end(),
			d_vec_posScreen.begin(), functor_Clip2Screen(winW, winH));

		for (int i = 0; i < lenses.size(); i++) {
			/*
			CircleLens* l = (CircleLens*)lenses[i];
			thrust::transform(d_vec_posScreen.begin(), d_vec_posScreen.end(),
				d_vec_posClip.begin(), d_vec_posScreenTarget.begin(),
				functor_Displace(l->x, l->y, l->radius, l->GetClipDepth(modelview, projection)));
				*/
			LineLens* l = (LineLens*)lenses[i];
			thrust::transform(d_vec_posScreen.begin(), d_vec_posScreen.end(),
				d_vec_posClip.begin(), d_vec_posScreenTarget.begin(),
				functor_Displace_Line(l->x, l->y, l->lSemiMajorAxis, l->lSemiMinorAxis, l->direction, l->GetClipDepth(modelview, projection)));
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
