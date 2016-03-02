#include <Displace.h>
#include <TransformFunc.h>
#include <vector_functions.h>
#include <helper_math.h>
#include <Lens.h>
#include <math_constants.h>
#include <thrust/extrema.h>


//when using thrust::device_vector instead of thrust::device_vector,
//the performance does not reduce much.

__device__ __host__ inline float4 Float3ToFloat4(float3 v)
{
	return make_float4(v.x, v.y, v.z, 1.0f);
}

__device__ __host__ inline float G(float x, float r)
{
	return pow((r - 1), 2) / (-r * r * x + r) + 2 - 1 / r;
}

__device__ __host__ inline float G_Diff(float x, float r)
{
	return pow((r - 1)/ (r * x - 1), 2);
}

__device__ __host__ float2 DisplaceCircleLens(float lensX, float lensY, float lensR, float2 screenPos, float& glyphSize, float r)//, float rSide = 0)
{
	float2 ret = screenPos;
	float2 dir = screenPos - make_float2(lensX, lensY);
	float disOrig = length(dir);
	float rOut = (lensR) / r; //including the focus and transition region
	if (disOrig < rOut) {
		float disNew = G(disOrig / rOut, r) * rOut;
		ret = make_float2(lensX, lensY) + dir / disOrig * disNew;
		glyphSize = G_Diff(disOrig / rOut, r);
	}
	return ret;
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
	__device__ __host__ float2 operator() (float4 p)
	{
		return Clip2ScreenGlobal(GetXY(p), w, h);
	}
	functor_Clip2Screen(int _w, int _h) :w(_w), h(_h){}
};

struct functor_Displace
{
	int lensX, lensY, circleR;
	float lensD;
	float focusRatio;
	template<typename Tuple>
	__device__ __host__ void operator() (Tuple t){//float2 screenPos, float4 clipPos) {
		float2 screenPos = thrust::get<0>(t);
		float4 clipPos = thrust::get<1>(t);
		float2 newScreenPos = screenPos;
		float brightness = 1.0f;
		float2 lensCen = make_float2(lensX, lensY);
		float2 vec = screenPos - lensCen;
		float dis2Cen = length(vec);
		const float thickDisp = 0.01;// 0.003;
		const float thickFocus = 0.01;// 0.003;
		const float dark = 0.05;
		float outerR = circleR / focusRatio;
		int myCase = 0;
		if (dis2Cen < outerR){
			if ((lensD - clipPos.z) > thickDisp){
				myCase = 1;//cutaway
			} else if (clipPos.z < lensD){
				myCase = 2;//displace
			} else if (dis2Cen > circleR){
				myCase = 3;//turn dark
			} else if ((clipPos.z - lensD) > thickFocus) {
				myCase = 4;//graduately turn dark
			} 
		}
		float2 dir = normalize(vec);
		switch (myCase){
		case 1:
			newScreenPos = dir * outerR + lensCen;
			break;
		case 2:
			newScreenPos = lensCen + dir * G(dis2Cen / outerR, focusRatio) * outerR;
			break;
		case 3:
			brightness = dark;
			break;
		case 4:
			brightness = max(dark, 1.0 / (1000 * (clipPos.z - lensD - thickFocus) + 1.0));
			break;
		}
		thrust::get<0>(t) = newScreenPos;
		thrust::get<3>(t) = brightness;
	}
	functor_Displace(int _lensX, int _lensY, int _circleR, float _lensD, float _focusRatio)
		: lensX(_lensX), lensY(_lensY), circleR(_circleR), lensD(_lensD), focusRatio(_focusRatio){}
};

struct functor_Displace_Line
{
	int x, y;
	float d;

	float lSemiMajorAxis, lSemiMinorAxis;
	float2 direction;

	__device__ __host__ float2 operator() (float2 screenPos, float4 clipPos) {
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


struct functor_Displace_NotFinish //no deformation when the lens construction is not finished
{
	__device__ float2 operator() (float2 screenPos, float4 clipPos) {
		float2 ret = screenPos;
		return ret;
	}
	functor_Displace_NotFinish(){}
};

struct functor_Displace_CurveB
{
	int x, y;
	CurveBLensInfo curveBLensInfo;
	float d;
	//CurveLensCtrlPoints curveLensCtrlPoints;

	template<typename Tuple>
	__device__ __host__ void operator() (Tuple t){
		float2 screenCoord = thrust::get<0>(t);
		float4 clipPos = thrust::get<1>(t);

		float2 ret = screenCoord;
		float brightness = 1.0f;


		//we may be able to use BezierPoints for in lens detection, for a better speed
		int numBezierPoints = curveBLensInfo.numBezierPoints;
		float2* BezierPoints = curveBLensInfo.BezierPoints;
		float2* BezierNormals = curveBLensInfo.BezierNormals;

		int numPosPoints = curveBLensInfo.numPosPoints;
		float2* subCtrlPointsPos = curveBLensInfo.subCtrlPointsPos;
		float2* posOffsetCtrlPoints = curveBLensInfo.posOffsetCtrlPoints;

		int numNegPoints = curveBLensInfo.numNegPoints;
		float2* subCtrlPointsNeg = curveBLensInfo.subCtrlPointsNeg;
		float2* negOffsetCtrlPoints = curveBLensInfo.negOffsetCtrlPoints;

		float width = curveBLensInfo.width;
		float ratio = curveBLensInfo.focusRatio;
		float rOut = width / ratio;

		//possible difference of the numPosPoints and numNegPoints makes the positive half and the negative half region do npt cover the whole lens region, according to current method
		const float DifResAdjust = 0.05;

		float2 center = make_float2(x, y);

		bool segmentNotFoundPos = true;
		int keySegmentId = -1;
		for (int ii = 0; ii < numPosPoints - 1 && segmentNotFoundPos; ii++) {
			float2 toPoint = screenCoord - (center + subCtrlPointsPos[ii]);
			float2 dir = normalize(subCtrlPointsPos[ii + 1] - subCtrlPointsPos[ii]);
			float2 minorDir = make_float2(-dir.y, dir.x);
			float disMinor = toPoint.x*minorDir.x + toPoint.y*minorDir.y;
			if (disMinor < rOut && (disMinor >= 0 || (numPosPoints < numNegPoints && disMinor >= -rOut*DifResAdjust))){
				float2 ctrlPointAbsolute1 = center + subCtrlPointsPos[ii];
				float2 ctrlPointAbsolute2 = center + subCtrlPointsPos[ii + 1];

				float2 normal1 = normalize(posOffsetCtrlPoints[ii] - subCtrlPointsPos[ii]);
				float2 normal2 = normalize(posOffsetCtrlPoints[ii + 1] - subCtrlPointsPos[ii + 1]);

				//first check if screenCoord and ctrlPointAbsolute2 are at the same side of Line (ctrlPointAbsolute1, normals[ii])
				//then check if screenCoord and ctrlPointAbsolute1 are at the same side of Line (ctrlPointAbsolute2, normals[ii+1])

				if (((screenCoord.x - ctrlPointAbsolute1.x)*normal1.y - (screenCoord.y - ctrlPointAbsolute1.y)*normal1.x)
					*((ctrlPointAbsolute2.x - ctrlPointAbsolute1.x)*normal1.y - (ctrlPointAbsolute2.y - ctrlPointAbsolute1.y)*normal1.x)
					>= 0) {
					if (((screenCoord.x - ctrlPointAbsolute2.x)*normal2.y - (screenCoord.y - ctrlPointAbsolute2.y)*normal2.x)
						*((ctrlPointAbsolute1.x - ctrlPointAbsolute2.x)*normal2.y - (ctrlPointAbsolute1.y - ctrlPointAbsolute2.y)*normal2.x)
						>= 0) {
						segmentNotFoundPos = false;
						keySegmentId = ii;

						if (clipPos.z < d)
						{
							float sin1 = dir.x*normal1.y - dir.y*normal1.x;//sin of the angle of dir x normals[ii]
							float sin2 = dir.x*normal2.y - dir.y*normal2.x;//sin of the angle of dir x normals[ii+1]

							float disMinorNewAbs = G(abs(disMinor) / rOut, ratio) * rOut;
							float2 intersectLeftOri = ctrlPointAbsolute1 + normal1 * (disMinor / sin1);
							float2 intersectRightOri = ctrlPointAbsolute2 + normal2 * (disMinor / sin2);
							float posRatio = length(screenCoord - intersectLeftOri) / length(intersectRightOri - intersectLeftOri);
							float2 intersectLeft = ctrlPointAbsolute1 + normal1 * (disMinorNewAbs / sin1);
							float2 intersectRight = ctrlPointAbsolute2 + normal2 * (disMinorNewAbs / sin2);
							ret = posRatio*intersectRight + (1 - posRatio)*intersectLeft;
						}
						else{
							brightness = clamp(1.3f - 300 * abs(clipPos.z - d), 0.1f, 1.0f);
						}
					}
				}
			}
		}
		if (segmentNotFoundPos){
			bool segmentNotFoundNeg = true;
			int keySegmentId = -1;
			for (int ii = 0; ii < numNegPoints - 1 && segmentNotFoundNeg; ii++) {
				float2 toPoint = screenCoord - (center + subCtrlPointsNeg[ii]);
				float2 dir = normalize(subCtrlPointsNeg[ii + 1] - subCtrlPointsNeg[ii]);
				float2 minorDir = make_float2(-dir.y, dir.x);
				float disMinor = toPoint.x*minorDir.x + toPoint.y*minorDir.y;
				if (disMinor >-rOut && (disMinor <= 0 || (numPosPoints > numNegPoints && disMinor <= rOut*DifResAdjust))){
					float2 ctrlPointAbsolute1 = center + subCtrlPointsNeg[ii];
					float2 ctrlPointAbsolute2 = center + subCtrlPointsNeg[ii + 1];

					float2 normal1 = normalize(negOffsetCtrlPoints[ii] - subCtrlPointsNeg[ii]);
					float2 normal2 = normalize(negOffsetCtrlPoints[ii + 1] - subCtrlPointsNeg[ii + 1]);

					//first check if screenCoord and ctrlPointAbsolute2 are at the same side of Line (ctrlPointAbsolute1, normals[ii])
					//then check if screenCoord and ctrlPointAbsolute1 are at the same side of Line (ctrlPointAbsolute2, normals[ii+1])

					if (((screenCoord.x - ctrlPointAbsolute1.x)*normal1.y - (screenCoord.y - ctrlPointAbsolute1.y)*normal1.x)
						*((ctrlPointAbsolute2.x - ctrlPointAbsolute1.x)*normal1.y - (ctrlPointAbsolute2.y - ctrlPointAbsolute1.y)*normal1.x)
						>= 0) {
						if (((screenCoord.x - ctrlPointAbsolute2.x)*normal2.y - (screenCoord.y - ctrlPointAbsolute2.y)*normal2.x)
							*((ctrlPointAbsolute1.x - ctrlPointAbsolute2.x)*normal2.y - (ctrlPointAbsolute1.y - ctrlPointAbsolute2.y)*normal2.x)
							>= 0) {
							segmentNotFoundNeg = false;
							keySegmentId = ii;

							if (clipPos.z < d)
							{
								float sin1 = dir.x*normal1.y - dir.y*normal1.x;//sin of the angle of dir x normals[ii]
								float sin2 = dir.x*normal2.y - dir.y*normal2.x;//sin of the angle of dir x normals[ii+1]

								float disMinorNewAbs = G(abs(disMinor) / rOut, ratio) * rOut;
								float2 intersectLeftOri = ctrlPointAbsolute1 + normal1 * (disMinor / sin1);
								float2 intersectRightOri = ctrlPointAbsolute2 + normal2 * (disMinor / sin2);
								float posRatio = length(screenCoord - intersectLeftOri) / length(intersectRightOri - intersectLeftOri);
								float2 intersectLeft = ctrlPointAbsolute1 - normal1 * (disMinorNewAbs / sin1);
								float2 intersectRight = ctrlPointAbsolute2 - normal2 * (disMinorNewAbs / sin2);
								ret = posRatio*intersectRight + (1 - posRatio)*intersectLeft;
							}
							else{
								brightness = clamp(1.3f - 300 * abs(clipPos.z - d), 0.1f, 1.0f);
							}
						}
					}
				}
			}

		}



		thrust::get<0>(t) = ret;
		thrust::get<3>(t) = brightness;
	}
	functor_Displace_CurveB(int _x, int _y, CurveBLensInfo _curveBLensInfo, float _d) :
		x(_x), y(_y), curveBLensInfo(_curveBLensInfo), d(_d){}
};


struct functor_ApproachTarget
{

	template<typename Tuple>
	__device__ __host__ void operator() (Tuple t) {
		float3 screenPos = make_float3(thrust::get<0>(t));
		float3 screenTarget = make_float3(thrust::get<1>(t));
		float3 dir = screenTarget - screenPos;
		if (length(dir) < 0.01) {
			thrust::get<0>(t) = Float3ToFloat4(screenTarget);
			thrust::get<2>(t) = thrust::get<3>(t);
		}
		else{
			thrust::get<0>(t) = Float3ToFloat4(screenPos + dir * 0.1);
			thrust::get<2>(t) = thrust::get<2>(t) +(thrust::get<3>(t) -thrust::get<2>(t)) * 0.1;
		}
		thrust::get<4>(t) = thrust::get<4>(t) +(thrust::get<5>(t) -thrust::get<4>(t)) * 0.1;
	}
};

struct functor_Unproject
{
	matrix4x4 inv_mv, inv_pj;
	int w, h;
	__device__ __host__ float4 operator() (float4 pClip, float2 pScreen)
	{
		float2 clip = Screen2Clip(pScreen, w, h);
		float4 clip2 = make_float4(clip.x, clip.y, pClip.z, pClip.w);
		return Clip2ObjectGlobal(clip2, &inv_mv.v[0].x, &inv_pj.v[0].x);
	}
	functor_Unproject(matrix4x4 _inv_mv, matrix4x4 _inv_pj, int _w, int _h) :
		inv_mv(_inv_mv), inv_pj(_inv_pj), w(_w), h(_h){}
};

Displace::Displace()
{
}

void Displace::LoadOrig(float4* v, int num)
{
	posOrig.assign(v, v + num);// , posOrig.begin());
	d_vec_posTarget.assign(num, make_float4(0, 0, 0, 1));
	d_vec_glyphSizeTarget.assign(num, 1);
	d_vec_glyphBrightTarget.assign(num, 1.0f);
	d_vec_disToAim.assign(num, 0);
}

void Displace::DisplacePoints(std::vector<float2>& pts, std::vector<Lens*> lenses
	, float* modelview, float* projection, int winW, int winH)
{
	for (int i = 0; i < lenses.size(); i++) {
		CircleLens* l = (CircleLens*)lenses[i];
		for (auto& p : pts) {
			float tmp = 1;
			float2 center = l->GetScreenPos(modelview, projection, winW, winH);
			p = DisplaceCircleLens(center.x, center.y, l->radius, p, tmp, l->focusRatio);
		}
	}
}


void Displace::Compute(float* modelview, float* projection, int winW, int winH,
	std::vector<Lens*> lenses, float4* ret, float* glyphSizeScale, float* glyphBright)
{
	int size = posOrig.size();
	matrix4x4 mv(modelview);
	matrix4x4 pj(projection);

	if (recomputeTarget) {
		thrust::device_vector<float4> d_vec_posClip(size);
		thrust::device_vector<float2> d_vec_posScreen(size);

		thrust::transform(posOrig.begin(), posOrig.end(), d_vec_posClip.begin(), functor_Object2Clip(mv, pj));
		thrust::transform(d_vec_posClip.begin(), d_vec_posClip.end(),
			d_vec_posScreen.begin(), functor_Clip2Screen(winW, winH));
		//reset to 1
		d_vec_glyphSizeTarget.assign(size, 1);

		//use this for the case that there is no lens, 
		//and the glyphs go back to the original positions
		if (lenses.size() < 1){
			d_vec_posTarget = posOrig;
		}
		else {
			for (int i = 0; i < lenses.size(); i++) {
				float2 lensScreenCenter = lenses[i]->GetScreenPos(modelview, projection, winW, winH);
				switch (lenses[i]->GetType()) {
					case LENS_TYPE::TYPE_CIRCLE:
					{
						CircleLens* l = (CircleLens*)lenses[i];
						thrust::for_each(
							thrust::make_zip_iterator(
							thrust::make_tuple(
							d_vec_posScreen.begin(),
							d_vec_posClip.begin(),
							d_vec_glyphSizeTarget.begin(),
							d_vec_glyphBrightTarget.begin()
							)),
							thrust::make_zip_iterator(
							thrust::make_tuple(
							d_vec_posScreen.end(),
							d_vec_posClip.end(),
							d_vec_glyphSizeTarget.end(),
							d_vec_glyphBrightTarget.end()
							)),
							functor_Displace(lensScreenCenter.x, lensScreenCenter.y, l->radius, l->GetClipDepth(modelview, projection), l->focusRatio));
						break;

					}
					case LENS_TYPE::TYPE_LINE:
					{
						LineLens* l = (LineLens*)lenses[i];
						thrust::transform(d_vec_posScreen.begin(), d_vec_posScreen.end(),
							d_vec_posClip.begin(), d_vec_posScreen.begin(),
							functor_Displace_Line(lensScreenCenter.x, lensScreenCenter.y, l->lSemiMajorAxis, l->lSemiMinorAxis, l->direction, l->GetClipDepth(modelview, projection)));
						break;
					}
					case LENS_TYPE::TYPE_CURVEB:
					{
						CurveBLens* l = (CurveBLens*)lenses[i];
						if (l->isConstructing){
							thrust::transform(d_vec_posScreen.begin(), d_vec_posScreen.end(),
								d_vec_posClip.begin(), d_vec_posScreen.begin(),
								functor_Displace_NotFinish());
						}
						else{
							thrust::for_each(
								thrust::make_zip_iterator(
								thrust::make_tuple(
								d_vec_posScreen.begin(),
								d_vec_posClip.begin(),
								d_vec_glyphSizeTarget.begin(),
								d_vec_glyphBrightTarget.begin()
								)),
								thrust::make_zip_iterator(
								thrust::make_tuple(
								d_vec_posScreen.end(),
								d_vec_posClip.end(),
								d_vec_glyphSizeTarget.end(),
								d_vec_glyphBrightTarget.end()
								)),
								functor_Displace_CurveB(lensScreenCenter.x, lensScreenCenter.y, l->curveBLensInfo, l->GetClipDepth(modelview, projection)));

						}
						break;
					}
				}
			}
			matrix4x4 invMV;
			matrix4x4 invPJ;
			invertMatrix(&mv.v[0].x, &invMV.v[0].x);
			invertMatrix(&pj.v[0].x, &invPJ.v[0].x);
			thrust::transform(d_vec_posClip.begin(), d_vec_posClip.end(), d_vec_posScreen.begin(), d_vec_posTarget.begin(),
				functor_Unproject(invMV, invPJ, winW, winH));
		}
		recomputeTarget = false;
	}

	thrust::device_vector<float4> d_vec_posCur(size);
	thrust::copy(ret, ret + size, d_vec_posCur.begin());
	thrust::device_vector<float> d_vec_glyphSizeScale(size);
	thrust::copy(glyphSizeScale, glyphSizeScale + size, d_vec_glyphSizeScale.begin());
	thrust::device_vector<float> d_vec_glyphBright(size);
	thrust::copy(glyphBright, glyphBright + size, d_vec_glyphBright.begin());
	
	thrust::for_each(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		d_vec_posCur.begin(),
		d_vec_posTarget.begin(), 
		d_vec_glyphSizeScale.begin(),
		d_vec_glyphSizeTarget.begin(),
		d_vec_glyphBright.begin(),
		d_vec_glyphBrightTarget.begin()
		)),
		thrust::make_zip_iterator(
		thrust::make_tuple(
		d_vec_posCur.end(),
		d_vec_posTarget.end(),
		d_vec_glyphSizeScale.end(),
		d_vec_glyphSizeTarget.end(),
		d_vec_glyphBright.end(),
		d_vec_glyphBrightTarget.end()
		)),
		functor_ApproachTarget());
	thrust::copy(d_vec_posCur.begin(), d_vec_posCur.end(), ret);
	thrust::copy(d_vec_glyphSizeScale.begin(), d_vec_glyphSizeScale.end(), glyphSizeScale);
	thrust::copy(d_vec_glyphBright.begin(), d_vec_glyphBright.end(), glyphBright);
}


struct disToAim_functor 
{ 
	const float3 aim;

	disToAim_functor(float3 _aim) : aim(_aim) {}

	__host__ __device__ float operator()(const float4 & x, const float & y) const 
	{
		return length(aim-make_float3(x)); 
	} 
};


float3 Displace::findClosetGlyph(float3 aim, int & snappedGlyphId)
{
	thrust::transform(posOrig.begin(), posOrig.end(), d_vec_disToAim.begin(), d_vec_disToAim.begin(), disToAim_functor(aim));
	snappedGlyphId = thrust::min_element(d_vec_disToAim.begin(), d_vec_disToAim.end()) - d_vec_disToAim.begin();
	return make_float3(posOrig[snappedGlyphId]);
}
