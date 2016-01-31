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
	d_vec_glyphSizeTarget.assign(num, 1);
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

__device__ __host__ inline float G(float x, float r)
{
	return pow((r - 1), 2) / (-r * r * x + r) + 2 - 1 / r;
}

__device__ __host__ inline float G_Diff(float x, float r)
{
	return pow((r - 1)/ (r * x - 1), 2);
}

__device__ __host__ float2 DisplaceCircleLens(float x, float y, float r, float2 screenPos, float& glyphSize, float focusRatio, float rSide = 0)
{
	float2 ret = screenPos;
	float2 dir = screenPos - make_float2(x, y);
	float disOrig = length(dir);
	float rOut = (r + rSide) / focusRatio; //including the focus and transition region
	if (disOrig < rOut) {
		float disNew = G(disOrig / rOut, focusRatio) * rOut;
		ret = make_float2(x, y) + dir / disOrig * disNew;
		glyphSize = G_Diff(disOrig / rOut, focusRatio);
	}
	return ret;
}

struct functor_Displace
{
	int x, y, r;
	float d;
	float focusRatio;
	float sideSize;
	template<typename Tuple>
	__device__ __host__ void operator() (Tuple t){//float2 screenPos, float4 clipPos) {
		float2 screenPos = thrust::get<0>(t);
		float4 clipPos = thrust::get<1>(t);
		float2 ret = screenPos;
		if (clipPos.z < d) {
			float glyphSize = 1;
			ret = DisplaceCircleLens(x, y, r, screenPos, glyphSize, focusRatio, (d - clipPos.z) * r * 64 * sideSize);
			thrust::get<3>(t) = glyphSize;
		}
		thrust::get<2>(t) = ret;
	}
	functor_Displace(int _x, int _y, int _r, float _d, float _focusRatio, float _sideSize) 
		: x(_x), y(_y), r(_r), d(_d), focusRatio(_focusRatio), sideSize(_sideSize){}
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

			float2 toPoint = screenPos - make_float2(x, y);
			
			//dot product of toPoint and direction
			float disMajor = toPoint.x*direction.x + toPoint.y*direction.y;
			if (abs(disMajor) < lSemiMajorAxis) {

				float2 minorDirection = make_float2(-direction.y, direction.x);
				//dot product of toPoint and minorDirection
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

struct functor_Displace_PolyLine
{
	int x, y;
	float d;
	PolyLineLensCtrlPoints polyLineLensCtrlPoints;

	float width;

	float lSemiMajor, lSemiMinor;
	float2 direction;

	__device__ float2 operator() (float2 screenPos, float4 clipPos) {
		float2 ret = screenPos;

		if (clipPos.z < d) {
			int numCtrlPoints = polyLineLensCtrlPoints.numCtrlPoints;
			float2* ctrlPoints = polyLineLensCtrlPoints.ctrlPoints;
			float2* dirs = polyLineLensCtrlPoints.dirs;
			float2* angleBisectors = polyLineLensCtrlPoints.angleBisectors;

			float ratio = 0.5;

			bool segmentNotFound = true;
			for (int ii = 0; ii < numCtrlPoints - 1 && segmentNotFound; ii++) {
				float2 center = make_float2(x, y);
				float2 toPoint = screenPos - (center + ctrlPoints[ii]);
				float2 dir = dirs[ii];
				float2 minorDir = make_float2(-dir.y, dir.x);
				float disMinor = toPoint.x*minorDir.x + toPoint.y*minorDir.y;
				if (abs(disMinor) < width / ratio)	{
					float2 ctrlPointAbsolute1 = center + ctrlPoints[ii];
					float2 ctrlPointAbsolute2 = center + ctrlPoints[ii+1];

					//first check if screenPos and ctrlPointAbsolute2 are at the same side of Line (ctrlPointAbsolute1, angleBisectors[ii])
					//then check if screenPos and ctrlPointAbsolute1 are at the same side of Line (ctrlPointAbsolute2, angleBisectors[ii+1])

					if (((screenPos.x - ctrlPointAbsolute1.x)*angleBisectors[ii].y - (screenPos.y - ctrlPointAbsolute1.y)*angleBisectors[ii].x)
						*((ctrlPointAbsolute2.x - ctrlPointAbsolute1.x)*angleBisectors[ii].y - (ctrlPointAbsolute2.y - ctrlPointAbsolute1.y)*angleBisectors[ii].x)
						>= 0) {
						if (((screenPos.x - ctrlPointAbsolute2.x)*angleBisectors[ii + 1].y - (screenPos.y - ctrlPointAbsolute2.y)*angleBisectors[ii + 1].x)
							*((ctrlPointAbsolute1.x - ctrlPointAbsolute2.x)*angleBisectors[ii + 1].y - (ctrlPointAbsolute1.y - ctrlPointAbsolute2.y)*angleBisectors[ii + 1].x)
							>= 0) {

							float sin1 = dir.x*angleBisectors[ii].y - dir.y*angleBisectors[ii].x;//sin of the angle of dir x angleBisectors[ii]
							float sin2 = dir.x*angleBisectors[ii + 1].y - dir.y*angleBisectors[ii + 1].x;//sin of the angle of dir x angleBisectors[ii+1]

							float rOut = width / ratio;
							float disMinorNewAbs = G(abs(disMinor) / rOut, ratio) * rOut;
							

							float2 intersectLeftOri = ctrlPointAbsolute1 + angleBisectors[ii] * (disMinor / sin1);
							float2 intersectRightOri = ctrlPointAbsolute2 + angleBisectors[ii + 1] * (disMinor / sin2);
							float posRatio = length(screenPos - intersectLeftOri) / length(intersectRightOri - intersectLeftOri);
							if (disMinor >= 0){
								float2 intersectLeft = ctrlPointAbsolute1 + angleBisectors[ii] * (disMinorNewAbs / sin1);
								float2 intersectRight = ctrlPointAbsolute2 + angleBisectors[ii + 1] * (disMinorNewAbs / sin2);
								ret = posRatio*intersectRight + (1 - posRatio)*intersectLeft;
							}
							else {
								float2 intersectLeft = ctrlPointAbsolute1 - angleBisectors[ii] * (disMinorNewAbs / sin1);
								float2 intersectRight = ctrlPointAbsolute2 - angleBisectors[ii + 1] * (disMinorNewAbs / sin2);
								ret = posRatio*intersectRight + (1 - posRatio)*intersectLeft;
							}

							segmentNotFound = false;
						}
					}

				}
			}

		}
		return ret;
	}

	functor_Displace_PolyLine(int _x, int _y, int _width, PolyLineLensCtrlPoints _polyLineLensCtrlPoints, float2 _direction, float _lSemiMajor, float _lSemiMinor, float _d) :
		x(_x), y(_y), width(_width), polyLineLensCtrlPoints(_polyLineLensCtrlPoints), direction(_direction), lSemiMajor(_lSemiMajor), lSemiMinor(_lSemiMinor), d(_d){}

};



struct functor_Displace_Curve
{
	int x, y;
	float d;
	CurveLensCtrlPoints curveLensCtrlPoints;
	float width;

	__device__ float2 operator() (float2 screenPos, float4 clipPos) {
		float2 ret = screenPos;

		
		if (clipPos.z < d) {
			int numCtrlPoints = curveLensCtrlPoints.numCtrlPoints;
			float2* ctrlPoints = curveLensCtrlPoints.ctrlPoints;

			float2* normals = curveLensCtrlPoints.normals;
			int numKeyPoints = curveLensCtrlPoints.numKeyPoints;
			float2* keyPoints = curveLensCtrlPoints.keyPoints;
			int* keyPointIds = curveLensCtrlPoints.keyPointIds;
			float ratio = curveLensCtrlPoints.ratio;

			float rOut = width / ratio;

			bool segmentNotFound = true;
			int keySegmentId = -1;
			for (int ii = 0; ii < numKeyPoints - 1 && segmentNotFound; ii++) {
				float2 center = make_float2(x, y);
				float2 toPoint = screenPos - (center + keyPoints[ii]);
				float2 dir = normalize(keyPoints[ii+1] - keyPoints[ii]);
				float2 minorDir = make_float2(-dir.y, dir.x);
				float disMinor = toPoint.x*minorDir.x + toPoint.y*minorDir.y;
				if (abs(disMinor) < width / ratio)	{
					float2 keyPointAbsolute1 = center + keyPoints[ii];
					float2 keyPointAbsolute2 = center + keyPoints[ii + 1];

					//first check if screenPos and ctrlPointAbsolute2 are at the same side of Line (keyPointAbsolute1, normals[ii])
					//then check if screenPos and ctrlPointAbsolute1 are at the same side of Line (keyPointAbsolute2, normals[ii+1])

					if (((screenPos.x - keyPointAbsolute1.x)*normals[ii].y - (screenPos.y - keyPointAbsolute1.y)*normals[ii].x)
						*((keyPointAbsolute2.x - keyPointAbsolute1.x)*normals[ii].y - (keyPointAbsolute2.y - keyPointAbsolute1.y)*normals[ii].x)
						>= 0) {
						if (((screenPos.x - keyPointAbsolute2.x)*normals[ii + 1].y - (screenPos.y - keyPointAbsolute2.y)*normals[ii + 1].x)
							*((keyPointAbsolute1.x - keyPointAbsolute2.x)*normals[ii + 1].y - (keyPointAbsolute1.y - keyPointAbsolute2.y)*normals[ii + 1].x)
							>= 0) {
					
							segmentNotFound = false;
							keySegmentId = ii;

							float sin1 = dir.x*normals[ii].y - dir.y*normals[ii].x;//sin of the angle of dir x normals[ii]
							float sin2 = dir.x*normals[ii + 1].y - dir.y*normals[ii + 1].x;//sin of the angle of dir x normals[ii+1]

							float disMinorNewAbs = G(abs(disMinor) / rOut, ratio) * rOut;
							float2 intersectLeftOri = keyPointAbsolute1 + normals[ii] * (disMinor / sin1);
							float2 intersectRightOri = keyPointAbsolute2 + normals[ii + 1] * (disMinor / sin2);
							float posRatio = length(screenPos - intersectLeftOri) / length(intersectRightOri - intersectLeftOri);
								
							//look for the original segment (formed of ctrlPoints)
							bool oriSegmentNotFound = true;
							int oriSegmentId = -1;

							for (int jj = keyPointIds[keySegmentId]; jj < keyPointIds[keySegmentId + 1] && oriSegmentNotFound; jj++) {
								float2 curToPoint = screenPos - (center + ctrlPoints[jj]);
								float curDisMajor = curToPoint.x*dir.x + curToPoint.y*dir.y;
								float2 curOriSeg = ctrlPoints[jj + 1] - ctrlPoints[jj];
								float oriSegDisMajor = curOriSeg.x*dir.x + curOriSeg.y*dir.y;
								if (curDisMajor >= 0 && curDisMajor <= oriSegDisMajor){
									oriSegmentId = jj;
									oriSegmentNotFound = false;

									float normCrossProduct = curOriSeg.x*curToPoint.y - curOriSeg.y*curToPoint.x;
									if (normCrossProduct >= 0){
										float2 intersectLeft = keyPointAbsolute1 + normals[keySegmentId] * (disMinorNewAbs / sin1);
										float2 intersectRight = keyPointAbsolute2 + normals[keySegmentId + 1] * (disMinorNewAbs / sin2);
										ret = posRatio*intersectRight + (1 - posRatio)*intersectLeft;
									}
									else {
										float2 intersectLeft = keyPointAbsolute1 - normals[keySegmentId] * (disMinorNewAbs / sin1);
										float2 intersectRight = keyPointAbsolute2 - normals[keySegmentId + 1] * (disMinorNewAbs / sin2);
										ret = posRatio*intersectRight + (1 - posRatio)*intersectLeft;
									}
								}
							}

							//possible for particles located near the normal line
							if (oriSegmentNotFound){
								if (disMinor >= 0){
									float2 intersectLeft = keyPointAbsolute1 + normals[ii] * (disMinorNewAbs / sin1);
									float2 intersectRight = keyPointAbsolute2 + normals[ii + 1] * (disMinorNewAbs / sin2);
									ret = posRatio*intersectRight + (1 - posRatio)*intersectLeft;
								}
								else {
									float2 intersectLeft = keyPointAbsolute1 - normals[ii] * (disMinorNewAbs / sin1);
									float2 intersectRight = keyPointAbsolute2 - normals[ii + 1] * (disMinorNewAbs / sin2);
									ret = posRatio*intersectRight + (1 - posRatio)*intersectLeft;
								}
							}
							
						}
					}
				}




			}
		}
		/*
		float xx = 30 - length(screenPos - make_float2(x, y));
		if (xx < 0)
			xx = 0;
		ret = screenPos + normalize(screenPos - make_float2(x, y))*xx;*/
		return ret;
	}

	functor_Displace_Curve(int _x, int _y, int _width, CurveLensCtrlPoints _curveLensCtrlPoints, float _d) :
		x(_x), y(_y), width(_width), curveLensCtrlPoints(_curveLensCtrlPoints), d(_d){}
};

//thrust::transform(d_vec_posScreen.begin(), d_vec_posScreen.end(),
//	d_vec_posScreenTarget.begin(), d_vec_posScreen.begin(),
//	functor_ApproachTarget());

struct functor_ApproachTarget
{
	template<typename Tuple>
	__device__ float2 operator() (Tuple t) {
		float2 screenPos = thrust::get<0>(t); 
		float2 screenTarget = thrust::get<1>(t);
		float2 dir = screenTarget - screenPos;
		float sizeDiff = thrust::get<3>(t) - thrust::get<2>(t);
		if (length(dir) < 0.5) {
			thrust::get<0>(t) = screenTarget;
			thrust::get<2>(t) = thrust::get<3>(t);
		}
		else{
			thrust::get<0>(t) = screenPos + dir * 0.1;
			thrust::get<2>(t) = thrust::get<2>(t) + sizeDiff * 0.1;
		}

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

void Displace::DisplacePoints(std::vector<float2>& pts, std::vector<Lens*> lenses)
{
	for (int i = 0; i < lenses.size(); i++) {
		CircleLens* l = (CircleLens*)lenses[i];
		for (auto& p : pts) {
			float tmp = 1;
			p = DisplaceCircleLens(l->x, l->y, l->radius, p, tmp, focusRatio);
		}
	}
}

void Displace::Compute(float* modelview, float* projection, int winW, int winH,
	std::vector<Lens*> lenses, float4* ret, float* glyphSizeScale)
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

		//reset to 1
		d_vec_glyphSizeTarget.assign(size, 1);


		for (int i = 0; i < lenses.size(); i++) {
			switch (lenses[i]->GetType()) {
				case LENS_TYPE::TYPE_CIRCLE:
				{
					CircleLens* l = (CircleLens*)lenses[i];
					thrust::for_each(
						thrust::make_zip_iterator(
						thrust::make_tuple(
						d_vec_posScreen.begin(),
						d_vec_posClip.begin(),
						d_vec_posScreenTarget.begin(),
						d_vec_glyphSizeTarget.begin()
						)),
						thrust::make_zip_iterator(
						thrust::make_tuple(
						d_vec_posScreen.end(),
						d_vec_posClip.end(),
						d_vec_posScreenTarget.end(),
						d_vec_glyphSizeTarget.end()
						)),
						functor_Displace(l->x, l->y, l->radius, l->GetClipDepth(modelview, projection), focusRatio, sideSize));
					break;

				}
				case LENS_TYPE::TYPE_LINE:
				{
					LineLens* l = (LineLens*)lenses[i];
					thrust::transform(d_vec_posScreen.begin(), d_vec_posScreen.end(),
						d_vec_posClip.begin(), d_vec_posScreenTarget.begin(),
						functor_Displace_Line(l->x, l->y, l->lSemiMajorAxis, l->lSemiMinorAxis, l->direction, l->GetClipDepth(modelview, projection)));
					break;
				}
				case LENS_TYPE::TYPE_POLYLINE:
				{
					PolyLineLens* l = (PolyLineLens*)lenses[i];
					if(l->numCtrlPoints>=2){
						thrust::transform(d_vec_posScreen.begin(), d_vec_posScreen.end(),
							d_vec_posClip.begin(), d_vec_posScreenTarget.begin(),
							functor_Displace_PolyLine(l->x, l->y, l->width, l->polyLineLensCtrlPoints, l->direction, l->lSemiMajor, l->lSemiMinor, l->GetClipDepth(modelview, projection)));
					}
					else{
						thrust::transform(d_vec_posScreen.begin(), d_vec_posScreen.end(),
							d_vec_posClip.begin(), d_vec_posScreenTarget.begin(),
							functor_Displace_NotFinish());
					}
					break;
				}
				case LENS_TYPE::TYPE_CURVE:
				{
					CurveLens* l = (CurveLens*)lenses[i];
					if (l->isConstructing){
						thrust::transform(d_vec_posScreen.begin(), d_vec_posScreen.end(),
							d_vec_posClip.begin(), d_vec_posScreenTarget.begin(),
							functor_Displace_NotFinish());
					}
					else{
						thrust::transform(d_vec_posScreen.begin(), d_vec_posScreen.end(),
							d_vec_posClip.begin(), d_vec_posScreenTarget.begin(),
							functor_Displace_Curve(l->x, l->y, l->width, l->curveLensCtrlPoints,l->GetClipDepth(modelview, projection)));
					}
					break;
				}
			}
			//thrust::transform(posOrig.begin(), posOrig.end(), 
			//	d_vec_Dist2LensBtm.begin(), func_CompDist2LensBtm(l->c, mv));
		}
		recomputeTarget = false;
	}


	 

	thrust::device_vector<float4> d_vec_posCur(size);
	thrust::copy(ret, ret + size, d_vec_posCur.begin());
	thrust::device_vector<float> d_vec_glyphSizeScale(size);
	thrust::copy(glyphSizeScale, glyphSizeScale + size, d_vec_glyphSizeScale.begin());

	thrust::transform(d_vec_posCur.begin(), d_vec_posCur.end(), d_vec_posClip.begin(), functor_Object2Clip(mv, pj));
	thrust::transform(d_vec_posClip.begin(), d_vec_posClip.end(),
		d_vec_posScreen.begin(), functor_Clip2Screen(winW, winH));

	thrust::for_each(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		d_vec_posScreen.begin(), 
		d_vec_posScreenTarget.begin(), 
		d_vec_glyphSizeScale.begin(),
		d_vec_glyphSizeTarget.begin()
		)),
		thrust::make_zip_iterator(
		thrust::make_tuple(
		d_vec_posScreen.end(),
		d_vec_posScreenTarget.end(),
		d_vec_glyphSizeScale.end(),
		d_vec_glyphSizeTarget.end()
		)),
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
	thrust::copy(d_vec_glyphSizeScale.begin(), d_vec_glyphSizeScale.end(), glyphSizeScale);
}
