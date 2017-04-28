#include "PhysicalParticleDeformProcessor.h"
#include "Lens.h"
#include "MeshDeformProcessor.h"
#include "Particle.h"


struct functor_UpdatePointCoordsAndBrightByLineLensMesh
{
	thrust::device_ptr<int> dev_ptr_tet;;
	thrust::device_ptr<float> dev_ptr_X;
	int tet_number;
	float3 lensCenter;
	float lSemiMajorAxis;
	float lSemiMinorAxis;
	float3 majorAxis;
	float focusRatio;
	float3 lensDir;
	bool isFreezingFeature;
	int snappedFeatureId;
	template<typename Tuple>
	__device__ __host__ void operator() (Tuple t)
	{
		if (isFreezingFeature && snappedFeatureId == thrust::get<5>(t)){
			thrust::get<3>(t) = thrust::get<0>(t);
			thrust::get<4>(t) = 1.0;
			return;
		}

		int vi = thrust::get<1>(t);
		if (vi >= 0 && vi < tet_number){
			float4 vb = thrust::get<2>(t);
			float3 vv[4];
			for (int k = 0; k < 4; k++){
				int iv = dev_ptr_tet[vi * 4 + k];
				vv[k] = make_float3(dev_ptr_X[3 * iv + 0], dev_ptr_X[3 * iv + 1], dev_ptr_X[3 * iv + 2]);
			}
			thrust::get<3>(t) = make_float4(vb.x * vv[0] + vb.y * vv[1] + vb.z * vv[2] + vb.w * vv[3], 1);
		}
		else{
			thrust::get<3>(t) = thrust::get<0>(t);
		}

		thrust::get<4>(t) = 1.0;
		const float dark = 0.5;
		float3 lenCen2P = make_float3(thrust::get<0>(t)) - lensCenter;

		//float alpha = 0.25f*2; //for Cosmology
		float alpha = 0.25f; //for FPM
		float lensCen2PMajorProj = dot(lenCen2P, majorAxis);
		float3 minorAxis = cross(lensDir, majorAxis);
		float lensCen2PMinorProj = dot(lenCen2P, minorAxis);
		if (abs(lensCen2PMajorProj) < lSemiMajorAxis){
			if (abs(lensCen2PMinorProj) < lSemiMinorAxis / focusRatio){
				float lensCen2PProj = dot(lenCen2P, lensDir);
				if (lensCen2PProj < 0){
					thrust::get<4>(t) = max(1.0f / (alpha * abs(lensCen2PProj) + 1.0f), dark);
				}
				else{
					thrust::get<4>(t) = 1.0f * (1 + alpha / 5 * abs(lensCen2PProj));
				}
			}
			else{
				//thrust::get<4>(t) = 1.0f;
				thrust::get<4>(t) = max(1.0f / (alpha * abs(abs(lensCen2PMinorProj) - lSemiMinorAxis / focusRatio) + 1.0f), dark);
			}
		}
		else{
			if (abs(lensCen2PMinorProj) < lSemiMinorAxis / focusRatio){
				thrust::get<4>(t) = max(1.0f / (alpha * abs(abs(lensCen2PMajorProj) - lSemiMajorAxis) + 1.0f), dark);
			}
			else{
				//thrust::get<4>(t) = 1.0f;
				float dMaj = abs(lensCen2PMajorProj) - lSemiMajorAxis;
				float dMin = abs(lensCen2PMinorProj) - lSemiMinorAxis / focusRatio;
				thrust::get<4>(t) = max(1.0f / (alpha * sqrt(dMin*dMin + dMaj*dMaj) + 1.0f), dark);
			}
			//thrust::get<4>(t) = 1.0f;
		}
	}
	functor_UpdatePointCoordsAndBrightByLineLensMesh(thrust::device_ptr<int> _dev_ptr_tet, thrust::device_ptr<float> _dev_ptr_X, int _tet_number, float3 _lensCenter, float _lSemiMajorAxisGlobal, float _lSemiMinorAxisGlobal, float3 _majorAxisGlobal, float _focusRatio, float3 _lensDir, bool _isFreezingFeature, int _snappedFeatureId) : dev_ptr_tet(_dev_ptr_tet), dev_ptr_X(_dev_ptr_X), tet_number(_tet_number), lensCenter(_lensCenter), lSemiMajorAxis(_lSemiMajorAxisGlobal), lSemiMinorAxis(_lSemiMinorAxisGlobal), majorAxis(_majorAxisGlobal), focusRatio(_focusRatio), lensDir(_lensDir), isFreezingFeature(_isFreezingFeature), snappedFeatureId(_snappedFeatureId){}
};



void PhysicalParticleDeformProcessor::UpdatePointCoordsAndBright_LineMeshLens_Thrust(std::shared_ptr<Particle> p, float* brightness, LineLens3D * l, bool isFreezingFeature, int snappedFeatureId)
{
	if (isFreezingFeature)
	{
		if (!p->hasFeature)
		{
			std::cout << "error of feature in particle data" << std::endl;
			exit(0);
		}
	}

	thrust::device_ptr<int> dev_ptr_tet(meshDeformer->GetTetDev());
	thrust::device_ptr<float> dev_ptr_X(meshDeformer->GetXDev());
	int tet_number = meshDeformer->GetTetNumber();

	thrust::for_each(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		meshDeformer->d_vec_vOri.begin(),
		meshDeformer->d_vec_vIdx.begin(),
		meshDeformer->d_vec_vBaryCoord.begin(),
		meshDeformer->d_vec_v.begin(),
		meshDeformer->d_vec_brightness.begin(),
		meshDeformer->d_vec_feature.begin()
		)),
		thrust::make_zip_iterator(
		thrust::make_tuple(
		meshDeformer->d_vec_vOri.end(),
		meshDeformer->d_vec_vIdx.end(),
		meshDeformer->d_vec_vBaryCoord.end(),
		meshDeformer->d_vec_v.end(),
		meshDeformer->d_vec_brightness.end(),
		meshDeformer->d_vec_feature.end()
		)),
		functor_UpdatePointCoordsAndBrightByLineLensMesh(dev_ptr_tet, dev_ptr_X, tet_number, l->c, l->lSemiMajorAxisGlobal, l->lSemiMinorAxisGlobal, l->majorAxisGlobal, l->focusRatio, l->lensDir, isFreezingFeature, snappedFeatureId));

	thrust::copy(meshDeformer->d_vec_v.begin(), meshDeformer->d_vec_v.end(), &(p->pos[0]));
	thrust::copy(meshDeformer->d_vec_brightness.begin(), meshDeformer->d_vec_brightness.end(), brightness);

}

void PhysicalParticleDeformProcessor::UpdatePointCoordsAndBright_UniformMesh(std::shared_ptr<Particle> particle, float* brightness, float* _mv)
{

	int n = particle->numParticles;
	float4* v = &(particle->pos[0]);

	int* tet = meshDeformer->GetTet();
	float* X = meshDeformer->GetX();
	for (int i = 0; i < n; i++){
		int vi = meshDeformer->vIdx[i];
		if (vi == -1){
			v[i] = make_float4(-100, -100, -100, 1);
		}
		else{
			float4 vb = meshDeformer->vBaryCoord[i];
			float3 vv[4];
			for (int k = 0; k < 4; k++){
				int iv = tet[vi * 4 + k];
				vv[k] = make_float3(X[3 * iv + 0], X[3 * iv + 1], X[3 * iv + 2]);
			}
			v[i] = make_float4(vb.x * vv[0] + vb.y * vv[1] + vb.z * vv[2] + vb.w * vv[3], 1);
		}
	}


	Lens* l = lenses->back();
	float3 lensCen = l->c;
	float focusRatio = l->focusRatio;
	float radius = ((CircleLens3D*)l)->objectRadius;
	float _invmv[16];
	invertMatrix(_mv, _invmv);
	float3 cameraObj = make_float3(Camera2Object(make_float4(0, 0, 0, 1), _invmv));
	float3 lensDir = normalize(cameraObj - lensCen);

	const float dark = 0.1;
	const float transRad = radius / focusRatio;
	for (int i = 0; i < particle->numParticles; i++) {
		float3 vert = make_float3(v[i]);
		//float3 lensCenFront = lensCen + lensDir * radius;
		float3 lensCenBack = lensCen - lensDir * radius;
		float3 lensCenFront2Vert = vert - lensCenBack;
		float lensCenFront2VertProj = dot(lensCenFront2Vert, lensDir);
		float3 moveVec = lensCenFront2Vert - lensCenFront2VertProj * lensDir;
		brightness[i] = 1.0;
		if (lensCenFront2VertProj < 0){
			float dist2Ray = length(moveVec);
			if (dist2Ray < radius / focusRatio){
				brightness[i] = max(dark, 1.0f / (0.5f * (-lensCenFront2VertProj) + 1.0f));;
			}
		}
	}
}


bool PhysicalParticleDeformProcessor::process(float* modelview, float* projection, int winWidth, int winHeight)
{
	if (!isActive)
		return false;

	if (meshDeformer->meshJustDeformed){

		if (lenses->size() < 0){
			meshDeformer->meshJustDeformed = false;
			return false;
			//TODO: do processing only by mesh, without computing the brightness but not 
		}
		else{
			Lens *l = lenses->back();

			if (l->type == TYPE_LINE){
				UpdatePointCoordsAndBright_LineMeshLens_Thrust(particle, &(particle->glyphBright[0]), (LineLens3D*)l, particle->isFreezingFeature, particle->snappedFeatureId);
			}
			else if (l->type == TYPE_CIRCLE){
				UpdatePointCoordsAndBright_UniformMesh(particle, &(particle->glyphBright[0]), modelview);
			}
			meshDeformer->meshJustDeformed = false;
			return true;
		}
	}
	else{
		return false;
	}
}