#include <iostream>
#include "ViewpointEvaluator.h"
#include "TransformFunc.h"
#include "Particle.h"
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

texture<float, 3, cudaReadModeElementType>  volumeVal;
texture<unsigned short, 3, cudaReadModeElementType>  volumeLabel;

texture<float4, 3, cudaReadModeElementType>  gradientTexOri;
texture<float4, 3, cudaReadModeElementType>  gradientTexFiltered;

ViewpointEvaluator::ViewpointEvaluator(std::shared_ptr<RayCastingParameters> _r, std::shared_ptr<Volume> v)
{
	rcp = _r;

	volume = v;

	volumeVal.normalized = false;
	volumeVal.filterMode = cudaFilterModeLinear;
	volumeVal.addressMode[0] = cudaAddressModeBorder;
	volumeVal.addressMode[1] = cudaAddressModeBorder;
	volumeVal.addressMode[2] = cudaAddressModeBorder;

	GPU_setConstants(&(rcp->transFuncP1), &(rcp->transFuncP2), &(rcp->la), &(rcp->ld), &(rcp->ls), &(volume->spacing));
	GPU_setVolume(&(volume->volumeCuda));

	rcp->tstep = 1.0; //generally don't need to sample beyond each voxel

	cudaMalloc(&d_hist, sizeof(float)*nbins);

	cubeFaceHists.resize(6);
	for (int i = 0; i < 6; i++){
		cudaMalloc(&cubeFaceHists[i], sizeof(float)*nbins);
	}

	cubeInfo.resize(6);
}


void ViewpointEvaluator::initDownSampledResultVolume(int3 sampleSize)
{
	if (resVol != 0)
		resVol.reset();
	resVol = std::make_shared<Volume>();
	resVol->setSize(sampleSize);

	//note that these two rely on the method to set the viewpoint of the sample. also currently most functions do not consider about the origin
	resVol->dataOrigin = indToLocal(0, 0, 0);
	resVol->spacing = indToLocal(1, 1, 1) - resVol->dataOrigin;
}

float3 ViewpointEvaluator::indToLocal(int i, int j, int k)
{
	return make_float3(i - 1, j - 1, k - 1)*make_float3(volume->size.x, volume->size.y, volume->size.z) / make_float3(resVol->size - 3)*volume->spacing;
}

void ViewpointEvaluator::setLabel(std::shared_ptr<VolumeCUDA> v)
{
	volumeLabel.normalized = false;
	volumeLabel.filterMode = cudaFilterModePoint;
	volumeLabel.addressMode[0] = cudaAddressModeBorder;
	volumeLabel.addressMode[1] = cudaAddressModeBorder;
	volumeLabel.addressMode[2] = cudaAddressModeBorder;

	checkCudaErrors(cudaBindTextureToArray(volumeLabel, v->content, v->channelDesc));

	labelBeenSet = true;
}


void ViewpointEvaluator::initJS06Sphere()
{
	if (JS06SphereInited)	return;

	if (d_r != 0) cudaFree(d_r);
	setSpherePoints();
	cudaMalloc(&d_r, sizeof(float)*numSphereSample);
	JS06SphereInited = true;
	Tao09DetailInited = false;
}

void ViewpointEvaluator::initTao09Detail()
{
	if (Tao09DetailInited)	return;

	std::cout << "initializing for viewpoint evaluation method" << std::endl;

	if (d_r != 0) cudaFree(d_r);
	setSpherePoints();
	cudaMalloc(&d_r, sizeof(float)*numSphereSample);
	
	float* gradient = 0;
	volume->computeGradient(gradient);
	volumeGradient.VolumeCUDA_deinit();
	volumeGradient.VolumeCUDA_init(volume->size, gradient, 0, 4);
	delete[] gradient;

	//float* bilateralVolumeRes = 0;
	//volume->computeBilateralFiltering(bilateralVolumeRes, 2, 0.2);
	//FILE * fp = fopen("bilat.raw", "wb");
	//fwrite(bilateralVolumeRes, sizeof(float), volume->size.x*volume->size.y*volume->size.z, fp);
	//fclose(fp);
	
	float* bilateralVolumeRes = new float[volume->size.x*volume->size.y*volume->size.z];
	FILE * fp = fopen("bilat.raw", "rb");
	fread(bilateralVolumeRes, sizeof(float), volume->size.x*volume->size.y*volume->size.z, fp);
	fclose(fp);

	float* bGradient = 0;
	volume->computeGradient(bilateralVolumeRes, volume->size, bGradient);
	filteredVolumeGradient.VolumeCUDA_deinit();
	filteredVolumeGradient.VolumeCUDA_init(volume->size, bGradient, 0, 4);
	delete[] bGradient;
	delete[] bilateralVolumeRes;
		
	gradientTexOri.normalized = false;
	gradientTexOri.filterMode = cudaFilterModeLinear;
	gradientTexOri.addressMode[0] = cudaAddressModeBorder;
	gradientTexOri.addressMode[1] = cudaAddressModeBorder;
	gradientTexOri.addressMode[2] = cudaAddressModeBorder;
	
	gradientTexFiltered.normalized = false;
	gradientTexFiltered.filterMode = cudaFilterModeLinear;
	gradientTexFiltered.addressMode[0] = cudaAddressModeBorder;
	gradientTexFiltered.addressMode[1] = cudaAddressModeBorder;
	gradientTexFiltered.addressMode[2] = cudaAddressModeBorder;
	
	Tao09DetailInited = true;
	JS06SphereInited = false;
}

void ViewpointEvaluator::compute_UniformSampling(VPMethod m)
{
	float maxRes = -999;
	int3 sampleSize = resVol->size;
	if (m == BS05){
	}
	else if (m == JS06Sphere){
		initJS06Sphere();
		for (int k = 0; k < sampleSize.z; k++){
			std::cout << "now doing k = " << k << std::endl;
			for (int j = 0; j < sampleSize.y; j++){
				for (int i = 0; i < sampleSize.x; i++){
					float3 eyeInLocal = indToLocal(i, j, k);
					float entroRes = computeEntropyJS06Sphere(eyeInLocal);
					resVol->values[k*sampleSize.y*sampleSize.x + j*sampleSize.x + i] = entroRes;
					if (entroRes>maxRes){
						maxRes = entroRes;
						optimalEyeInLocal = eyeInLocal;
					}
				}
			}
		}
	}
	optimalEyeValid = true;
}


void ViewpointEvaluator::compute_SkelSampling(VPMethod m)
{
	float maxRes = -999;
	//int3 sampleSize = resVol->size;
	if (m == BS05){
	}
	else if (m == JS06Sphere){
		initJS06Sphere();
		for (int i = 0; i < skelViews.size(); i++){
			for (int j = 0; j < skelViews[i]->numParticles; j++){
				float3 eyeInLocal = make_float3(skelViews[i]->pos[j]);
				float entroRes = computeEntropyJS06Sphere(eyeInLocal);
				if (entroRes>maxRes){
					maxRes = entroRes;
					optimalEyeInLocal = eyeInLocal;
				}
			}
		}
	}
	else if (m == Tao09Detail){
		initTao09Detail();

		checkCudaErrors(cudaBindTextureToArray(gradientTexOri, volumeGradient.content, volumeGradient.channelDesc));
		checkCudaErrors(cudaBindTextureToArray(gradientTexFiltered, filteredVolumeGradient.content, filteredVolumeGradient.channelDesc));

		for (int i = 0; i < skelViews.size(); i++){
			for (int j = 0; j < skelViews[i]->numParticles; j++){
				float3 eyeInLocal = make_float3(skelViews[i]->pos[j]);
				float entroRes = computeEntropyTao09Detail(eyeInLocal);
				if (entroRes>maxRes){
					maxRes = entroRes;
					optimalEyeInLocal = eyeInLocal;
				}
			}
		}
		
		checkCudaErrors(cudaUnbindTexture(gradientTexOri));
		checkCudaErrors(cudaUnbindTexture(gradientTexFiltered));
	}
	optimalEyeValid = true;
}

void ViewpointEvaluator::saveResultVol(const char* fname)
{
	resVol->saveRawToFile(fname);
}

void ViewpointEvaluator::setSpherePoints(int n)
{
	if (spherePointSet) return;

	//source: https://www.openprocessing.org/sketch/41142

	numSphereSample = n;
	sphereSamples.resize(n);

	float phi = (sqrt(5) + 1) / 2 - 1; // golden ratio
	float ga = phi * 2 * M_PI;           // golden angle

	for (int i = 1; i <= numSphereSample; ++i) {
		float lon = ga*i;
		lon /= 2 * M_PI; lon -= floor(lon); lon *= 2 * M_PI;
		if (lon > M_PI)  lon -= 2 * M_PI;

		// Convert dome height (which is proportional to surface area) to latitude
		float lat = asin(-1 + 2 * i / (float)numSphereSample);

		sphereSamples[i - 1] = SpherePoint(lat, lon);
	}
	if (d_sphereSamples != 0){
		cudaFree(d_sphereSamples);
	}
	cudaMalloc(&d_sphereSamples, sizeof(float)*numSphereSample * 3);
	cudaMemcpy(d_sphereSamples, (float*)(&sphereSamples[0]), sizeof(float)*numSphereSample * 3, cudaMemcpyHostToDevice);
	spherePointSet = true;
}

__constant__ float colorTableDiverge[33][4] = {
	0, 0.231372549, 0.298039216, 0.752941176,
	0.03125, 0.266666667, 0.352941176, 0.8,
	0.0625, 0.301960784, 0.407843137, 0.843137255,
	0.09375, 0.341176471, 0.458823529, 0.882352941,
	0.125, 0.384313725, 0.509803922, 0.917647059,
	0.15625, 0.423529412, 0.556862745, 0.945098039,
	0.1875, 0.466666667, 0.603921569, 0.968627451,
	0.21875, 0.509803922, 0.647058824, 0.984313725,
	0.25, 0.552941176, 0.690196078, 0.996078431,
	0.28125, 0.596078431, 0.725490196, 1,
	0.3125, 0.639215686, 0.760784314, 1,
	0.34375, 0.682352941, 0.788235294, 0.992156863,
	0.375, 0.721568627, 0.815686275, 0.976470588,
	0.40625, 0.760784314, 0.835294118, 0.956862745,
	0.4375, 0.8, 0.850980392, 0.933333333,
	0.46875, 0.835294118, 0.858823529, 0.901960784,
	0.5, 0.866666667, 0.866666667, 0.866666667,
	0.53125, 0.898039216, 0.847058824, 0.819607843,
	0.5625, 0.925490196, 0.82745098, 0.77254902,
	0.59375, 0.945098039, 0.8, 0.725490196,
	0.625, 0.960784314, 0.768627451, 0.678431373,
	0.65625, 0.968627451, 0.733333333, 0.62745098,
	0.6875, 0.968627451, 0.694117647, 0.580392157,
	0.71875, 0.968627451, 0.650980392, 0.529411765,
	0.75, 0.956862745, 0.603921569, 0.482352941,
	0.78125, 0.945098039, 0.552941176, 0.435294118,
	0.8125, 0.925490196, 0.498039216, 0.388235294,
	0.84375, 0.898039216, 0.439215686, 0.345098039,
	0.875, 0.870588235, 0.376470588, 0.301960784,
	0.90625, 0.835294118, 0.31372549, 0.258823529,
	0.9375, 0.796078431, 0.243137255, 0.219607843,
	0.96875, 0.752941176, 0.156862745, 0.184313725,
	1, 0.705882353, 0.015686275, 0.149019608,
};

__device__ float3 GetColourDiverge2(float v)
{
	//can be accelerated using binary search!!
	int pos = 0;
	bool notFound = true;
	const int numItemColorTableDiverge = 33;
	while (pos < numItemColorTableDiverge - 1 && notFound) {
		if (colorTableDiverge[pos][0] <= v && colorTableDiverge[pos + 1][0] >= v)
			notFound = false;
		pos++;
	}
	float ratio = (v - colorTableDiverge[pos][0]) / (colorTableDiverge[pos + 1][0] - colorTableDiverge[pos][0]);
	
	float3 c = make_float3(
		ratio*(colorTableDiverge[pos + 1][1] - colorTableDiverge[pos][1]) + colorTableDiverge[pos][1],
		ratio*(colorTableDiverge[pos + 1][2] - colorTableDiverge[pos][2]) + colorTableDiverge[pos][2],
		ratio*(colorTableDiverge[pos + 1][3] - colorTableDiverge[pos][3]) + colorTableDiverge[pos][3]);

	return(c);
}


__device__
int intersectBox2(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
	// compute intersection of ray with all six bbox planes
	float3 invR = make_float3(1.0f) / r.d;
	float3 tbot = invR * (boxmin - r.o);
	float3 ttop = invR * (boxmax - r.o);

	// re-order intersections to find smallest and largest on each axis
	float3 tmin = fminf(ttop, tbot);
	float3 tmax = fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
	float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}



__constant__ float transFuncP1;
__constant__ float transFuncP2;
__constant__ float la;
__constant__ float ld;
__constant__ float ls;
__constant__ float3 spacing;

void ViewpointEvaluator::GPU_setVolume(const VolumeCUDA *vol)
{
	checkCudaErrors(cudaBindTextureToArray(volumeVal, vol->content, vol->channelDesc));
}


void ViewpointEvaluator::GPU_setConstants(float* _transFuncP1, float* _transFuncP2, float* _la, float* _ld, float* _ls, float3* _spacing)
{
	checkCudaErrors(cudaMemcpyToSymbol(transFuncP1, _transFuncP1, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(transFuncP2, _transFuncP2, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(la, _la, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(ld, _ld, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(ls, _ls, sizeof(float)));

	checkCudaErrors(cudaMemcpyToSymbol(spacing, _spacing, sizeof(float3)));
}


struct functor_computeEntropy
{
	float sum;
	__device__ __host__ float operator() (float r)
	{
		if (r < 0.00001){
			return 0;
		}
		else{
			float qj = r / sum;
			return -qj*log(qj);
		}
	}
	functor_computeEntropy(float s) : sum(s){}
};





__global__ void d_computeSphereColor(float density, float brightness,
	float3 eyeInLocal, int3 volumeSize, int maxSteps, float tstep, bool useColor, float * r, int numSphereSample, float *sphereSamples, float *hist, int nbins, bool useHist, bool useLabelCount, bool useDist, VPMethod vpmethod)
{

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numSphereSample)	return;

	const float opacityThreshold = 0.95f;


	Ray eyeRay;
	eyeRay.o = eyeInLocal;
	eyeRay.d = make_float3(sphereSamples[3 * i], sphereSamples[3 * i + 1], sphereSamples[3 * i + 2]);

	float tnear, tfar;
	const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
	const float3 boxMax = spacing*make_float3(volumeSize); 
	intersectBox2(eyeRay, boxMin, boxMax, &tnear, &tfar);
	tnear = 0.01f;	//!!!NOTE!!! this tnear is not in the clip space but in the original space

	// march along ray from front to back, accumulating color
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;

	float4 sum = make_float4(0.0f); //for JS
	unsigned short label = 0; //for label count
	float detailDescriptor = 0; //for TaoDetail

	float lightingThr = 0.000001; //used for the threshold of TaoDetail

	for (int i = 0; i<maxSteps; i++)
	{
		float3 coord = pos / spacing;
		float sample = tex3D(volumeVal, coord.x, coord.y, coord.z);
		float funcRes = clamp((sample - transFuncP2) / (transFuncP1 - transFuncP2), 0.0, 1.0);

		// lookup in transfer function texture
		float4 col;

		float3 cc;
		if (useColor)
			cc = GetColourDiverge2(clamp(funcRes, 0.0f, 1.0f));
		else
			cc = make_float3(funcRes, funcRes, funcRes);

		////currently ignore lighting
		col = make_float4(la*cc, funcRes);

		col.w *= density;

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		
		float visibility = 1.0f - sum.w;
		
		// "over" operator for front-to-back blending
		sum = sum + col*(1.0f - sum.w);

		if (vpmethod == Tao09Detail){ //if not Tao09Detail, the sampled texture may not be prepared
			float curDetail = 0;
			float3 normalOri = make_float3(tex3D(gradientTexOri, coord.x, coord.y, coord.z)) / spacing;
			float3 normalFiltered = make_float3(tex3D(gradientTexFiltered, coord.x, coord.y, coord.z)) / spacing;
			if (length(normalOri) > lightingThr){
				if (length(normalFiltered) > lightingThr){
					curDetail = 1 - dot(normalize(normalOri), normalize(normalFiltered));
				}
				else{
					curDetail = 1;
				}
			}
			detailDescriptor = detailDescriptor + curDetail*visibility;
		}

		// exit early if opaque
		if (sum.w > opacityThreshold){
			break;
		}
		else if (useLabelCount){
			unsigned short curlabel = tex3D(volumeLabel, coord.x, coord.y, coord.z);
			if (curlabel > label)
			{
				label = curlabel;
			}
		}

		t += tstep;

		if (t > tfar){
			break;
		}

		pos += step;
	}

	sum *= brightness;

	float uv = sum.w;
	if (vpmethod == Tao09Detail)
		uv = detailDescriptor;

	r[i] = uv;

	if (useLabelCount){
		r[i] = label;
		if (useHist){
			int bin;
			if (label > 0)
				bin = 1;
			else
				bin = 0;
			atomicAdd(hist + bin, 1);
		}
	}
	else if(useDist){
		float dis;
		if (uv < 0.00001)
			dis = 0;
		else
			dis = t;

		r[i] = uv;

		if (useHist){
			float maxDist = fmaxf(fmaxf(boxMax.x, boxMax.y), boxMax.z);
			float minDist = 0;
			// !!! change the range into [0,1] !!!
			int bin = min((int)((dis - minDist) / (maxDist - minDist) *nbins), nbins - 1);
			atomicAdd(hist + bin, 1);		
		}
	}
	else{
		if (useHist){
			// !!! this is true only when we know uv is in [0,1] !!!
			int bin = min((int)(uv*nbins), nbins - 1);
			atomicAdd(hist + bin, 1);
		}
	}
}


struct is_solid
{
	__host__ __device__
	bool operator()(float x)
	{
		return x>0.00001;
	}
};


float ViewpointEvaluator::computeEntropyJS06Sphere(float3 eyeInLocal)
{
	int threadsPerBlock = 64;
	int blocksPerGrid = (numSphereSample + threadsPerBlock - 1) / threadsPerBlock;

	cudaMemset(d_hist, 0, sizeof(float)*nbins);

	if (useLabelCount && !labelBeenSet){
		std::cout << "label not set yet! " << std::endl;
		exit(0);
	}

	d_computeSphereColor << <blocksPerGrid, threadsPerBlock >> >(rcp->density, rcp->brightness, eyeInLocal, volume->size, rcp->maxSteps, rcp->tstep, rcp->useColor, d_r, numSphereSample, d_sphereSamples, d_hist, nbins, useHist, useLabelCount, useDist, JS06Sphere);

	float ret;
	if (useHist){
		if (useLabelCount){
			ret = computeVectorEntropy(d_hist, maxLabel + 1);
		}
		else if (useColor){
			ret = computeVectorEntropy(d_hist, nbins);
		}
	}
	else if (useTrad){
		ret = computeVectorEntropy(d_r, numSphereSample);
	}
	else{
		std::cout << "entropy computation not defined! " << std::endl;
		exit(0);
	}

	return ret;
}

float ViewpointEvaluator::computeEntropyTao09Detail(float3 eyeInLocal)
{
	int threadsPerBlock = 64;
	int blocksPerGrid = (numSphereSample + threadsPerBlock - 1) / threadsPerBlock;

	cudaMemset(d_hist, 0, sizeof(float)*nbins);

	if (useLabelCount && !labelBeenSet){
		std::cout << "label not set yet! " << std::endl;
		exit(0);
	}

	d_computeSphereColor << <blocksPerGrid, threadsPerBlock >> >(rcp->density, rcp->brightness, eyeInLocal, volume->size, rcp->maxSteps, rcp->tstep, rcp->useColor, d_r, numSphereSample, d_sphereSamples, d_hist, nbins, useHist, useLabelCount, useDist, Tao09Detail);

	float ret;
	if (useHist){
		if (useLabelCount){
			ret = computeVectorEntropy(d_hist, maxLabel + 1);
		}
		else if (useColor){
			ret = computeVectorEntropy(d_hist, nbins);
		}
	}
	else if (useTrad){
		ret = computeVectorEntropy(d_r, numSphereSample);
	}
	else{
		std::cout << "entropy computation not defined! " << std::endl;
		exit(0);
	}

	return ret;
}


float ViewpointEvaluator::computeVectorEntropy(float* ary, int size)
{
	thrust::device_vector< float > iVec(ary, ary + size);

	std::vector<float> stl_vector(size);
	thrust::copy(iVec.begin(), iVec.end(), stl_vector.begin());

	float sum = thrust::reduce(iVec.begin(), iVec.end(), (float)0, thrust::plus<float>());
	thrust::transform(iVec.begin(), iVec.end(), iVec.begin(), functor_computeEntropy(sum));
	return thrust::reduce(iVec.begin(), iVec.end(), (float)0, thrust::plus<float>());
}


__global__ void d_computeCubeColorHist(float density, float brightness,
	float3 eyeInLocal, float3 viewVec, float3 upVec, int3 volumeSize, int maxSteps, float tstep, bool useColor, float * r, int numSphereSample, float *sphereSamples, float *hist0, float *hist1, float *hist2, float *hist3, float *hist4, float *hist5, int nbins, bool useHist, bool useLabelCount, VPMethod vpmethod)//, bool useDist)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numSphereSample)	return;

	const float opacityThreshold = 0.95f;

	Ray eyeRay;
	eyeRay.o = eyeInLocal;
	eyeRay.d = make_float3(sphereSamples[3 * i], sphereSamples[3 * i + 1], sphereSamples[3 * i + 2]);

	float tnear, tfar;
	const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
	const float3 boxMax = spacing*make_float3(volumeSize);
	intersectBox2(eyeRay, boxMin, boxMax, &tnear, &tfar);
	tnear = 0.01f;	//!!!NOTE!!! this tnear is not in the clip space but in the original space

	// march along ray from front to back, accumulating color
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;

	float4 sum = make_float4(0.0f); //for JS
	unsigned short label = 0; //for label count
	float detailDescriptor= 0; //for TaoDetail

	float lightingThr = 0.000001; //used for the threshold of TaoDetail


	for (int i = 0; i<maxSteps; i++)
	{
		float3 coord = pos / spacing;
		float sample = tex3D(volumeVal, coord.x, coord.y, coord.z);
		float funcRes = clamp((sample - transFuncP2) / (transFuncP1 - transFuncP2), 0.0, 1.0);

		// lookup in transfer function texture
		float4 col;

		float3 cc;
		if (useColor)
			cc = GetColourDiverge2(clamp(funcRes, 0.0f, 1.0f));
		else
			cc = make_float3(funcRes, funcRes, funcRes);

		////currently ignore lighting
		col = make_float4(la*cc, funcRes);

		col.w *= density;

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		// "over" operator for front-to-back blending

		float visibility = 1.0f - sum.w;

		sum = sum + col*(1.0f - sum.w);


		if (vpmethod == Tao09Detail){ //if not Tao09Detail, the sampled texture may not be prepared
			float curDetail = 0;
			float3 normalOri = make_float3(tex3D(gradientTexOri, coord.x, coord.y, coord.z)) / spacing;
			float3 normalFiltered = make_float3(tex3D(gradientTexFiltered, coord.x, coord.y, coord.z)) / spacing;
			if (length(normalOri) > lightingThr){
				if (length(normalFiltered) > lightingThr){
					curDetail = 1 - dot(normalize(normalOri), normalize(normalFiltered));
				}
				else{
					curDetail = 1;
				}
			}
			detailDescriptor = detailDescriptor + curDetail*visibility;
		}

		// exit early if opaque
		if (sum.w > opacityThreshold){
			break;
		}
		else if (useLabelCount){
			unsigned short curlabel = tex3D(volumeLabel, coord.x, coord.y, coord.z);
			if (curlabel > label)
			{
				label = curlabel;
			}
		}

		t += tstep;
		if (t > tfar){
			break;
		}
		pos += step;
	}
	sum *= brightness;


	int bin;
	if (useLabelCount){
		// !!! this is true only when we know max label is 1 !!!
		if (label > 0)		
			bin = 1;
		else
			bin = 0;
	}
	else{
		if (vpmethod == JS06Sphere){
			float uv = sum.x; // !!! NOTE !!! this is only true for grey scale color
			// !!! this is true only when we know uv is in [0,1] !!!
			bin = min((int)(uv*nbins), nbins - 1);
		}
		else if (vpmethod == Tao09Detail){
			float uv = detailDescriptor;
			// !!! this is true only when we know uv is in [0,1] !!!
			bin = min((int)(uv*nbins), nbins - 1);
		}
	}

	//suppose x coord is along viewVew, suppose upVec and viewVew are normalized and perpendicular
	float3 sidevec = cross(upVec, viewVec);
	float rayz = dot(eyeRay.d, upVec), rayx = dot(eyeRay.d, viewVec), rayy = dot(eyeRay.d, sidevec);

	float xabs = abs(rayx), yabs = abs(rayy), zabs = abs(rayz);
	if (xabs > yabs && xabs > zabs){
		if (rayx > 0){ //front
			atomicAdd(hist0 + bin, 1);
		}
		else{ //back
			atomicAdd(hist1 + bin, 1);
		}
	}
	else if (yabs > xabs && yabs > zabs){
		if (rayy > 0){ //left
			atomicAdd(hist2 + bin, 1);
		}
		else{ //right
			atomicAdd(hist3 + bin, 1);
		}
	}
	else{ //zabs is the max
		if (rayz > 0){ //up
			atomicAdd(hist4 + bin, 1);
		}
		else{ //below
			atomicAdd(hist5 + bin, 1);
		}
	}
}


void ViewpointEvaluator::computeCubeEntropy(float3 eyeInLocal, float3 viewDir, float3 upDir, VPMethod m)
{
	if (m == Tao09Detail){
		initTao09Detail();

		int threadsPerBlock = 64;
		int blocksPerGrid = (numSphereSample + threadsPerBlock - 1) / threadsPerBlock;

		for (int i = 0; i < 6; i++){
			cudaMemset(cubeFaceHists[i], 0, sizeof(float)*nbins);
		}

		if (useLabelCount && !labelBeenSet){
			std::cout << "label not set yet! " << std::endl;
			exit(0);
		}

		checkCudaErrors(cudaBindTextureToArray(gradientTexOri, volumeGradient.content, volumeGradient.channelDesc));
		checkCudaErrors(cudaBindTextureToArray(gradientTexFiltered, filteredVolumeGradient.content, filteredVolumeGradient.channelDesc));

		d_computeCubeColorHist << <blocksPerGrid, threadsPerBlock >> >(rcp->density, rcp->brightness, eyeInLocal, viewDir, upDir, volume->size, rcp->maxSteps, rcp->tstep, rcp->useColor, d_r, numSphereSample, d_sphereSamples, cubeFaceHists[0], cubeFaceHists[1], cubeFaceHists[2], cubeFaceHists[3], cubeFaceHists[4], cubeFaceHists[5], nbins, useHist, useLabelCount, VPMethod::Tao09Detail);

		checkCudaErrors(cudaUnbindTexture(gradientTexOri));
		checkCudaErrors(cudaUnbindTexture(gradientTexFiltered));

		for (int i = 0; i < 6; i++){
			if (useHist){
				if (useLabelCount){
					cubeInfo[i] = computeVectorEntropy(cubeFaceHists[i], maxLabel + 1);
				}
				else if (useColor){
					cubeInfo[i] = computeVectorEntropy(cubeFaceHists[i], nbins);
				}
			}
			else{
				std::cout << "entropy computation not defined! " << std::endl;
				exit(0);
			}
		}
	}
	else{
		return;
	}

}

void ViewpointEvaluator::computeCubeEntropy(float3 eyeInLocal, float3 viewDir, float3 upDir)
{
	int threadsPerBlock = 64;
	int blocksPerGrid = (numSphereSample + threadsPerBlock - 1) / threadsPerBlock;

//	cudaMemset(d_hist, 0, sizeof(float)*nbins);
	for (int i = 0; i < 6; i++){
		cudaMemset(cubeFaceHists[i], 0, sizeof(float)*nbins);
	}

	if (useLabelCount && !labelBeenSet){
		std::cout << "label not set yet! " << std::endl;
		exit(0);
	}

	d_computeCubeColorHist << <blocksPerGrid, threadsPerBlock >> >(rcp->density, rcp->brightness, eyeInLocal, viewDir, upDir, volume->size, rcp->maxSteps, rcp->tstep, rcp->useColor, d_r, numSphereSample, d_sphereSamples, cubeFaceHists[0], cubeFaceHists[1], cubeFaceHists[2], cubeFaceHists[3], cubeFaceHists[4], cubeFaceHists[5], nbins, useHist, useLabelCount, VPMethod::JS06Sphere);

	for (int i = 0; i < 6; i++){
		
		if (useHist){
			if (useLabelCount){
				cubeInfo[i] = computeVectorEntropy(cubeFaceHists[i], maxLabel + 1);
			}
			else if (useColor){
				cubeInfo[i] = computeVectorEntropy(cubeFaceHists[i], nbins);
			}
		}
		else{
			std::cout << "entropy computation not defined! " << std::endl;
			exit(0);
		}

	}
	return;
}