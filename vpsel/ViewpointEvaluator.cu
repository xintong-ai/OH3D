
#include <iostream>
#include "ViewpointEvaluator.h"
#include "Volume.h"
#include "TransformFunc.h"
#include <thrust/device_vector.h>


texture<float, 3, cudaReadModeElementType>  volumeVal;


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

ViewpointEvaluator::ViewpointEvaluator(std::shared_ptr<Volume> v)
{
	volume = v;

	volumeVal.normalized = false;
	volumeVal.filterMode = cudaFilterModeLinear;
	volumeVal.addressMode[0] = cudaAddressModeBorder;
	volumeVal.addressMode[1] = cudaAddressModeBorder;
	volumeVal.addressMode[2] = cudaAddressModeBorder;

	GPU_setConstants(&rcp.transFuncP1, &rcp.transFuncP2, &rcp.la, &rcp.ld, &rcp.ls, &(volume->spacing));
	GPU_setVolume(&(volume->volumeCuda));

	rcp.tstep = 1.0; //generally don't need to sample beyond each voxel

	cudaMalloc(&d_hist, sizeof(float)*nbins);
}


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


void ViewpointEvaluator::setSpherePoints(int n)
{
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
	cudaMalloc(&d_sphereSamples, sizeof(float)*numSphereSample*3);
	cudaMemcpy(d_sphereSamples, (float*)(&sphereSamples[0]), sizeof(float)*numSphereSample * 3, cudaMemcpyHostToDevice);
}





__global__ void d_computeVolumeVisibility(float density, float brightness,
	float3 eyeInWorld, int3 volumeSize, int maxSteps, float tstep, bool useColor, float * r)
{

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if (x >= volumeSize.x || y >= volumeSize.y || z >= volumeSize.z)
	{
		return;
	}

	const float opacityThreshold = 0.95f;

	Ray eyeRay;
	eyeRay.o = eyeInWorld;
	float3 voxelInWorld = make_float3(x, y, z);
	eyeRay.d = normalize(voxelInWorld - eyeRay.o);

	float tnear, tfar;
	tnear = 0.01f;	//!!!NOTE!!! this tnear is not in the clip space but in the original space
	tfar = length(voxelInWorld - eyeRay.o);

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;
	float lightingThr = 0.000001;


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
		sum = sum + col*(1.0f - sum.w);

		// exit early if opaque
		if (sum.w > opacityThreshold){
			break;
		}

		t += tstep;

		if (t > tfar){
			break;
		}

		pos += step;
	}

	sum *= brightness;

	float vj;
	if (t > tfar){
		vj = sum.w;
	}
	else{
		vj = 0.0f;
	}

	r[z*volumeSize.x * volumeSize.y + y*volumeSize.x + x] = vj;
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


float ViewpointEvaluator::computeVolumewhiseEntropy(float3 eyeInWorld, float * d_r)
{
	cudaExtent size = make_cudaExtent(volume->size.x, volume->size.y, volume->size.z);
	unsigned int dim = 32;
	dim3 blockSize(dim, dim, 1);
	dim3 gridSize(iDivUp(size.width, blockSize.x), iDivUp(size.height, blockSize.y), iDivUp(size.depth, blockSize.z));

	d_computeVolumeVisibility << <gridSize, blockSize >> >(rcp.density, rcp.brightness, eyeInWorld, volume->size, rcp.maxSteps, rcp.tstep, rcp.useColor, d_r);

	thrust::device_vector< float > iVec(d_r, d_r + volume->size.x*volume->size.y*volume->size.z);
	float sum = thrust::reduce(iVec.begin(), iVec.end(), (float)0, thrust::plus<float>());
	thrust::transform(iVec.begin(), iVec.end(), iVec.begin(), functor_computeEntropy(sum));
	float ret = thrust::reduce(iVec.begin(), iVec.end(), (float)0, thrust::plus<float>());

	return ret;
};



__global__ void d_computeSphereUtility(float density, float brightness,
	float3 eyeInWorld, int3 volumeSize, int maxSteps, float tstep, bool useColor, float * r, int numSphereSample, float *sphereSamples, float *hist, int nbins)
{

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numSphereSample)	return;

	const float opacityThreshold = 0.95f;


	Ray eyeRay;
	eyeRay.o = eyeInWorld;
	eyeRay.d = make_float3(sphereSamples[3 * i], sphereSamples[3 * i + 1], sphereSamples[3 * i + 2]);

	float tnear, tfar;
	const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
	const float3 boxMax = spacing*make_float3(volumeSize); 
	intersectBox2(eyeRay, boxMin, boxMax, &tnear, &tfar);
	tnear = 0.01f;	//!!!NOTE!!! this tnear is not in the clip space but in the original space

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;
	float lightingThr = 0.000001;


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
		sum = sum + col*(1.0f - sum.w);

		// exit early if opaque
		if (sum.w > opacityThreshold){
			break;
		}

		t += tstep;

		if (t > tfar){
			break;
		}

		pos += step;
	}

	sum *= brightness;

	float uv;
	if (t > tfar){
		uv = sum.w;
	}
	else{
		uv = 0.0f;
	}

	r[i] = uv;

	// !!! this is true only when we know uv is in [0,1] !!!
	int bin = min((int)(uv*nbins), nbins - 1);
	atomicAdd(hist + bin, 1);
}

float ViewpointEvaluator::computeSpherewhiseEntropy(float3 eyeInWorld, float * d_r)
{
	int threadsPerBlock = 64;
	int blocksPerGrid = (numSphereSample + threadsPerBlock - 1) / threadsPerBlock;

	cudaMemset(d_hist, 0, sizeof(float)*nbins);

	d_computeSphereUtility << <blocksPerGrid, threadsPerBlock >> >(rcp.density, rcp.brightness, eyeInWorld, volume->size, rcp.maxSteps, rcp.tstep, rcp.useColor, d_r, numSphereSample, d_sphereSamples, d_hist, nbins);

	thrust::device_vector< float > iVec(d_hist, d_hist + nbins);
	thrust::transform(iVec.begin(), iVec.end(), iVec.begin(), functor_computeEntropy((float)numSphereSample));
	float ret = thrust::reduce(iVec.begin(), iVec.end(), (float)0, thrust::plus<float>());

	return ret;
}