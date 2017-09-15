#include "VolumeRenderableCUDAKernel.h"
#include <helper_math.h>
#include <iostream>
#include "TransformFunc.h"
#include "PositionBasedDeformProcessor.h"
#include "myDefineRayCasting.h"

#include <stdlib.h>

#include <thrust/device_vector.h>

#include <cubicTex3D.cu>

#define VOLUMERENDER_TF_PREINTSIZE    1024
#define VOLUMERENDER_TF_PREINTSTEPS   1024


// texture

texture<float, 3, cudaReadModeElementType>  volumeTexValueForRC;

texture<float4, 3, cudaReadModeElementType>  volumeTexGradient;

texture<unsigned short, 3, cudaReadModeElementType>  volumeLabelValue;

surface<void, cudaSurfaceType3D> volumeSurfaceOut;

texture<float, 3, cudaReadModeElementType>   tex_inputImageDepth;
texture<uchar4, 2, cudaReadModeElementType>   tex_inputImageColor;


__constant__ float4x4 c_MVMatrix;
__constant__ float4x4 c_MVPMatrix;
__constant__ float4x4 c_invMVMatrix;
__constant__ float4x4 c_invMVPMatrix;
__constant__ float3x3 c_NormalMatrix;
__constant__ float transFuncP1;
__constant__ float transFuncP2;
__constant__ float la;
__constant__ float ld;
__constant__ float ls;

__constant__ float density;
__constant__ float brightness,
__constant__ int maxSteps;
__constant__ float tstep;
__constant__ bool useColor;


__constant__ float3 spacing;

__constant__ bool useLabel = false;

__constant__ int numColorTableItems = 33;
__constant__ float colorTable[33][4] = {
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

__device__ float3 GetColourDiverge(float v)
{
	//if (v > 0.8)v = (v-0.8)/2+0.8;//for NEK image
	//can be accelerated!!
	int pos = 0;
	bool notFound = true;
	while (pos < numColorTableItems - 1 && notFound) {
		if (colorTable[pos][0] <= v && colorTable[pos + 1][0] >= v)
			notFound = false;
		else
			pos++;
	}
	float ratio = (v - colorTable[pos][0]) / (colorTable[pos + 1][0] - colorTable[pos][0]);


	float3 c = make_float3(
		ratio*(colorTable[pos + 1][1] - colorTable[pos][1]) + colorTable[pos][1],
		ratio*(colorTable[pos + 1][2] - colorTable[pos][2]) + colorTable[pos][2],
		ratio*(colorTable[pos + 1][3] - colorTable[pos][3]) + colorTable[pos][3]);

	return(c);
}

__constant__ int numColorTableItemsTomato = 5;
__constant__ float colorTableTomato[5][4] = {
	0, 0.0, 0.0, 0,
	30 / 255.0, 51 / 255.0, 8 / 255.0, 0 / 255.0,
	42 / 255.0, 255 / 255.0, 99 / 255.0, 71 / 255.0,
	68 / 255.0, 255 / 255.0, 212 / 255.0, 204.0 / 255.0,
	1.0, 1.0, 1.0, 1.0
};

__device__ float4 GetColourTomato(float v)
{
	//Now use cutOff
	int pos = 0;
	bool notFound = true;
	while (pos < numColorTableItemsTomato - 1 && notFound) {
		if (colorTableTomato[pos][0] <= v && colorTableTomato[pos + 1][0] >= v)
			notFound = false;
		else{
			pos++;
		}
	}
	float ratio = (v - colorTableTomato[pos][0]) / (colorTableTomato[pos + 1][0] - colorTableTomato[pos][0]);
	float3 c;

	if (pos > 0){
		c = make_float3(
			ratio*(colorTableTomato[pos + 1][1] - colorTableTomato[pos][1]) + colorTableTomato[pos][1],
			ratio*(colorTableTomato[pos + 1][2] - colorTableTomato[pos][2]) + colorTableTomato[pos][2],
			ratio*(colorTableTomato[pos + 1][3] - colorTableTomato[pos][3]) + colorTableTomato[pos][3]);	
	}
	else{
		c = make_float3(colorTableTomato[pos][1], colorTableTomato[pos][2], colorTableTomato[pos][3]);
	}

	if (pos > 1){
		return make_float4(c, 0.1);
	}
	else if (pos > 0){
		return make_float4(c, 0.05);
	}
	else
		return make_float4(c, 0);
}


__constant__ int numColorTableItemsColon = 2;
__constant__ float colorTableColon[2][4] = {
	-0.0001, 254 / 255.0, 133 / 255.0, 90 / 255.0,
	1.00001, 200 / 255.0, 83 / 255.0, 63 / 255.0
};

__device__ float3 GetColourColon(float v)
{
	int pos = 0;
	bool notFound = true;
	while (pos < numColorTableItemsColon - 1 && notFound) {
		if (colorTableColon[pos][0] <= v && colorTableColon[pos + 1][0] >= v)
			notFound = false;
		else
			pos++;
	}
	float ratio = (v - colorTableColon[pos][0]) / (colorTableColon[pos + 1][0] - colorTableColon[pos][0]);


	float3 c = make_float3(
		ratio*(colorTableColon[pos + 1][1] - colorTableColon[pos][1]) + colorTableColon[pos][1],
		ratio*(colorTableColon[pos + 1][2] - colorTableColon[pos][2]) + colorTableColon[pos][2],
		ratio*(colorTableColon[pos + 1][3] - colorTableColon[pos][3]) + colorTableColon[pos][3]);

	return(c);
}



/////NOTE!!! add code to delete them afterwards
static cudaArray *d_transferArray = 0;
static cudaArray *d_transferIntegrate = 0;

texture<float4, 1, cudaReadModeElementType>           transferIntegrateTex;
surface<void, 1>                                      transferIntegrateSurf;

texture<float4, cudaTextureType2DLayered, cudaReadModeElementType>   transferLayerPreintTex;
surface<void, cudaSurfaceType2DLayered>                             transferLayerPreintSurf;

// 1D transfer function texture
texture<float4, 1, cudaReadModeElementType>           transferTex;

//float curfuncRes = clamp((sample - transFuncP2) / (transFuncP1 - transFuncP2), 0.0, 1.0);
//float lastFuncRes = clamp((lastSample - transFuncP2) / (transFuncP1 - transFuncP2), 0.0, 1.0);


__global__ void
d_integrate_trapezoidal(cudaExtent extent, float transFuncP1, float transFuncP2)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;

	// for higher speed could use hierarchical approach for sum
	if (x >= extent.width)
	{
		return;
	}

	float stepsize = 1.0 / float(extent.width - 1);
	float to = float(x) * stepsize;

	float4 outclr = make_float4(0, 0, 0, 0);
	float incr = stepsize;

	//float funcRes = __saturatef((0.0 - transFuncP2) / (transFuncP1 - transFuncP2));
	//float4 lastval = make_float4(funcRes, funcRes, funcRes, funcRes);	// tex1D(transferTex, 0);
	float4 lastval = tex1D(transferTex, 0);

	float cur = incr;

	while (cur < to + incr * 0.5)
	{
		//float funcRes = __saturatef((cur - transFuncP2) / (transFuncP1 - transFuncP2));
		//float4 val = make_float4(funcRes, funcRes, funcRes, funcRes); // tex1D(transferTex, cur);
		float4 val = tex1D(transferTex, cur);

		float4 trapezoid = (lastval + val) / 2.0f;
		lastval = val;

		outclr += trapezoid;
		cur += incr;
	}

	// surface writes need byte offsets for x!
	surf1Dwrite(outclr, transferIntegrateSurf, x * sizeof(float4));
}


__global__ void
d_preintegrate(float steps, cudaExtent extent, float transFuncP1, float transFuncP2)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= extent.width || y >= extent.height)
	{
		return;
	}

	float sx = float(x) / float(extent.width);
	float sy = float(y) / float(extent.height);

	float smax = max(sx, sy);
	float smin = min(sx, sy);

	float4 iv;

	if (x != y)
	{
		// assumes square textures!
		float fracc = smax - smin;
		fracc = 1.0 / (fracc*steps);

		float4 intmax = tex1D(transferIntegrateTex, smax);
		float4 intmin = tex1D(transferIntegrateTex, smin);
		iv.x = (intmax.x - intmin.x)*fracc;
		iv.y = (intmax.y - intmin.y)*fracc;
		iv.z = (intmax.z - intmin.z)*fracc;
		iv.w = (intmax.w - intmin.w)*fracc;
		//iv.w = (1.0 - exp(-(intmax.w - intmin.w) * fracc));
	}
	else
	{
		//float funcRes = __saturatef((smin - transFuncP2) / (transFuncP1 - transFuncP2));
		//float4 sample = make_float4(funcRes, funcRes, funcRes, funcRes);  //tex1D(transferTex, smin);
		float4 sample = tex1D(transferTex, smin);

		iv.x = sample.x;
		iv.y = sample.y;
		iv.z = sample.z;
		iv.w = sample.w;
		//iv.w = (1.0 - exp(-sample.w));
	}

	//iv.x = __saturatef(iv.x);
	//iv.y = __saturatef(iv.y);
	//iv.z = __saturatef(iv.z);
	//iv.w = __saturatef(iv.w);

	// surface writes need byte offsets for x!
	surf2DLayeredwrite(iv, transferLayerPreintSurf, x * sizeof(float4), y, 0);
}


void updatePreIntTabelNew(cudaArray *d_transferFunc)
{
	cudaChannelFormatDesc channelFloat4 = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaBindTextureToArray(transferTex, d_transferFunc, channelFloat4));

	{
		cudaExtent extent = { VOLUMERENDER_TF_PREINTSTEPS, 0, 0 };
		dim3 blockSize(32, 1, 1);
		dim3 gridSize(iDivUp(extent.width, blockSize.x), 1, 1);
		d_integrate_trapezoidal << <gridSize, blockSize >> >(extent, 0,0);
	}
	{
		cudaExtent extent = { VOLUMERENDER_TF_PREINTSIZE, VOLUMERENDER_TF_PREINTSIZE, 1 };
		dim3 blockSize(16, 16, 1);
		dim3 gridSize(iDivUp(extent.width, blockSize.x), iDivUp(extent.height, blockSize.y), 1);
		d_preintegrate << <gridSize, blockSize >> >(float(VOLUMERENDER_TF_PREINTSTEPS), extent, 0,0);
		//d_preintegrate << <gridSize, blockSize >> >(4.0, extent, r1, r2);
	}
}


void initPreIntTabel()
{
	transferLayerPreintTex.normalized = true;
	transferLayerPreintTex.filterMode = cudaFilterModeLinear;
	transferLayerPreintTex.addressMode[0] = cudaAddressModeClamp;
	transferLayerPreintTex.addressMode[1] = cudaAddressModeClamp;

	cudaChannelFormatDesc channelFloat4 = cudaCreateChannelDesc<float4>();
	cudaExtent extent = { VOLUMERENDER_TF_PREINTSIZE, VOLUMERENDER_TF_PREINTSIZE, 1 };
	checkCudaErrors(cudaMalloc3DArray(&d_transferArray, &channelFloat4, extent, cudaArrayLayered | cudaArraySurfaceLoadStore));
	checkCudaErrors(cudaBindTextureToArray(transferLayerPreintTex, d_transferArray, channelFloat4));
	checkCudaErrors(cudaBindSurfaceToArray(transferLayerPreintSurf, d_transferArray, channelFloat4));

	transferIntegrateTex.normalized = true;
	transferIntegrateTex.filterMode = cudaFilterModeLinear;
	transferIntegrateTex.addressMode[0] = cudaAddressModeClamp;
	transferIntegrateTex.addressMode[1] = cudaAddressModeClamp;
	transferIntegrateTex.addressMode[2] = cudaAddressModeClamp;
	checkCudaErrors(cudaMallocArray(&d_transferIntegrate, &channelFloat4, VOLUMERENDER_TF_PREINTSTEPS, 0, cudaArraySurfaceLoadStore));
	checkCudaErrors(cudaBindTextureToArray(transferIntegrateTex, d_transferIntegrate, channelFloat4));
	checkCudaErrors(cudaBindSurfaceToArray(transferIntegrateSurf, d_transferIntegrate, channelFloat4));
}



void VolumeRender_setVolume(const VolumeCUDA *vol)
{
	checkCudaErrors(cudaBindTextureToArray(volumeTexValueForRC, vol->content, vol->channelDesc));
}

void VolumeRender_setLabelVolume(const VolumeCUDA *v)
{
	volumeLabelValue.normalized = false;
	volumeLabelValue.filterMode = cudaFilterModePoint;
	volumeLabelValue.addressMode[0] = cudaAddressModeBorder;
	volumeLabelValue.addressMode[1] = cudaAddressModeBorder;
	volumeLabelValue.addressMode[2] = cudaAddressModeBorder;

	checkCudaErrors(cudaBindTextureToArray(volumeLabelValue, v->content, v->channelDesc));
	
	bool trueVariable = true;
	checkCudaErrors(cudaMemcpyToSymbol(useLabel, &trueVariable, sizeof(bool)));
}


void VolumeRender_setConstants(float *MVMatrix, float *MVPMatrix, float *invMVMatrix, float *invMVPMatrix, float* NormalMatrix, float3* _spacing, RayCastingParameters* rcp)
{
	size_t sizeof4x4Matrix = sizeof(float4)* 4;
	checkCudaErrors(cudaMemcpyToSymbol(c_MVMatrix, MVMatrix, sizeof4x4Matrix));
	checkCudaErrors(cudaMemcpyToSymbol(c_MVPMatrix, MVPMatrix, sizeof4x4Matrix));
	checkCudaErrors(cudaMemcpyToSymbol(c_invMVMatrix, invMVMatrix, sizeof4x4Matrix));
	checkCudaErrors(cudaMemcpyToSymbol(c_invMVPMatrix, invMVPMatrix, sizeof4x4Matrix));
	checkCudaErrors(cudaMemcpyToSymbol(c_NormalMatrix, NormalMatrix, sizeof(float3)* 3));

	checkCudaErrors(cudaMemcpyToSymbol(transFuncP1, &(rcp->transFuncP1), sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(transFuncP2, &(rcp->transFuncP2), sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(la, &(rcp->la), sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(ld, &(rcp->ld), sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(ls, &(rcp->ls), sizeof(float)));

	checkCudaErrors(cudaMemcpyToSymbol(density, &(rcp->density), sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(brightness, &(rcp->brightness), sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(maxSteps, &(rcp->maxSteps), sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(tstep, &(rcp->tstep), sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(useColor, &(rcp->useColor), sizeof(bool)));

	checkCudaErrors(cudaMemcpyToSymbol(spacing, _spacing, sizeof(float3)));
}

void VolumeRender_init()
{
	// set texture parameters
	volumeTexGradient.normalized = false;
	volumeTexGradient.filterMode = cudaFilterModeLinear;      // linear interpolation
	volumeTexGradient.addressMode[0] = cudaAddressModeBorder;  // clamp texture coordinates
	volumeTexGradient.addressMode[1] = cudaAddressModeBorder;
	volumeTexGradient.addressMode[2] = cudaAddressModeBorder;

	volumeTexValueForRC.normalized = false;
	volumeTexValueForRC.filterMode = cudaFilterModeLinear;
	volumeTexValueForRC.addressMode[0] = cudaAddressModeBorder;
	volumeTexValueForRC.addressMode[1] = cudaAddressModeBorder;
	volumeTexValueForRC.addressMode[2] = cudaAddressModeBorder;
	
	tex_inputImageDepth.normalized = false;
	tex_inputImageDepth.filterMode = cudaFilterModePoint;
	tex_inputImageDepth.addressMode[0] = cudaAddressModeBorder;
	tex_inputImageDepth.addressMode[1] = cudaAddressModeBorder;
	tex_inputImageDepth.addressMode[2] = cudaAddressModeBorder;

	tex_inputImageColor.normalized = false;
	tex_inputImageColor.filterMode = cudaFilterModePoint;
	tex_inputImageColor.addressMode[0] = cudaAddressModeBorder;
	tex_inputImageColor.addressMode[1] = cudaAddressModeBorder;

	transferTex.normalized = true;
	transferTex.filterMode = cudaFilterModeLinear;
	transferTex.addressMode[0] = cudaAddressModeClamp;

	initPreIntTabel();
}

void VolumeRender_deinit()
{
	checkCudaErrors(cudaFreeArray(d_transferArray));
	checkCudaErrors(cudaFreeArray(d_transferIntegrate));
}


////////////////////////////////////////////////// Direct Volume Rendering////////////////////////////////////////

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
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


__device__ uint rgbaFloatToInt(float4 rgba)
{
	rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}


__device__ float3 phongModel(float3 a, float3 pos_in_eye, float3 normal){
	float Shininess = 5;// 25;

	//float3 light_in_eye = make_float3(0.0, 2.0, 0.0);
	//float3 light_in_eye = make_float3(0.0, -200.0, 0.0);
	float3 light_in_eye = make_float3(0.0, 200.0, 200);

	float3 s = normalize(light_in_eye - pos_in_eye);
	float3 v = normalize(-pos_in_eye);
	float3 r = reflect(-s, normal);
	float3 ambient = a * la;
	//float sDotN = max(dot(s, normal), 0.0);
	float sDotN = abs(dot(s, normal));

	float3 diffuse = a * sDotN * ld;
	float3 spec = make_float3(0.0);
	//if (sDotN > 0.0)
	//spec = a * pow(max(dot(r, v), 0.0f), Shininess)* ls;
	spec = a * pow(fabs(dot(r, v)), Shininess)* ls;
	return ambient + diffuse + spec;
}


__global__ void d_render(uint *d_output, uint imageW, uint imageH, float3 eyeInLocal, int3 volumeSize)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	const float opacityThreshold = 0.95f;

	const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
	const float3 boxMax = spacing*make_float3(volumeSize);
	//const float3 boxMin = make_float3(0.0f, 114.0f, 0.0f);
	//const float3 boxMax = spacing*make_float3(256, 115, 256);//for NEK image


	//pixel_Index = clamp( round(uv * num_Pixels - 0.5), 0, num_Pixels-1 );
	float u = ((x + 0.5) / (float)imageW)*2.0f - 1.0f;
	float v = ((y + 0.5) / (float)imageH)*2.0f - 1.0f;

	Ray eyeRay;
	eyeRay.o = eyeInLocal;
	float4 pixelInClip = make_float4(u, v, -1.0f, 1.0f);
	float3 pixelInWorld = make_float3(divW(mul(c_invMVPMatrix, pixelInClip)));
	eyeRay.d = normalize(pixelInWorld - eyeRay.o);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (tnear < 0.0f) tnear = 0.01f;     // clamp to near plane according to the projection matrix

	if (tfar<tnear)//	if (!hit)
	{
		float4 sum = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		d_output[y*imageW + x] = rgbaFloatToInt(sum);
		return;
	}

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;
	float lightingThr = 0.000001;

	float fragDepth = 1.0;


	for (int i = 0; i<maxSteps; i++)
	{
		float3 coord = pos / spacing;
		float sample = tex3D(volumeTexValueForRC, coord.x, coord.y, coord.z);
		float funcRes = clamp((sample - transFuncP2) / (transFuncP1 - transFuncP2), 0.0, 1.0);

		float3 normalInWorld = make_float3(tex3D(volumeTexGradient, coord.x, coord.y, coord.z)) / spacing;

		// lookup in transfer function texture
		float4 col;

		float3 cc;
		if (useColor)
			cc = GetColourDiverge(clamp(funcRes, 0.0f, 1.0f));
		else
			cc = make_float3(funcRes, funcRes, funcRes);


		float3 posInEye = mul(c_MVMatrix, pos);
		if (length(normalInWorld) > lightingThr)
		{
			float3 normal_in_eye = normalize(mul(c_NormalMatrix, normalInWorld));
			col = make_float4(phongModel(cc, posInEye, normal_in_eye), funcRes * 1.0);
		}
		else
		{
			col = make_float4(la*cc, funcRes);
		}

		col.w *= density;

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		// "over" operator for front-to-back blending
		sum = sum + col*(1.0f - sum.w);

		// exit early if opaque
		if (sum.w > opacityThreshold){
			float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			fragDepth = posInClip.z / 2.0 + 0.5;
			break;
		}

		t += tstep;

		if (t > tfar){
			float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			fragDepth = posInClip.z / 2.0 + 0.5;
			break;
		}

		pos += step;
	}

	sum *= brightness;
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
}


void VolumeRender_render(uint *d_output, uint imageW, uint imageH, float3 eyeInLocal, int3 volumeSize)
{
	dim3 blockSize = dim3(16, 16, 1);
	dim3 gridSize = dim3(iDivUp(imageW, blockSize.x), iDivUp(imageH, blockSize.y));

	d_render << <gridSize, blockSize >> >(d_output, imageW, imageH, eyeInLocal, volumeSize);

	//clean what was used

	//checkCudaErrors(cudaUnbindTexture(volumeTexValueForRC));
	//checkCudaErrors(cudaUnbindTexture(volumeTexGradient));
}


__global__ void d_OmniVolumeRender(uint *d_output, uint imageW, uint imageH, float3 eyeInLocal, int3 volumeSize)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	const float r_eyeCircle = 0.5;
	//const float f = 5; //focal length
	const float r_cylinder = 10;

	const float opacityThreshold = 0.95f;

	const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
	const float3 boxMax = spacing*make_float3(volumeSize);

	Ray eyeRay;

	//this treatment is putting MVMatrix as seeing from the middle of left eye and right eye
	float thetaDif = acos(r_eyeCircle / r_cylinder);//the rotation angle between  .. and ..
	float3 viewDir = normalize(make_float3(mul(c_invMVMatrix, make_float4(0, 0, -1, 0))));
	float3 upDir = normalize(make_float3(mul(c_invMVMatrix, make_float4(0, 1, 0, 0))));
	float3 sideDir = cross(viewDir, upDir);
	if (y < imageH / 2){
		//start the image from the invert direction of viewDir
		float rho = 2 * 3.1415926*x / (float)imageW;
		float theta = 3.1415926 * y / (float)(imageH/2) * 0.9; //intentionally times 0.9 to avoid distraction

		float3 dirFromCenter = cos(theta)*upDir + sin(theta) * (cos(rho)*(-viewDir) + sin(rho)*(-sideDir));
		float3 pointOnCylinder = eyeInLocal + r_cylinder * dirFromCenter;
		float3 leftEyeDirFromCenter = cos(theta)*upDir + sin(theta) * (cos(rho + thetaDif)*(-viewDir) + sin(rho + thetaDif)*(-sideDir));
		float3 leftEye = eyeInLocal + r_eyeCircle * leftEyeDirFromCenter;

		eyeRay.o = leftEye;
		eyeRay.d = normalize(pointOnCylinder - eyeRay.o);
	}
	else{
		//start the image from the invert direction of viewDir
		float rho = 2 * 3.1415926*x / (float)imageW;
		float theta = 3.1415926 * (y - imageH / 2) / (float)(imageH / 2) * 0.9; //intentionally times 0.9 to avoid distraction

		float3 dirFromCenter = cos(theta)*upDir + sin(theta) * (cos(rho)*(-viewDir) + sin(rho)*(-sideDir));
		float3 pointOnCylinder = eyeInLocal + r_cylinder * dirFromCenter;
		float3 leftEyeDirFromCenter = cos(theta)*upDir + sin(theta) * (cos(rho - thetaDif)*(-viewDir) + sin(rho - thetaDif)*(-sideDir));
		float3 leftEye = eyeInLocal + r_eyeCircle * leftEyeDirFromCenter;

		eyeRay.o = leftEye;
		eyeRay.d = normalize(pointOnCylinder - eyeRay.o);
	}

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (tnear < 0.0f) tnear = 0.01f;     // clamp to near plane according to the projection matrix

	if (tfar<tnear)//	if (!hit)
	{
		float4 sum = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		d_output[y*imageW + x] = rgbaFloatToInt(sum);
		return;
	}

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;
	float lightingThr = 0.000001;

	float fragDepth = 1.0;


	for (int i = 0; i<maxSteps; i++)
	{
		float3 coord = pos / spacing;
		float sample = tex3D(volumeTexValueForRC, coord.x, coord.y, coord.z);
		float funcRes = clamp((sample - transFuncP2) / (transFuncP1 - transFuncP2), 0.0, 1.0);

		float3 normalInWorld = make_float3(tex3D(volumeTexGradient, coord.x, coord.y, coord.z)) / spacing;

		// lookup in transfer function texture
		float4 col;

		float3 cc;
		float colDensity = 0;

		if (useColor){
			//for tomato
			float4 ret = GetColourTomato(clamp(sample, 0.0f, 1.0f));
			cc = make_float3(ret);
			////cc = GetColourTomato(clamp(funcRes, 0.0f, 1.0f));
			//colDensity = ret.w;
			colDensity = funcRes;

			////for colon
			//	cc = GetColourColon(funcRes);
			//colDensity = funcRes;
		}
		else{
			cc = make_float3(funcRes, funcRes, funcRes);
			colDensity = funcRes;
		}

		float3 posInEye = mul(c_MVMatrix, pos);
		if (length(normalInWorld) > lightingThr)
		{
			float3 normal_in_eye = normalize(mul(c_NormalMatrix, normalInWorld));
			col = make_float4(phongModel(cc, posInEye, normal_in_eye), colDensity);
		}
		else
		{
			col = make_float4(la*cc, colDensity);
		}

		col.w *= density;

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		// "over" operator for front-to-back blending
		sum = sum + col*(1.0f - sum.w);

		// exit early if opaque
		if (sum.w > opacityThreshold){
			float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			fragDepth = posInClip.z / 2.0 + 0.5;
			break;
		}

		t += tstep;

		if (t > tfar){
			float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			fragDepth = posInClip.z / 2.0 + 0.5;
			break;
		}

		pos += step;
	}

	sum *= brightness;
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

void OmniVolumeRender_render(uint *d_output, uint imageW, uint imageH, float3 eyeInLocal, int3 volumeSize)
{
	dim3 blockSize = dim3(16, 16, 1);
	dim3 gridSize = dim3(iDivUp(imageW, blockSize.x), iDivUp(imageH, blockSize.y));

	d_OmniVolumeRender << <gridSize, blockSize >> >(d_output, imageW, imageH, eyeInLocal, volumeSize);
}


//iossurface type rendering
__global__ void d_render_immer_iso(uint *d_output, uint imageW, uint imageH, float3 eyeInLocal, int3 volumeSize)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	const float opacityThreshold = 0.95f;

	const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
	const float3 boxMax = spacing*make_float3(volumeSize);

	//pixel_Index = clamp( round(uv * num_Pixels - 0.5), 0, num_Pixels-1 );
	float u = ((x + 0.5) / (float)imageW)*2.0f - 1.0f;
	float v = ((y + 0.5) / (float)imageH)*2.0f - 1.0f;

	Ray eyeRay;
	eyeRay.o = eyeInLocal;
	float4 pixelInClip = make_float4(u, v, -1.0f, 1.0f);
	float3 pixelInWorld = make_float3(divW(mul(c_invMVPMatrix, pixelInClip)));
	eyeRay.d = normalize(pixelInWorld - eyeRay.o);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (tnear < 0.0f) tnear = 0.01f;     // clamp to near plane according to the projection matrix

	if (tfar<tnear)
		//	if (!hit)
	{
		float4 sum = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		d_output[y*imageW + x] = rgbaFloatToInt(sum);
		return;
	}

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;
	float lightingThr = 0.000001;

	float fragDepth = 1.0;

	const int isoConst = 3;
	const float isoDis = 0.05;
	const float isoList[isoConst] = { 0.294, 0.61, 0.99 + isoDis };

	for (int i = 0; i<maxSteps; i++)
	{
		float3 coord = pos / spacing;
		float sample = tex3D(volumeTexValueForRC, coord.x, coord.y, coord.z);
		
		//float funcRes = clamp((sample - transFuncP2) / (transFuncP1 - transFuncP2), 0.0, 1.0);
		float funcRes = 0;
		float colDensity = 0;

		for (int j = 0; j < isoConst; j++){
			if (abs(sample - isoList[j]) < isoDis){
				//funcRes = sample;
				funcRes = j*1.0 / (isoConst-1);

				//funcRes = 1.0 - abs(sample - isoList[j]) / isoDis;
				//funcRes = clamp(funcRes + j/10.0f, 0.0, 1.0);
				//colDensity = 1.0;// clamp(funcRes + j / 10.0f, 0.0, 1.0);
				colDensity = 1.0 - abs(sample - isoList[j]) / isoDis;
				colDensity = clamp(colDensity + j / 10.0f, 0.0, 1.0);
				//colDensity = (j+1.0 )/ isoConst;
				break;
			}
			else if (sample < isoList[j]){
				funcRes = j*1.0 / (isoConst - 1);
				colDensity = j / 10.0f;
				break;
			}
		}


		float3 normalInWorld = make_float3(tex3D(volumeTexGradient, coord.x, coord.y, coord.z)) / spacing;

		// lookup in transfer function texture
		float4 col;

		float3 cc;
		unsigned short curlabel = 0;
		if(useLabel)
			curlabel = tex3D(volumeLabelValue, coord.x, coord.y, coord.z);

		//if (useColor)
			cc = GetColourDiverge(clamp(funcRes, 0.0f, 1.0f));
		//else{
		//	cc = make_float3(funcRes, funcRes, funcRes);

		//	if (useLabel && curlabel > 1)
		//	{
		//		cc = make_float3(funcRes, 0.0f, 0.0f);
		//	}
		//	else if (useLabel && curlabel > 0){
		//		cc = make_float3(funcRes, funcRes, 0.0f);
		//	}
		//}

		float3 posInEye = mul(c_MVMatrix, pos);
		if (length(normalInWorld) > lightingThr)
		{
			float3 normal_in_eye = normalize(mul(c_NormalMatrix, normalInWorld));
			col = make_float4(phongModel(cc, posInEye, normal_in_eye), colDensity);
		}
		else
		{
			col = make_float4(la*cc, colDensity);
		}

		col.w *= density;

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		// "over" operator for front-to-back blending
		sum = sum + col*(1.0f - sum.w);

		// exit early if opaque
		if (sum.w > opacityThreshold){
			float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			fragDepth = posInClip.z / 2.0 + 0.5;
			break;
		}

		t += tstep;

		if (t > tfar){
			float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			fragDepth = posInClip.z / 2.0 + 0.5;
			break;
		}

		pos += step;
	}

	sum *= brightness;
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

//the function is very similar with d_render(), except for more color map options specifically used by the deformation project
//can be combined at a proper time
__global__ void d_render_immer(uint *d_output, uint imageW, uint imageH,  float3 eyeInLocal, int3 volumeSize)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	const float opacityThreshold = 0.95f;

	const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
	const float3 boxMax = spacing*make_float3(volumeSize);

	//pixel_Index = clamp( round(uv * num_Pixels - 0.5), 0, num_Pixels-1 );
	float u = ((x + 0.5) / (float)imageW)*2.0f - 1.0f;
	float v = ((y + 0.5) / (float)imageH)*2.0f - 1.0f;

	Ray eyeRay;
	eyeRay.o = eyeInLocal;
	float4 pixelInClip = make_float4(u, v, -1.0f, 1.0f);
	float3 pixelInWorld = make_float3(divW(mul(c_invMVPMatrix, pixelInClip)));
	eyeRay.d = normalize(pixelInWorld - eyeRay.o);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (tnear < 0.0f) tnear = 0.01f;     // clamp to near plane according to the projection matrix

	if (tfar<tnear)
		//	if (!hit)
	{
		float4 sum = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		d_output[y*imageW + x] = rgbaFloatToInt(sum);
		return;
	}

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;
	float lightingThr = 0.000001;

	float fragDepth = 1.0;


	for (int i = 0; i<maxSteps; i++)
	{
		float3 coord = pos / spacing;
		float sample = tex3D(volumeTexValueForRC, coord.x, coord.y, coord.z);
		float funcRes = clamp((sample - transFuncP2) / (transFuncP1 - transFuncP2), 0.0, 1.0);

		float3 normalInWorld = make_float3(tex3D(volumeTexGradient, coord.x, coord.y, coord.z)) / spacing;

		// lookup in transfer function texture
		float4 col;

		float3 cc;
		float colDensity = 0;

		unsigned short curlabel = 1;
		
		if(useLabel)
			curlabel = tex3D(volumeLabelValue, coord.x, coord.y, coord.z);

		if (useColor){
			//for tomato
			float4 ret = GetColourTomato(clamp(sample, 0.0f, 1.0f));
			cc = make_float3(ret);
			////cc = GetColourTomato(clamp(funcRes, 0.0f, 1.0f));
			//colDensity = ret.w;
			colDensity = funcRes;


			////for colon
			//if (useLabel && curlabel > 0){
			//	cc = make_float3(0.0f, funcRes, funcRes);
			//}
			//else{
			//	cc = GetColourColon(funcRes);
			//}
			//colDensity = funcRes;
		}
		else{
			cc = make_float3(funcRes, funcRes, funcRes);
			//for baseline
			if (useLabel && curlabel > 1)
			{
				cc = make_float3(funcRes, 0.0f, 0.0f);
			}
			else if (useLabel && curlabel > 0){
				cc = make_float3(0.0f, funcRes, funcRes);
			}

			colDensity = funcRes;
		}

		float3 posInEye = mul(c_MVMatrix, pos);
		if (length(normalInWorld) > lightingThr)
		{
			float3 normal_in_eye = normalize(mul(c_NormalMatrix, normalInWorld));
			col = make_float4(phongModel(cc, posInEye, normal_in_eye), colDensity);
		}
		else
		{
			col = make_float4(la*cc, colDensity);
		}

		col.w *= tstep;

		col.w *= density;

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;

		// "over" operator for front-to-back blending
		sum = sum + col*(1.0f - sum.w);

		// exit early if opaque
		if (sum.w > opacityThreshold){
			float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			fragDepth = posInClip.z / 2.0 + 0.5;
			break;
		}

		t += tstep;

		if (t > tfar){
			float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			fragDepth = posInClip.z / 2.0 + 0.5;
			break;
		}

		pos += step;
	}

	sum *= brightness;
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
}


__device__ float3
getOrigianlSamplePos(int3 volumeSize, float3 pos,
float3 start, float3 end, float3 spacing, float r, float deformationScale, float deformationScaleVertical, float3 dir2nd)
{
	//if (pos.x >= volumeSize.width || pos.y >= volumeSize.height || pos.z >= volumeSize.depth
	//	|| pos.x < 0 || pos.y < 0 || pos.z < 0)
	//{
	//	return;
	//}

	float3 tunnelVec = normalize(end - start);
	float tunnelLength = length(end - start);

	float3 voxelVec = pos - start;
	float l = dot(voxelVec, tunnelVec);
	if (l > 0 && l < tunnelLength){
		float disToStart = length(voxelVec);
		float l2 = dot(voxelVec, dir2nd);
		if (abs(l2) < deformationScaleVertical){
			float3 prjPoint = start + l*tunnelVec + l2*dir2nd;
			float3 dir = normalize(pos - prjPoint);
			float dis = length(pos - prjPoint);

			if (dis < r){
				return make_float3(-1, -1, r - dis);
			}
			else if (dis < deformationScale){
				float3 prjPoint = start + l*tunnelVec + l2*dir2nd;
				float3 dir = normalize(pos - prjPoint);
				float3 samplePos = prjPoint + dir* (dis - r) / (deformationScale - r)*deformationScale;
				return samplePos;
				//samplePos /= spacing;
			}
			else{
				return make_float3(-1, -1, -1);
			}
		}
		else{
			return make_float3(-1, -1, -1);
		}
	}
	else{
		return make_float3(-1, -1, -1);
	}
}


__global__ void d_render_immer_withClipping(uint *d_output, uint imageW, uint imageH,
	float3 eyeInLocal, int3 volumeSize, bool useSplineInterpolation,
	float3 tunnelStart, float3 tunnelEnd, float3 rectVerticalDir, float r, float deformationScale, float deformationScaleVertical)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	const float opacityThreshold = 0.95f;

	const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
	const float3 boxMax = spacing*make_float3(volumeSize);

	//pixel_Index = clamp( round(uv * num_Pixels - 0.5), 0, num_Pixels-1 );
	float u = ((x + 0.5) / (float)imageW)*2.0f - 1.0f;
	float v = ((y + 0.5) / (float)imageH)*2.0f - 1.0f;

	Ray eyeRay;
	eyeRay.o = eyeInLocal;
	float4 pixelInClip = make_float4(u, v, -1.0f, 1.0f);
	float3 pixelInWorld = make_float3(divW(mul(c_invMVPMatrix, pixelInClip)));
	eyeRay.d = normalize(pixelInWorld - eyeRay.o);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (tnear < 0.0f) tnear = 0.01f;     // clamp to near plane according to the projection matrix

	if (tfar<tnear)
		//	if (!hit)
	{
		float4 sum = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		d_output[y*imageW + x] = rgbaFloatToInt(sum);
		return;
	}

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;
	float lightingThr = 0.000001;

	float fragDepth = 1.0;

		for (int i = 0; i < maxSteps; i++)
		{
			float3 coord = pos / spacing;
			float sample = tex3D(volumeTexValueForRC, coord.x, coord.y, coord.z);
			float funcRes = clamp((sample - transFuncP2) / (transFuncP1 - transFuncP2), 0.0, 1.0);

			float3 normalInWorld = make_float3(tex3D(volumeTexGradient, coord.x, coord.y, coord.z)) / spacing;

			// lookup in transfer function texture
			float4 col;

			float3 cc;
			float colDensity = 0;

			unsigned short curlabel = 1;

			if (useLabel)
				curlabel = tex3D(volumeLabelValue, coord.x, coord.y, coord.z);

			if (useColor){
				//for tomato
				float4 ret = GetColourTomato(clamp(sample, 0.0f, 1.0f));
				cc = make_float3(ret);
				////cc = GetColourTomato(clamp(funcRes, 0.0f, 1.0f));
				//colDensity = ret.w;
				colDensity = funcRes;


				////for colon
				//if (useLabel && curlabel > 0){
				//	cc = make_float3(0.0f, funcRes, funcRes);
				//}
				//else{
				//	cc = GetColourColon(funcRes);
				//}
				//colDensity = funcRes;
			}
			else{
				cc = make_float3(funcRes, funcRes, funcRes);
				//for baseline
				if (useLabel && curlabel > 1)
				{
					cc = make_float3(funcRes, 0.0f, 0.0f);
				}
				else if (useLabel && curlabel > 0){
					cc = make_float3(0.0f, funcRes, funcRes);
				}

				colDensity = funcRes;
			}

			float3 posInEye = mul(c_MVMatrix, pos);
			if (length(normalInWorld) > lightingThr)
			{
				float3 normal_in_eye = normalize(mul(c_NormalMatrix, normalInWorld));
				col = make_float4(phongModel(cc, posInEye, normal_in_eye), colDensity);
			}
			else
			{
				col = make_float4(la*cc, colDensity);
			}

			if (r > 0){
				float3 origianlPos = getOrigianlSamplePos(volumeSize, pos,
					tunnelStart, tunnelEnd, spacing, r, deformationScale, deformationScaleVertical, rectVerticalDir);
				if (origianlPos.z > -0.01){
					col = make_float4(0, 0, 0, 0);

				}
			}

			col.w *= tstep;

			col.w *= density;

			// pre-multiply alpha
			col.x *= col.w;
			col.y *= col.w;
			col.z *= col.w;

			// "over" operator for front-to-back blending
			sum = sum + col*(1.0f - sum.w);

			// exit early if opaque
			if (sum.w > opacityThreshold){
				float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
				fragDepth = posInClip.z / 2.0 + 0.5;
				break;
			}

			t += tstep;

			if (t > tfar){
				float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
				fragDepth = posInClip.z / 2.0 + 0.5;
				break;
			}

			pos += step;
		}
	
	
	sum *= brightness;
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
}


__global__ void d_render_preint(uint *d_output, uint imageW, uint imageH, float3 eyeInLocal, int3 volumeSize, bool useSplineInterpolation)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	const float opacityThreshold = 0.95f;

	const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
	const float3 boxMax = spacing*make_float3(volumeSize);

	//pixel_Index = clamp( round(uv * num_Pixels - 0.5), 0, num_Pixels-1 );
	float u = ((x + 0.5) / (float)imageW)*2.0f - 1.0f;
	float v = ((y + 0.5) / (float)imageH)*2.0f - 1.0f;

	Ray eyeRay;
	eyeRay.o = eyeInLocal;
	float4 pixelInClip = make_float4(u, v, -1.0f, 1.0f);
	float3 pixelInWorld = make_float3(divW(mul(c_invMVPMatrix, pixelInClip)));
	eyeRay.d = normalize(pixelInWorld - eyeRay.o);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (tnear < 0.0f) tnear = 0.01f;     // clamp to near plane according to the projection matrix

	if (tfar<tnear)
		//	if (!hit)
	{
		float4 sum = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		d_output[y*imageW + x] = rgbaFloatToInt(sum);
		return;
	}

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;

	float tstepv = tstep;
	float3 step = eyeRay.d*tstepv;
	float lightingThr = 0.0001;

	float fragDepth = 1.0;

	float3 coord = pos / spacing;
	float lastSample = tex3D(volumeTexValueForRC, coord.x, coord.y, coord.z);
	float3 last_normalInWorld = make_float3(tex3D(volumeTexGradient, coord.x, coord.y, coord.z)) / spacing;
	float3 last_normal_in_eye = normalize(mul(c_NormalMatrix, last_normalInWorld));
	t += tstepv;
	pos += step;



	for (int i = 1; i<maxSteps; i++)
	{
		if (i > 1 && i % 256 == 0 && tstepv < 0.51) {
			//increase step size to improve performance, when the ray is already far, since then we do not need short step size that much then
			//but still limit the tstepv not to exceed 1
			tstepv *= 2;
			step *= 2;
		}

		float3 coord = pos / spacing;
		
		float sample;
		if (useSplineInterpolation)
			sample = cubicTex3D(volumeTexValueForRC, coord.x, coord.y, coord.z);
		else
			sample = tex3D(volumeTexValueForRC, coord.x, coord.y, coord.z);

		float4 cpreint = tex2DLayered(transferLayerPreintTex, sample, lastSample, 0);

		float3 cc;
		
		cc = make_float3(cpreint.x, cpreint.y, cpreint.z);
		cc.x = __saturatef(cc.x);
		cc.y = __saturatef(cc.y);
		cc.z = __saturatef(cc.z);
		
		//float colDensity = __saturatef(cpreint.w);
		float colDensity = cpreint.w * tstepv;
		//float colDensity = cpreint.w;
		
		float4 col;
		
		float3 normalInWorld;
		if (useSplineInterpolation){
			normalInWorld = make_float3(cubicTex3D_1st_derivative_x(volumeTexValueForRC, coord),
				cubicTex3D_1st_derivative_y(volumeTexValueForRC, coord),
				cubicTex3D_1st_derivative_z(volumeTexValueForRC, coord)) / spacing;
			normalInWorld = make_float3(1, 1, 0);
		}
		else{
			normalInWorld = make_float3(tex3D(volumeTexGradient, coord.x, coord.y, coord.z)) / spacing;
		}
		float3 posInEye = mul(c_MVMatrix, pos);
		if (length(normalInWorld) > lightingThr)// && abs(sample - lastSample) > 0.0001)
		{
			float3 normal_in_eye = normalize(mul(c_NormalMatrix, normalInWorld));
			col = make_float4(phongModel(cc, posInEye, normal_in_eye), colDensity);

			//col = make_float4(phongModel(cc, posInEye, (last_normal_in_eye + normal_in_eye)/2), colDensity); //need to solve the case when last_normal_in_eye + normal_in_eye == 0
			last_normal_in_eye = normal_in_eye;
			//using average of last_normal_in_eye and normal_in_eye is just one option, and may not be the best.
			//see the other paper
		}
		else
		{
			col = make_float4(la*cc, colDensity);
			last_normal_in_eye = make_float3(0,0,0);
		}
		
		col.w *= density;

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		// "over" operator for front-to-back blending

		sum = sum + col*(1.0f - sum.w);

		// exit early if opaque
		if (sum.w > opacityThreshold){
			float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			fragDepth = posInClip.z / 2.0 + 0.5;
			break;
		}

		t += tstepv;

		if (t > tfar){
			float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			fragDepth = posInClip.z / 2.0 + 0.5;
			break;
		}

		pos += step;

		lastSample = sample;
	}

	sum *= brightness;
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
}



__global__ void d_render_preint_coloringDeformedElement(uint *d_output, uint imageW, uint imageH,
	float3 eyeInLocal, int3 volumeSize, bool useSplineInterpolation, 
	float3 tunnelStart, float3 tunnelEnd, float3 rectVerticalDir, float r, float deformationScale, float deformationScaleVertical)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	const float opacityThreshold = 0.95f;

	const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
	const float3 boxMax = spacing*make_float3(volumeSize);

	//pixel_Index = clamp( round(uv * num_Pixels - 0.5), 0, num_Pixels-1 );
	float u = ((x + 0.5) / (float)imageW)*2.0f - 1.0f;
	float v = ((y + 0.5) / (float)imageH)*2.0f - 1.0f;

	Ray eyeRay;
	eyeRay.o = eyeInLocal;
	float4 pixelInClip = make_float4(u, v, -1.0f, 1.0f);
	float3 pixelInWorld = make_float3(divW(mul(c_invMVPMatrix, pixelInClip)));
	eyeRay.d = normalize(pixelInWorld - eyeRay.o);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (tnear < 0.0f) tnear = 0.01f;     // clamp to near plane according to the projection matrix

	if (tfar<tnear)
		//	if (!hit)
	{
		float4 sum = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		d_output[y*imageW + x] = rgbaFloatToInt(sum);
		return;
	}

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;

	float tstepv = tstep;
	float3 step = eyeRay.d*tstepv;
	float lightingThr = 0.0001;

	float fragDepth = 1.0;

	float3 coord = pos / spacing;
	float lastSample = tex3D(volumeTexValueForRC, coord.x, coord.y, coord.z);
	float3 last_normalInWorld = make_float3(tex3D(volumeTexGradient, coord.x, coord.y, coord.z)) / spacing;
	float3 last_normal_in_eye = normalize(mul(c_NormalMatrix, last_normalInWorld));
	t += tstepv;
	pos += step;



	for (int i = 1; i<maxSteps; i++)
	{
		if (i > 1 && i % 256 == 0 && tstepv < 0.51) {
			//increase step size to improve performance, when the ray is already far, since then we do not need short step size that much then
			//but still limit the tstepv not to exceed 1
			tstepv *= 2;
			step *= 2;
		}
		float3 coord = pos / spacing;

		float sample;
		if (useSplineInterpolation)
			sample = cubicTex3D(volumeTexValueForRC, coord.x, coord.y, coord.z);
		else
			sample = tex3D(volumeTexValueForRC, coord.x, coord.y, coord.z);

		float4 cpreint = tex2DLayered(transferLayerPreintTex, sample, lastSample, 0);

		float3 cc;

		cc = make_float3(cpreint.x, cpreint.y, cpreint.z);
		cc.x = __saturatef(cc.x);
		cc.y = __saturatef(cc.y);
		cc.z = __saturatef(cc.z);
		
		
		if (r>0){
			float3 origianlPos = getOrigianlSamplePos(volumeSize, pos,
				tunnelStart, tunnelEnd, spacing, r, deformationScale, deformationScaleVertical, rectVerticalDir);
			if (origianlPos.x > -0.01){
				const float3 errorColor = make_float3(1, 0, 0);
				float maxDis = deformationScale / 2.0;
				float ratio = length(pos - origianlPos) / maxDis*0.8; //0.2 is selected parameter
				cc = cc*(1 - ratio) + errorColor*ratio;
			}
			////to deal with partial volume effect. may be useful
			//else if (origianlPos.z > 0){
			//	if (origianlPos.z / r < 0.1){
			//		const float3 errorColor = make_float3(1, 0, 0);
			//		float maxDis = deformationScale / 2;
			//		float ratio = maxDis / maxDis*0.8; //0.2 is selected parameter
			//		cc = cc*(1 - ratio) + errorColor*ratio;
			//	}
			//}
		}
		

		//float colDensity = __saturatef(cpreint.w);
		float colDensity = cpreint.w * tstepv;

		float4 col;
		float3 normalInWorld = make_float3(tex3D(volumeTexGradient, coord.x, coord.y, coord.z)) / spacing;
		float3 posInEye = mul(c_MVMatrix, pos);
		if (length(normalInWorld) > lightingThr)// && abs(sample - lastSample) > 0.0001)
		{
			float3 normal_in_eye = normalize(mul(c_NormalMatrix, normalInWorld));
			col = make_float4(phongModel(cc, posInEye, normal_in_eye), colDensity);

			//col = make_float4(phongModel(cc, posInEye, (last_normal_in_eye + normal_in_eye) / 2), colDensity);//need to solve the case when last_normal_in_eye + normal_in_eye == 0
			//using average of last_normal_in_eye and normal_in_eye is just one option, and may not be the best.
			//another possible way is to use cubicTex3D_1st_derivative_x(texture tex, float3 coord)

			last_normal_in_eye = normal_in_eye;
		}
		else
		{
			col = make_float4(la*cc, colDensity);
			last_normal_in_eye = make_float3(0, 0, 0);
		}


		col.w *= density;

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		// "over" operator for front-to-back blending

		sum = sum + col*(1.0f - sum.w);

		// exit early if opaque
		if (sum.w > opacityThreshold){
			float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			fragDepth = posInClip.z / 2.0 + 0.5;
			break;
		}

		t += tstepv;

		if (t > tfar){
			float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			fragDepth = posInClip.z / 2.0 + 0.5;
			break;
		}

		pos += step;

		lastSample = sample;
	}

	sum *= brightness;
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
}


//used specifically for immersive deformation project
void VolumeRender_renderImmer(uint *d_output, uint imageW, uint imageH,
	float3 eyeInLocal, int3 volumeSize, RayCastingParameters* rcp, PositionBasedDeformProcessor* pd, bool usePreInt, bool useSplineInterpolation, bool useCliping)
{
	dim3 blockSize = dim3(16, 16, 1);
	dim3 gridSize = dim3(iDivUp(imageW, blockSize.x), iDivUp(imageH, blockSize.y));

	if (useCliping){
		d_render_immer_withClipping << <gridSize, blockSize >> >(d_output, imageW, imageH, eyeInLocal, volumeSize, useSplineInterpolation,
			pd->getTunnelStart(), pd->getTunnelEnd(), pd->getRectVerticalDir(), pd->r, pd->getDeformationScale(), pd->getDeformationScaleVertical());
	}
	else if (usePreInt){
		if (pd->isColoringDeformedPart){ //care!! pd must be non-0 here
			d_render_preint_coloringDeformedElement << <gridSize, blockSize >> >(d_output, imageW, imageH, eyeInLocal, volumeSize, useSplineInterpolation,
				pd->getTunnelStart(), pd->getTunnelEnd(), pd->getRectVerticalDir(), pd->r, pd->getDeformationScale(), pd->getDeformationScaleVertical());
		}
		else{
			d_render_preint << <gridSize, blockSize >> >(d_output, imageW, imageH, eyeInLocal, volumeSize, useSplineInterpolation);
		}
	}
	else{
		d_render_immer << <gridSize, blockSize >> >(d_output, imageW, imageH, eyeInLocal, volumeSize);
		//d_render_immer_iso << <gridSize, blockSize >> >(d_output, imageW, imageH, eyeInLocal, volumeSize); //for Neghip
	}

}


__global__ void
d_computeGradient(cudaExtent volumeSize)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
	{
		return;
	}

	float4 grad = make_float4(0.0);

	int indz1 = z - 2, indz2 = z + 2;
	if (indz1 < 0)	indz1 = 0;
	if (indz2 > volumeSize.depth - 1) indz2 = volumeSize.depth - 1;
	grad.z = (tex3D(volumeTexValueForRC, x + 0.5, y + 0.5, indz2 + 0.5) - tex3D(volumeTexValueForRC, x + 0.5, y + 0.5, indz1 + 0.5)) / (indz2 - indz1);

	int indy1 = y - 2, indy2 = y + 2;
	if (indy1 < 0)	indy1 = 0;
	if (indy2 > volumeSize.height - 1) indy2 = volumeSize.height - 1;
	grad.y = (tex3D(volumeTexValueForRC, x + 0.5, indy2 + 0.5, z + 0.5) - tex3D(volumeTexValueForRC, x + 0.5, indy1 + 0.5, z + 0.5)) / (indy2 - indy1);

	int indx1 = x - 2, indx2 = x + 2;
	if (indx1 < 0)	indx1 = 0;
	if (indx2 > volumeSize.width - 1) indx2 = volumeSize.width - 1;
	grad.x = (tex3D(volumeTexValueForRC, indx2 + 0.5, y + 0.5, z + 0.5) - tex3D(volumeTexValueForRC, indx1 + 0.5, y + 0.5, z + 0.5)) / (indx2 - indx1);

	surf3Dwrite(grad, volumeSurfaceOut, x * sizeof(float4), y, z);
}

void VolumeRender_computeGradient(const VolumeCUDA *volumeCUDAInput, VolumeCUDA *volumeCUDAGradient)
{
	cudaExtent size = volumeCUDAInput->size;
	unsigned int dim = 32;
	dim3 blockSize(dim, dim, 1);
	dim3 gridSize(iDivUp(size.width, blockSize.x), iDivUp(size.height, blockSize.y), iDivUp(size.depth, blockSize.z));

	checkCudaErrors(cudaBindTextureToArray(volumeTexValueForRC, volumeCUDAInput->content, volumeCUDAInput->channelDesc));
	checkCudaErrors(cudaBindSurfaceToArray(volumeSurfaceOut, volumeCUDAGradient->content));

	d_computeGradient << <gridSize, blockSize >> >(size);

	checkCudaErrors(cudaUnbindTexture(volumeTexValueForRC));
}


void VolumeRender_setGradient(const VolumeCUDA *gradVol)
{
	checkCudaErrors(cudaBindTextureToArray(volumeTexGradient, gradVol->content, gradVol->channelDesc));
}

//similar with d_render_immer. might be faster if directly use the result of d_render_immer
__global__ void d_LabelProcessor(uint imageW, uint imageH, float density, float brightness, float3 eyeInLocal, int3 volumeSize, int maxSteps, float tstep, bool useColor,char* screenMark)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	if (!screenMark[y*imageW + x]){
			return;
	}

	const float opacityThreshold = 0.95f;

	const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
	const float3 boxMax = spacing*make_float3(volumeSize);

	//pixel_Index = clamp( round(uv * num_Pixels - 0.5), 0, num_Pixels-1 );
	float u = ((x + 0.5) / (float)imageW)*2.0f - 1.0f;
	float v = ((y + 0.5) / (float)imageH)*2.0f - 1.0f;

	Ray eyeRay;
	eyeRay.o = eyeInLocal;
	float4 pixelInClip = make_float4(u, v, -1.0f, 1.0f);
	float3 pixelInWorld = make_float3(divW(mul(c_invMVPMatrix, pixelInClip)));
	eyeRay.d = normalize(pixelInWorld - eyeRay.o);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (tnear < 0.0f) tnear = 0.01f;     // clamp to near plane according to the projection matrix

	if (tfar<tnear)
	{
		return;
	}


	//tnear = (tfar - tnear) / 80 + tnear;
	maxSteps = maxSteps / 2;
	float temp = (tfar - tnear) / maxSteps;
	tstep = fmax(tstep, temp);

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;
	float lightingThr = 0.000001;

	float fragDepth = 1.0;


	for (int i = 0; i<maxSteps; i++)
	{
		float3 coord = pos / spacing;
		float sample = tex3D(volumeTexValueForRC, coord.x, coord.y, coord.z);
		float funcRes = clamp((sample - transFuncP2) / (transFuncP1 - transFuncP2), 0.0, 1.0);

		float3 normalInWorld = make_float3(tex3D(volumeTexGradient, coord.x, coord.y, coord.z)) / spacing;

		// lookup in transfer function texture
		float4 col;

		float3 cc;
		unsigned short curlabel = tex3D(volumeLabelValue, coord.x, coord.y, coord.z);

		if (useLabel && curlabel > 1)
		{
			cc = make_float3(1.0f, 0.0f, 0.0f);
		}
		else if (useLabel && curlabel > 0){
			cc = make_float3(1.0f, 1.0f, 0.0f);
		}
		else{
			if (useColor)
				cc = GetColourDiverge(clamp(funcRes, 0.0f, 1.0f));
			else
				cc = make_float3(funcRes, funcRes, funcRes);
		}

		float3 posInEye = mul(c_MVMatrix, pos);
		if (length(normalInWorld) > lightingThr)
		{
			float3 normal_in_eye = normalize(mul(c_NormalMatrix, normalInWorld));
			col = make_float4(phongModel(cc, posInEye, normal_in_eye), funcRes * 1.0);
		}
		else
		{
			col = make_float4(la*cc, funcRes);
		}

		col.w *= density;

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		// "over" operator for front-to-back blending
		sum = sum + col*(1.0f - sum.w);

		if (sum.w > opacityThreshold*0.8){
			unsigned short ll = 1;

			surf3Dwrite(ll, volumeSurfaceOut, floor(coord.x) * sizeof(unsigned short), floor(coord.y), floor(coord.z));
		}


		// exit early if opaque
		if (sum.w > opacityThreshold){
			float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			fragDepth = posInClip.z / 2.0 + 0.5;
			break;
		}

		t += tstep;

		if (t > tfar){
			float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			fragDepth = posInClip.z / 2.0 + 0.5;
			break;
		}

		pos += step;
	}

	sum *= brightness;

}

//this function can be placed at the LabelVolumeProcessor.cpp, but put it here to save some constant setting
void LabelProcessor(uint imageW, uint imageH,
	float density, float brightness,
	float3 eyeInLocal, int3 volumeSize, int maxSteps, float tstep, bool useColor, char* screenMark, VolumeCUDA *volumeCUDALabel)
{
	dim3 blockSize = dim3(16, 16, 1);
	dim3 gridSize = dim3(iDivUp(imageW, blockSize.x), iDivUp(imageH, blockSize.y));

	checkCudaErrors(cudaBindSurfaceToArray(volumeSurfaceOut, volumeCUDALabel->content));

	d_LabelProcessor << <gridSize, blockSize >> >(imageW, imageH, density, brightness, eyeInLocal, volumeSize, maxSteps, tstep, useColor, screenMark);
}





__global__ void d_render_withDepthInput(uint *d_output, uint imageW, uint imageH, float3 eyeInLocal, int3 volumeSize, float densityBonus)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	const float opacityThreshold = 0.95f;

	const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
	const float3 boxMax = spacing*make_float3(volumeSize);
	//const float3 boxMin = make_float3(0.0f, 114.0f, 0.0f);
	//const float3 boxMax = spacing*make_float3(256, 115, 256);//for NEK image


	float u = ((x + 0.5) / (float)imageW)*2.0f - 1.0f;
	float v = ((y + 0.5) / (float)imageH)*2.0f - 1.0f;

	float inputDepth = tex3D(tex_inputImageDepth, x, y, 0.5);
	uchar4 inputColor_uchar = tex2D(tex_inputImageColor, x, y);
	float4 inputColor = make_float4(inputColor_uchar.x / 255.0, inputColor_uchar.y / 255.0, inputColor_uchar.z / 255.0, inputColor_uchar.w / 255.0);


	Ray eyeRay;
	eyeRay.o = eyeInLocal;
	float4 pixelInClip = make_float4(u, v, -1.0f, 1.0f);
	float3 pixelInWorld = make_float3(divW(mul(c_invMVPMatrix, pixelInClip)));
	eyeRay.d = normalize(pixelInWorld - eyeRay.o);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (tnear < 0.0f) tnear = 0.01f;     // clamp to near plane according to the projection matrix

	if (tfar<tnear)
		//	if (!hit)
	{
		float4 sum = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		if (inputDepth < 1.0){
			//cases when previous renderable result is outside of the bounding box
			d_output[y*imageW + x] = rgbaFloatToInt(inputColor * density * brightness * densityBonus);
		}
		else{
			d_output[y*imageW + x] = rgbaFloatToInt(sum);
		}
		return;
	}

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;
	float lightingThr = 0.000001;

	float fragDepth = 1.0;


	for (int i = 0; i<maxSteps; i++)
	{
		float3 coord = pos / spacing;
		float sample = tex3D(volumeTexValueForRC, coord.x, coord.y, coord.z);
		float funcRes = clamp((sample - transFuncP2) / (transFuncP1 - transFuncP2), 0.0, 1.0);

		float3 normalInWorld = make_float3(tex3D(volumeTexGradient, coord.x, coord.y, coord.z)) / spacing;

		// lookup in transfer function texture
		float4 col;

		float3 cc;
		if (useColor)
			cc = GetColourDiverge(clamp(funcRes, 0.0f, 1.0f));
		else
			cc = make_float3(funcRes, funcRes, funcRes);


		float3 posInEye = mul(c_MVMatrix, pos);
		if (length(normalInWorld) > lightingThr)
		{
			float3 normal_in_eye = normalize(mul(c_NormalMatrix, normalInWorld));
			col = make_float4(phongModel(cc, posInEye, normal_in_eye), funcRes * 1.0);
		}
		else
		{
			col = make_float4(la*cc, funcRes);
		}

		col.w *= density;

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		// "over" operator for front-to-back blending
		sum = sum + col*(1.0f - sum.w);

		float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
		float curDepth = posInClip.z / 2.0 + 0.5;
		if (curDepth > inputDepth){
			inputColor.w *= density * densityBonus; //give extra density

			// pre-multiply alpha
			inputColor.x *= inputColor.w;
			inputColor.y *= inputColor.w;
			inputColor.z *= inputColor.w;

			sum = sum + inputColor*(1.0f - sum.w);
			fragDepth = curDepth;
			// note!!here for now, ignore the cases that the ray can pass the input image and continue integrating
			break;
		}


		// exit early if opaque
		if (sum.w > opacityThreshold){
			//float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			fragDepth = curDepth;
			break;
		}

		t += tstep;

		if (t > tfar){
			//float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0)));
			if (inputDepth < 1.0){
				//cases when previous renderable result is further than the bounding box
				inputColor.w *= density * 1; //give extra density

				// pre-multiply alpha
				inputColor.x *= inputColor.w;
				inputColor.y *= inputColor.w;
				inputColor.z *= inputColor.w;

				sum = sum + inputColor*(1.0f - sum.w);
				fragDepth = inputDepth;
			}
			else{
				fragDepth = curDepth;
			}
			break;
		}

		pos += step;
	}

	sum *= brightness;
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

void setInputImageInfo(const cudaArray_t c_inputImageDepthArray, const cudaArray_t c_inputImageColorArray)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	checkCudaErrors(cudaBindTextureToArray(tex_inputImageDepth, c_inputImageDepthArray, channelDesc));
	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	checkCudaErrors(cudaBindTextureToArray(tex_inputImageColor, c_inputImageColorArray, channelDesc2));
	
}

void VolumeRender_renderWithDepthInput(uint *d_output, uint imageW, uint imageH,
	float density, float brightness, float3 eyeInLocal, int3 volumeSize, int maxSteps, float tstep, bool useColor, float densityBonus)
{
	dim3 blockSize = dim3(16, 16, 1);
	dim3 gridSize = dim3(iDivUp(imageW, blockSize.x), iDivUp(imageH, blockSize.y));

	d_render_withDepthInput << <gridSize, blockSize >> >(d_output, imageW, imageH, eyeInLocal, volumeSize, densityBonus);
}


