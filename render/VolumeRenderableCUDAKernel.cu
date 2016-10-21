#include "VolumeRenderableCUDAKernel.h"
#include <helper_math.h>
#include <iostream>
#include <TransformFunc.h>

#include <stdlib.h>


typedef struct
{
	float3 m[3];
} float3x3;

typedef struct
{
	float4 m[4];
} float4x4;

struct Ray
{
	float3 o;    // origin
	float3 d;    // direction
};

// texture

texture<float, 3, cudaReadModeElementType>  volumeTexValueForRC;

texture<float4, 3, cudaReadModeElementType>  volumeTexGradient;

surface<void, cudaSurfaceType3D> volumeSurfaceOut;

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
__constant__ float3 spacing;

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

// transform vector by matrix with translation
__device__
float4 mul(const float4x4 &M, const float4 &v)
{
	float4 r;
	r.w = dot(v, M.m[3]);
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);

	return r;
}

__device__
float3 mul(const float4x4 &M, const float3 &v)
{
	float4 v4 = make_float4(v, 1.0);
	float3 r;
	r.x = dot(v4, M.m[0]);
	r.y = dot(v4, M.m[1]);
	r.z = dot(v4, M.m[2]);
	return r;
}

__device__
float3 mul(const float3x3 &M, const float3 &v)
{
	float3 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	return r;
}

__device__
float4 divW(float4 v)
{
	float invW = 1 / v.w;
	return(make_float4(v.x * invW, v.y * invW, v.z * invW, 1.0f));
}



void VolumeRender_setVolume(const VolumeCUDA *vol)
{
	checkCudaErrors(cudaBindTextureToArray(volumeTexValueForRC, vol->content, vol->channelDesc));
}




void VolumeRender_setConstants(float *MVMatrix, float *MVPMatrix, float *invMVMatrix, float *invMVPMatrix, float* NormalMatrix, float* _transFuncP1, float* _transFuncP2, float* _la, float* _ld, float* _ls, float3* _spacing)
{
	size_t sizeof4x4Matrix = sizeof(float4)* 4;
	checkCudaErrors(cudaMemcpyToSymbol(c_MVMatrix, MVMatrix, sizeof4x4Matrix));
	checkCudaErrors(cudaMemcpyToSymbol(c_MVPMatrix, MVPMatrix, sizeof4x4Matrix));
	checkCudaErrors(cudaMemcpyToSymbol(c_invMVMatrix, invMVMatrix, sizeof4x4Matrix));
	checkCudaErrors(cudaMemcpyToSymbol(c_invMVPMatrix, invMVPMatrix, sizeof4x4Matrix));
	checkCudaErrors(cudaMemcpyToSymbol(c_NormalMatrix, NormalMatrix, sizeof(float3)* 3));

	checkCudaErrors(cudaMemcpyToSymbol(transFuncP1, _transFuncP1, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(transFuncP2, _transFuncP2, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(la, _la, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(ld, _ld, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(ls, _ls, sizeof(float)));

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
}

void VolumeRender_deinit()
{
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
	//float la = 1.0, ld = 1.0, ls = 1.0;
	float Shininess = 5;

	float3 light_in_eye = make_float3(0.0, 0.0, 0.0);

	float3 s = normalize(light_in_eye - pos_in_eye);
	float3 v = normalize(-pos_in_eye);
	float3 r = reflect(-s, normal);
	float3 ambient = a * la;
	//float sDotN = max(dot(s, normal), 0.0);
	float sDotN = abs(dot(s, normal));

	float3 diffuse = a * sDotN * ld;
	float3 spec = make_float3(0.0);
	//if (sDotN > 0.0)
	spec = a * pow(max(dot(r, v), 0.0f), Shininess)* ls;
	return ambient + diffuse + spec;
}


__device__ float3 GetColourXin(float v, float vmin, float vmax)
{
	float3 c = make_float3(1.0, 1.0, 1.0); // white
	float dv;

	if (v < vmin)
		v = vmin;
	if (v > vmax)
		v = vmax;
	dv = vmax - vmin;

	if (v < (vmin + 0.25 * dv)) {
		c.x = 0;
		c.y = 4 * (v - vmin) / dv;
	}
	else if (v < (vmin + 0.5 * dv)) {
		c.x = 0;
		c.z = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
	}
	else if (v < (vmin + 0.75 * dv)) {
		c.x = 4 * (v - vmin - 0.5 * dv) / dv;
		c.z = 0;
	}
	else {
		c.y = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
		c.z = 0;
	}

	return(c);
}

__device__ float3 GetColourDiverge(float v)
{
	//if (v > 0.8)v = (v-0.8)/2+0.8;//for NEK image
	//can be accelerated!!
	int pos = 0;
	bool notFound = true;
	while (pos < numColorTableItems - 1 && notFound) {
		if (colorTable[pos][0] <= v && colorTable[pos + 1][0] >= v)
			notFound = false;
		pos++;
	}
	float ratio = (v - colorTable[pos][0]) / (colorTable[pos + 1][0] - colorTable[pos][0]);


	float3 c = make_float3(
		ratio*(colorTable[pos + 1][1] - colorTable[pos][1]) + colorTable[pos][1],
		ratio*(colorTable[pos + 1][2] - colorTable[pos][2]) + colorTable[pos][2],
		ratio*(colorTable[pos + 1][3] - colorTable[pos][3]) + colorTable[pos][3]);

	return(c);
}


__global__ void d_render_preint(uint *d_output, uint imageW, uint imageH, float density, float brightness, float3 eyeInWorld, int3 volumeSize, int maxSteps, float tstep, bool useColor)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	//const int maxSteps = 1024*2;
	//const float tstep = 0.25f;

	const float opacityThreshold = 0.95f;

	const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
	const float3 boxMax = spacing*make_float3(volumeSize);
	//const float3 boxMin = make_float3(0.0f, 114.0f, 0.0f);
	//const float3 boxMax = spacing*make_float3(256, 115, 256);//for NEK image


	//pixel_Index = clamp( round(uv * num_Pixels - 0.5), 0, num_Pixels-1 );
	float u = ((x + 0.5) / (float)imageW)*2.0f - 1.0f;
	float v = ((y + 0.5) / (float)imageH)*2.0f - 1.0f;

	Ray eyeRay;
	eyeRay.o = eyeInWorld;
	float4 pixelInClip = make_float4(u, v, -1.0f, 1.0f);
	float3 pixelInWorld = make_float3(divW(mul(c_invMVPMatrix, pixelInClip)));
	eyeRay.d = normalize(pixelInWorld - eyeRay.o);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (!hit)
	{
		float4 sum = make_float4(0.0f);
		//sum = make_float4(0.5f, 0.9f, 0.2f, 1.0f);
		sum = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		d_output[y*imageW + x] = rgbaFloatToInt(sum);
		return;
	}

	//float4 sum1 = make_float4(0.0f);
	//sum1 = make_float4(0.9f, 0.2f, 0.5f, 1.0f);
	//d_output[y*imageW + x] = rgbaFloatToInt(sum1);
	//return;

	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

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


		float3 posInWorld = mul(c_MVMatrix, pos);
		if (length(normalInWorld) > lightingThr)
		{
			float3 normal_in_eye = normalize(mul(c_NormalMatrix, normalInWorld));
			col = make_float4(phongModel(cc, posInWorld, normal_in_eye), funcRes * 1.0);
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



__global__ void d_render_preint_withLensBlending(uint *d_output, uint imageW, uint imageH, float density, float brightness, float3 eyeInWorld, int3 volumeSize, int maxSteps, float tstep, bool useColor, float* dev_pts)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	const float opacityThreshold = 0.95f;

	const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
	const float3 boxMax = spacing*make_float3(volumeSize);

	float u = ((x + 0.5) / (float)imageW)*2.0f - 1.0f;
	float v = ((y + 0.5) / (float)imageH)*2.0f - 1.0f;

	Ray eyeRay;
	eyeRay.o = eyeInWorld;
	float4 pixelInClip = make_float4(u, v, -1.0f, 1.0f);
	float3 pixelInWorld = make_float3(divW(mul(c_invMVPMatrix, pixelInClip)));
	eyeRay.d = normalize(pixelInWorld - eyeRay.o);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (!hit)
	{
		float4 sum = make_float4(0.0f);
		sum = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		d_output[y*imageW + x] = rgbaFloatToInt(sum);
		return;
	}



	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;
	float lightingThr = 0.000001;

	float fragDepth = 1.0;

	const float4 lineCol = make_float4(0.82f, 0.31f, 0.67f, 1.0f)/1.2f;
	const float lineWidth = 1.0f;
	const float2 ws = make_float2(imageW, imageH);

	bool needBlend = false;

	float3 *points = (float3 *)dev_pts;
	float2 pointsScreen[8];
	float depth[8];
	for (int i = 0; i < 8; i++){
		float4 posInClip = divW(mul(c_MVPMatrix, make_float4(points[i], 1.0f)));
		depth[i] = posInClip.z / 2.0f + 0.5f;
		pointsScreen[i] = (make_float2(posInClip.x, posInClip.y) / 2.0f + 0.5f)*ws;
	}
	float2 pix = make_float2(x, y);
	for (int i = 0; i < 4; i++){
		float2 vec = pix - pointsScreen[i];

		float2 dir = normalize(pointsScreen[i + 4] - pointsScreen[i]);
		float dis = dot(vec, dir);
		
		float2 minordir = make_float2(-dir.y, dir.x);
		float disMinor = abs(dot(vec, minordir));

		if (disMinor < lineWidth && dis>0 && dis<length(pointsScreen[i + 4] - pointsScreen[i])){
			needBlend = true;

			//!!!estimation, not precise
			///see https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
			float3 v1 = eyeRay.o - points[i], v2 = points[i + 4] - points[i], v3 = normalize(cross(eyeRay.d, cross(eyeRay.d, v2)));
			tfar = min(tfar, length(cross(v2, v1)) / abs(dot(v2, v3)));
			break;
		}
	}
	if (!needBlend){
		for (int i = 0; i < 4; i++){
			float2 vec = pix - pointsScreen[i];
			int j = i + 1;
			if (j >= 4)
				j = 0;
			float2 dir = normalize(pointsScreen[j] - pointsScreen[i]);
			float dis = dot(vec, dir);

			float2 minordir = make_float2(-dir.y, dir.x);
			float disMinor = abs(dot(vec, minordir));

			if (disMinor < lineWidth && dis>0 && dis<length(pointsScreen[j] - pointsScreen[i])){
				needBlend = true;

				//!!!estimation, not precise
				///see https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
				float3 v1 = eyeRay.o - points[i], v2 = points[j] - points[i], v3 = normalize(cross(eyeRay.d, cross(eyeRay.d, v2)));
				tfar = min(tfar, length(cross(v2, v1)) / abs(dot(v2, v3)));
				break;
			}
		}
	}



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


		float3 posInWorld = mul(c_MVMatrix, pos);
		if (length(normalInWorld) > lightingThr)
		{
			float3 normal_in_eye = normalize(mul(c_NormalMatrix, normalInWorld));
			col = make_float4(phongModel(cc, posInWorld, normal_in_eye), funcRes * 1.0f);
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
			float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0f)));
			fragDepth = posInClip.z / 2.0f + 0.5f;
			break;
		}

		t += tstep;

		if (t > tfar){
			if (needBlend){
				sum = sum + lineCol*(1.0f - sum.w);
				float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0f)));
				fragDepth = posInClip.z / 2.0f + 0.5f;
			}
			else{
				float4 posInClip = divW(mul(c_MVPMatrix, make_float4(pos, 1.0f)));
				fragDepth = posInClip.z / 2.0f + 0.5f;
			}
			break;
		}

		pos += step;
	}

	sum *= brightness;
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

void VolumeRender_render(uint *d_output, uint imageW, uint imageH,
	float density, float brightness,
	float3 eyeInWorld, int3 volumeSize, int maxSteps, float tstep, bool useColor)
{
	dim3 blockSize = dim3(16, 16, 1);
	dim3 gridSize = dim3(iDivUp(imageW, blockSize.x), iDivUp(imageH, blockSize.y));

	d_render_preint << <gridSize, blockSize >> >(d_output, imageW, imageH, density, brightness, eyeInWorld, volumeSize, maxSteps, tstep, useColor);

	//clean what was used

	//checkCudaErrors(cudaUnbindTexture(volumeTexValueForRC));
	//checkCudaErrors(cudaUnbindTexture(volumeTexGradient));
}

void VolumeRender_render_withLensBlending(uint *d_output, uint imageW, uint imageH,
	float density, float brightness,
	float3 eyeInWorld, int3 volumeSize, int maxSteps, float tstep, bool useColor, std::vector<float3> lensPoints)
{
	dim3 blockSize = dim3(16, 16, 1);
	dim3 gridSize = dim3(iDivUp(imageW, blockSize.x), iDivUp(imageH, blockSize.y));

	float* dev_pts;
	cudaMalloc((void**)&dev_pts, sizeof(float3)* 8);
	cudaMemcpy(dev_pts, &(lensPoints[0]), sizeof(float3)* 8, cudaMemcpyHostToDevice);

	d_render_preint_withLensBlending << <gridSize, blockSize >> >(d_output, imageW, imageH, density, brightness, eyeInWorld, volumeSize, maxSteps, tstep, useColor, dev_pts);

	cudaFree(dev_pts);
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
	if (indy2 > y >= volumeSize.height - 1) indy2 = y >= volumeSize.height - 1;
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