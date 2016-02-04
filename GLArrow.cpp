#include "GLArrow.h"
#include "vector_types.h"
#include "vector_functions.h"

#define _USE_MATH_DEFINES
#include <math.h>

GLArrow::GLArrow()
{
	//build the regular mesh in the constructor
	//using namespace std;
	float x, y, z;
	for (int i = 0; i < nDivision; i++) {
		float ang1 = 2.0 * M_PI*i / nDivision;
		float ang2 = 2.0 * M_PI*((i+1)%nDivision) / nDivision;
		float x1 = cos(ang1) * width;
		float y1 = sin(ang1) * width;
		float x2 = cos(ang2) * width;
		float y2 = sin(ang2) * width;
		float z0 = -1 * range;
		float z1 = z0 +2 * range*rodRatio;
		float z2 = 1 * range;
		grids.push_back(make_float4(0, 0, z0, 1.0));
		grids.push_back(make_float4(x1, y1, z0, 1.0));
		grids.push_back(make_float4(x2, y2, z0, 1.0));

		grids.push_back(make_float4(x1, y1, z0, 1.0));
		grids.push_back(make_float4(x2, y2, z0, 1.0));
		grids.push_back(make_float4(x2, y2, z1, 1.0));

		grids.push_back(make_float4(x1, y1, z0, 1.0));
		grids.push_back(make_float4(x1, y1, z1, 1.0));
		grids.push_back(make_float4(x2, y2, z1, 1.0));

		grids.push_back(make_float4(x1, y1, z1, 1.0));
		grids.push_back(make_float4(x2, y2, z1, 1.0));
		grids.push_back(make_float4(x2 * 2, y2 * 2, z1, 1.0));

		grids.push_back(make_float4(x1, y1, z1, 1.0));
		grids.push_back(make_float4(x1 * 2, y1 * 2, z1, 1.0));
		grids.push_back(make_float4(x2 * 2, y2 * 2, z1, 1.0));

		grids.push_back(make_float4(x1 * 2, y1 * 2, z1, 1.0));
		grids.push_back(make_float4(x2 * 2, y2 * 2, z1, 1.0));
		grids.push_back(make_float4(0, 0, z2, 1.0));

		normals.push_back(make_float3(0, 0, -1));
		normals.push_back(make_float3(0, 0, -1));
		normals.push_back(make_float3(0, 0, -1));

		float3 normalRodSurface = normalize(make_float3((x1 + x2) / 2, (y1 + y2) / 2, 0));

		normals.push_back(normalRodSurface);
		normals.push_back(normalRodSurface);
		normals.push_back(normalRodSurface);
		normals.push_back(normalRodSurface);
		normals.push_back(normalRodSurface);
		normals.push_back(normalRodSurface);


		normals.push_back(make_float3(0, 0, -1));
		normals.push_back(make_float3(0, 0, -1));
		normals.push_back(make_float3(0, 0, -1));
		normals.push_back(make_float3(0, 0, -1));
		normals.push_back(make_float3(0, 0, -1));
		normals.push_back(make_float3(0, 0, -1));

		float3 normalTipSurface = normalize(cross(make_float3(2 * x2 - 2 * x1, 2 * y2 - 2 * y1, 0), make_float3(-2 * x1, -2 * x2, z2 - z1)));

		normals.push_back(normalTipSurface);
		normals.push_back(normalTipSurface);
		normals.push_back(normalTipSurface);
	}

	indices.resize(18 * nDivision);
	for (int i = 0; i < 18 * nDivision; i++) {
		indices[i] = i;
	}
	
	/*
	grids.push_back(make_float4(0, 0, 0, 1));
	grids.push_back(make_float4(1, 0, 0, 1));
	grids.push_back(make_float4(0, 1, 0, 1));
	grids.push_back(make_float4(0, 0, 1, 1));
	normals.push_back(normalize(make_float3(-1, -1, -1)));
	normals.push_back(make_float3(1, 0, 0));
	normals.push_back(make_float3(0, 1, 0));
	normals.push_back(make_float3(0, 0, 1));
	//nVerts.push_back(4);


	indices.push_back(0);
	indices.push_back(1);
	indices.push_back(2);
	indices.push_back(0);
	indices.push_back(2);
	indices.push_back(3);
	indices.push_back(0);
	indices.push_back(1);
	indices.push_back(3);
	indices.push_back(1);
	indices.push_back(2);
	indices.push_back(3);
	*/
}

