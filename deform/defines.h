#ifndef MODES_H
#define MODES_H

#include "vector_types.h"
#include <stdio.h>
#include <string.h>

struct matrix4x4
{
	float4 v[4];

	matrix4x4(float* _v){
		memcpy(&v[0].x, _v, sizeof(float4) * 4);
	}

	matrix4x4(){}
};

#endif
