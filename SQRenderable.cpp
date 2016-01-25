#include "SQRenderable.h"
#include <teem/ten.h>
//#include <teem/ellMacros.h>
//#include <teem/limn.h>
//#include <teem/ell.h>
using namespace std;

SQRenderable::SQRenderable(vector<float4> pos, vector<float> val) :
GlyphRenderable(pos)
{
	/* input variables */
	double ten[7] = { 1.0, 1.0, 0.0, 0.0, -0.5, 0.0, -0.3 }; /* tensor coefficients */
	double eps = 1e-4; /* small value >0; defines the smallest tensor
					   * norm at which tensor orientation is still meaningful */

	/* example code starts here */
	double evals[3], evecs[9], uv[2], abc[3], norm;

	tenEigensolve_d(evals, evecs, ten);
	tenGlyphBqdUvEval(uv, evals);
	tenGlyphBqdAbcUv(abc, uv, 3.0);
	norm = ELL_3V_LEN(evals);
	if (norm<eps) {
		double weight = norm / eps;
		abc[0] = weight*abc[0] + (1 - weight);
		abc[1] = weight*abc[1] + (1 - weight);
		abc[2] = weight*abc[2] + (1 - weight);
	}

	/* input variable */
	int glyphRes = 20; /* controls how fine the tesselation will be */

	/* example code starts here */
	limnPolyData *lpd = limnPolyDataNew();
	limnPolyDataSpiralBetterquadric(lpd, (1 << limnPolyDataInfoNorm),
		abc[0], abc[1], abc[2], 0.0,
		2 * glyphRes, glyphRes);
	limnPolyDataVertexNormals(lpd);

	double absevals[3];
	for (int k = 0; k<3; k++)
		absevals[k] = fabs(evals[k]);
	double trans[16] = { absevals[0] * evecs[0], absevals[1] * evecs[3],
		absevals[2] * evecs[6], 0,
		absevals[0] * evecs[1], absevals[1] * evecs[4],
		absevals[2] * evecs[7], 0,
		absevals[0] * evecs[2], absevals[1] * evecs[5],
		absevals[2] * evecs[8], 0,
		0, 0, 0, 1 };
	unsigned int zone = tenGlyphBqdZoneUv(uv);
	if (0 == zone || 5 == zone || 6 == zone || 7 == zone || 8 == zone) {
		/* we need an additional rotation */
		double ZtoX[16] = { 0, 0, 1, 0,
			0, 1, 0, 0,
			-1, 0, 0, 0,
			0, 0, 0, 1 };
		ell_4m_mul_d(trans, trans, ZtoX);
	}
	double gltrans[16];
	ELL_4M_TRANSPOSE(gltrans, trans); /* OpenGL expects column-major format */
}
