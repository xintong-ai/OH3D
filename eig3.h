//copied from http://sachinashanbhag.blogspot.com/2010/10/computing-eigenvectors-of-3x3-symmetric.html

/* Eigen-decomposition for symmetric 3x3 real matrices.
   Public domain, copied from the public domain Java library JAMA. */

#ifndef _eig_h

/* Symmetric matrix A => eigenvectors in columns of V, corresponding
   eigenvalues in d. */
//template<typename T>
//void eigen_decomposition(T A[9], T V[9], T d[3]);


/* Eigen decomposition code for symmetric 3x3 matrices, copied from the public
domain Java Matrix library JAMA. */

#include <math.h>

#ifdef MAX
#undef MAX
#endif

#define MAX(a, b) ((a)>(b)?(a):(b))

#define n 3

template<typename T>
static T hypot2(T x, T y) {
	return sqrt(x*x + y*y);
}

// Symmetric Householder reduction to tridiagonal form.

template<typename T>
static void tred2(T V[9], T d[n], T e[n]) {

	//  This is derived from the Algol procedures tred2 by
	//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
	//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
	//  Fortran subroutine in EISPACK.

	for (int j = 0; j < n; j++) {
		d[j] = V[(n - 1) * n + j];
	}

	// Householder reduction to tridiagonal form.

	for (int i = n - 1; i > 0; i--) {

		// Scale to avoid under/overflow. 

		T scale = 0.0;
		T h = 0.0;
		for (int k = 0; k < i; k++) {
			scale = scale + fabs(d[k]);
		}
		if (scale == 0.0) {
			e[i] = d[i - 1];
			for (int j = 0; j < i; j++) {
				d[j] = V[(i - 1) * n + j];
				V[i * n + j] = 0.0;
				V[j * n + i] = 0.0;
			}
		}
		else {

			// Generate Householder vector.

			for (int k = 0; k < i; k++) {
				d[k] /= scale;
				h += d[k] * d[k];
			}
			T f = d[i - 1];
			T g = sqrt(h);
			if (f > 0) {
				g = -g;
			}
			e[i] = scale * g;
			h = h - f * g;
			d[i - 1] = f - g;
			for (int j = 0; j < i; j++) {
				e[j] = 0.0;
			}

			// Apply similarity transformation to remaining columns.

			for (int j = 0; j < i; j++) {
				f = d[j];
				V[j * n + i] = f;
				g = e[j] + V[j * n + j] * f;
				for (int k = j + 1; k <= i - 1; k++) {
					g += V[(k)* n + j] * d[k];
					e[k] += V[(k)* n + j] * f;
				}
				e[j] = g;
			}
			f = 0.0;
			for (int j = 0; j < i; j++) {
				e[j] /= h;
				f += e[j] * d[j];
			}
			T hh = f / (h + h);
			for (int j = 0; j < i; j++) {
				e[j] -= hh * d[j];
			}
			for (int j = 0; j < i; j++) {
				f = d[j];
				g = e[j];
				for (int k = j; k <= i - 1; k++) {
					V[(k)* n + j] -= (f * e[k] + g * d[k]);
				}
				d[j] = V[(i - 1) * n + j];
				V[i * n + j] = 0.0;
			}
		}
		d[i] = h;
	}

	// Accumulate transformations.

	for (int i = 0; i < n - 1; i++) {
		V[(n - 1) * n + i] = V[i * n + i];
		V[i * n + i] = 1.0;
		T h = d[i + 1];
		if (h != 0.0) {
			for (int k = 0; k <= i; k++) {
				d[k] = V[k * n + i + 1] / h;
			}
			for (int j = 0; j <= i; j++) {
				T g = 0.0;
				for (int k = 0; k <= i; k++) {
					g += V[k * n + i + 1] * V[(k)* n + j];
				}
				for (int k = 0; k <= i; k++) {
					V[(k)* n + j] -= g * d[k];
				}
			}
		}
		for (int k = 0; k <= i; k++) {
			V[k * n + i + 1] = 0.0;
		}
	}
	for (int j = 0; j < n; j++) {
		d[j] = V[(n - 1) * n + j];
		V[(n - 1) * n + j] = 0.0;
	}
	V[(n - 1) * n + n - 1] = 1.0;
	e[0] = 0.0;
}

// Symmetric tridiagonal QL algorithm.

template<typename T>
static void tql2(T V[9], T d[n], T e[n]) {

	//  This is derived from the Algol procedures tql2, by
	//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
	//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
	//  Fortran subroutine in EISPACK.

	for (int i = 1; i < n; i++) {
		e[i - 1] = e[i];
	}
	e[n - 1] = 0.0;

	T f = 0.0;
	T tst1 = 0.0;
	T eps = pow(2.0, -52.0);
	for (int l = 0; l < n; l++) {

		// Find small subdiagonal element

		tst1 = MAX(tst1, fabs(d[l]) + fabs(e[l]));
		int m = l;
		while (m < n) {
			if (fabs(e[m]) <= eps*tst1) {
				break;
			}
			m++;
		}

		// If m == l, d[l] is an eigenvalue,
		// otherwise, iterate.

		if (m > l) {
			int iter = 0;
			do {
				iter = iter + 1;  // (Could check iteration count here.)

				// Compute implicit shift

				T g = d[l];
				T p = (d[l + 1] - g) / (2.0 * e[l]);
				T r = hypot2(p, 1.0f);
				if (p < 0) {
					r = -r;
				}
				d[l] = e[l] / (p + r);
				d[l + 1] = e[l] * (p + r);
				T dl1 = d[l + 1];
				T h = g - d[l];
				for (int i = l + 2; i < n; i++) {
					d[i] -= h;
				}
				f = f + h;

				// Implicit QL transformation.

				p = d[m];
				T c = 1.0;
				T c2 = c;
				T c3 = c;
				T el1 = e[l + 1];
				T s = 0.0;
				T s2 = 0.0;
				for (int i = m - 1; i >= l; i--) {
					c3 = c2;
					c2 = c;
					s2 = s;
					g = c * e[i];
					h = c * p;
					r = hypot2(p, e[i]);
					e[i + 1] = s * r;
					s = e[i] / r;
					c = p / r;
					p = c * d[i] - s * g;
					d[i + 1] = h + s * (c * g + s * d[i]);

					// Accumulate transformation.

					for (int k = 0; k < n; k++) {
						h = V[k * n + i + 1];
						V[k * n + i + 1] = s * V[k * n + i] + c * h;
						V[k * n + i] = c * V[k * n + i] - s * h;
					}
				}
				p = -s * s2 * c3 * el1 * e[l] / dl1;
				e[l] = s * p;
				d[l] = c * p;

				// Check for convergence.

			} while (fabs(e[l]) > eps*tst1);
		}
		d[l] = d[l] + f;
		e[l] = 0.0;
	}

	// Sort eigenvalues and corresponding vectors.

	for (int i = 0; i < n - 1; i++) {
		int k = i;
		T p = d[i];
		for (int j = i + 1; j < n; j++) {
			if (d[j] < p) {
				k = j;
				p = d[j];
			}
		}
		if (k != i) {
			d[k] = d[i];
			d[i] = p;
			for (int j = 0; j < n; j++) {
				p = V[j * n + i];
				V[j * n + i] = V[j * n + k];
				V[j * n + k] = p;
			}
		}
	}
}

template<typename T>
void eigen_decomposition(T A[9], T V[9], T d[n]) {
	T e[n];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			V[i * n + j] = A[i * n + j];
		}
	}
	tred2(V, d, e);
	tql2(V, d, e);
}


#endif
