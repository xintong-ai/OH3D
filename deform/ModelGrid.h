#ifndef MODEL_GRID_H
#define MODEL_GRID_H
#include <vector>
template <class TYPE>
class GridMesh;

//added by cl
enum GRID_TYPE{
	UNIFORM_GRID,
	LINESPLIT_UNIFORM_GRID
};

class ModelGrid
{
	GridMesh<float>* gridMesh;
	std::vector<float4> vBaryCoord;
	std::vector<int> vIdx;
	const float	time_step = 1 / 30.0;
public:
	ModelGrid(float dmin[3], float dmax[3], int n);
	void UpdatePointCoords(float4* v, int n);
	void InitGridDensity(float4* v, int n);

	int GetTNumber();
	int* GetT();
	float* GetX();
	int GetNumber();
	unsigned int* GetL();
	int GetLNumber();
	void Initialize(float time_step);
	void Update(float lensCenter[3], float lenDir[3], float focusRatio, float radius);
	float3 GetGridMin();
	float3 GetGridMax();
	int3 GetNumSteps();
	float GetStep();
	float* GetE();
	int* GetTet();
	void SetElasticity(float* v);
};
#endif