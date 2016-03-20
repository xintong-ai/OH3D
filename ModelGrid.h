#ifndef MODEL_GRID_H
#define MODEL_GRID_H

template <class TYPE>
class GridMesh;

class ModelGrid
{
	GridMesh<float>* gridMesh;
public:
	ModelGrid(float dmin[3], float dmax[3], int n);

	int GetTNumber();
	int* GetT();
	float* GetX();
	int GetNumber();
	unsigned int* GetL();
	int GetLNumber();
	void Initialize(float time_step);
	void Update(float time_step, float lensCenter[3], float lenDir[3]);
	float3 GetGridMin();
	float3 GetGridMax();
	int3 GetNumSteps();
	float GetStep();
	float* GetE();
	int* GetTet();
	void SetElasticity(float* v);
};
#endif