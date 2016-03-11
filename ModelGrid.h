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
};
#endif