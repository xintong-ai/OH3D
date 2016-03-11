#include <ModelGrid.h>
#include <GridMesh.h>
ModelGrid::ModelGrid(float dmin[3], float dmax[3], int n)
{
	gridMesh = new GridMesh<float>(dmin, dmax, n);
}

int ModelGrid::GetTNumber()
{
	return gridMesh->t_number;
}

int* ModelGrid::GetT()
{
	return gridMesh->T;
}

float* ModelGrid::GetX()
{
	return gridMesh->X;
}

int ModelGrid::GetNumber()
{
	return gridMesh->number;
}

unsigned int* ModelGrid::GetL()
{
	return gridMesh->L;
}

int ModelGrid::GetLNumber()
{
	return gridMesh->l_number;
}
