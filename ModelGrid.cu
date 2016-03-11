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

float* ModelGrid::GetE()
{
	return gridMesh->EL;
}


int ModelGrid::GetLNumber()
{
	return gridMesh->l_number;
}

void ModelGrid::Initialize(float time_step)
{
	return gridMesh->Initialize(time_step);
}

void ModelGrid::Update(float time_step, float lensCenter[3])
{
	gridMesh->Update(time_step, 64, lensCenter);
}

float3 ModelGrid::GetGridMin()
{
	return gridMesh->gridMin;
}

float3 ModelGrid::GetGridMax()
{
	return gridMesh->gridMax;
}

int3 ModelGrid::GetNumSteps()
{
	return make_int3(gridMesh->nStep[0], gridMesh->nStep[1], gridMesh->nStep[2]);
}

float ModelGrid::GetStep()
{
	return gridMesh->step;
}

void ModelGrid::SetElasticity(float* v)
{
	std::copy(v, v + gridMesh->tet_number, gridMesh->EL);
}
