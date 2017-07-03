#include "Trace.h"
//#include <fstream>
using namespace std;

void Trace::save(const char* fileName)
{
	FILE *pFile;
	pFile = fopen(fileName, "wb");
	if (pFile == NULL) {
		fputs("Saving Trace file error: ", stderr);
		fputs(fileName, stderr);
		fputs("\n", stderr);
		exit(1);
	}
	int n = pos.size();
	fwrite((char*)&n, sizeof(int), 1, pFile);
	fwrite((char*)&pos[0], sizeof(float3), n, pFile);

	fclose(pFile);
}

void UncertainTrace::save(const char* fileName)
{
	FILE *pFile;
	pFile = fopen(fileName, "wb");
	if (pFile == NULL) {
		fputs("Saving uncertain Trace file error: ", stderr);
		fputs(fileName, stderr);
		fputs("\n", stderr);
		exit(1);
	}
	int n = pos.size();
	fwrite((char*)&n, sizeof(int), 1, pFile);
	fwrite((char*)&pos[0], sizeof(float3), n, pFile);
	if (stds.size() == n){
		fwrite((char*)&stds[0], sizeof(float3), n, pFile);
	}
	else{
		float3 temp = make_float3(0, 0, 0);
		for (int i = 0; i < n; i++){
			fwrite((char*)&temp, sizeof(float3), 1, pFile);
		}
	}

	fclose(pFile);
}