#include "PlyMeshReader.h"
#include "PolyMesh.h"
#include <helper_math.h>

//for linux
#include <string.h>
#include <float.h>


void PlyMeshReader::LoadPLY(const char* filename, std::shared_ptr<PolyMesh> polyMesh)
{
	//mesh* m = new mesh();

	float* Vertex_Buffer;
	float* Normal_Buffer;
	TotalConnectedTriangles = 0;
	TotalConnectedPoints = 0;
	const char* pch = strstr(filename, ".ply");

	if (pch != NULL)
	{
		FILE* file = fopen(filename, "r");
		std::cout << "Mesh file name: " << filename << std::endl;

		fseek(file, 0, SEEK_END);
		long fileSize = ftell(file);
		std::cout << "fileSize:" << fileSize << std::endl;

		try
		{
			Vertex_Buffer = (float*)malloc(ftell(file));
			Normal_Buffer = (float*)malloc(ftell(file));
		}
		catch (char*)
		{
			return;
		}
		if (Vertex_Buffer == NULL) return;
		if (Normal_Buffer == NULL) return;
		fseek(file, 0, SEEK_SET);

		Faces_Triangles = new float[fileSize];
		Normals = new float[fileSize];
		indices = new unsigned int[fileSize];

		if (file)
		{
			int i = 0;
			int temp = 0;
			int quads_index = 0;
			int triangle_index = 0;
			int normal_index = 0;
			char buffer[1000];

			fgets(buffer, 300, file);			// ply

			// READ HEADER
			// -----------------

			// Find number of vertexes
			while (strncmp("element vertex", buffer, strlen("element vertex")) != 0)
			{
				fgets(buffer, 300, file);			// format
			}
			strcpy(buffer, buffer + strlen("element vertex"));
			sscanf(buffer, "%i", &TotalConnectedPoints);
			std::cout << "TotalConnectedPoints:" << TotalConnectedPoints << std::endl;


			// Find number of vertexes
			fseek(file, 0, SEEK_SET);
			while (strncmp("element face", buffer, strlen("element face")) != 0)
			{
				fgets(buffer, 300, file);			// format
			}
			strcpy(buffer, buffer + strlen("element face"));
			int TotalFaces;
			sscanf(buffer, "%i", &TotalFaces);
			std::cout << "TotalFaces:" << TotalFaces << std::endl;


			// go to end_header
			while (strncmp("end_header", buffer, strlen("end_header")) != 0)
			{
				fgets(buffer, 300, file);			// format
			}

			//----------------------

			float xmin(FLT_MAX), xmax(-FLT_MAX), ymin(FLT_MAX), ymax(-FLT_MAX), zmin(FLT_MAX), zmax(-FLT_MAX);
			// read verteces
			i = 0;
			for (int iterator = 0; iterator < TotalConnectedPoints; iterator++)
			{
				fgets(buffer, 300, file);

#if 1
				sscanf(buffer, "%f %f %f %f %f %f", &Vertex_Buffer[i], &Vertex_Buffer[i + 1], &Vertex_Buffer[i + 2]
					, &Normal_Buffer[i], &Normal_Buffer[i + 1], &Normal_Buffer[i + 2]);
#else
				sscanf(buffer, "%f %f %f", &Vertex_Buffer[i], &Vertex_Buffer[i + 1], &Vertex_Buffer[i + 2]);
#endif
				//if(iterator < 5)
				//	cout<<"vertex : "<<Vertex_Buffer[i]<<","<<Vertex_Buffer[i+1]<<","<<Vertex_Buffer[i+2]<<endl;

				//////some statistics by Xin Tong
				if (Vertex_Buffer[i] < xmin)
					xmin = Vertex_Buffer[i];
				if (Vertex_Buffer[i] > xmax)
					xmax = Vertex_Buffer[i];

				if (Vertex_Buffer[i + 1] < ymin)
					ymin = Vertex_Buffer[i + 1];
				if (Vertex_Buffer[i + 1] > ymax)
					ymax = Vertex_Buffer[i + 1];

				if (Vertex_Buffer[i + 2] < zmin)
					zmin = Vertex_Buffer[i + 2];
				if (Vertex_Buffer[i + 2] > zmax)
					zmax = Vertex_Buffer[i + 2];

				i += 3;
			}
			//std::cout << "ranges: " << xmin << "," << xmax << "," << ymin << "," << ymax << "," << zmin << "," << zmax << std::endl;
			//col_dims[0] = xmax;
			//col_dims[1] = ymax;
			//col_dims[2] = zmax;

			// read faces
			i = 0;
			for (int iterator = 0; iterator < TotalFaces; iterator++)
			{
				fgets(buffer, 300, file);

				if (buffer[0] == '3')
				{

					int vertex1 = 0, vertex2 = 0, vertex3 = 0;
					//sscanf(buffer,"%i%i%i\n", vertex1,vertex2,vertex3 );
					buffer[0] = ' ';
					sscanf(buffer, "%i%i%i", &vertex1, &vertex2, &vertex3);
					//if (iterator<5)
					//	std::cout << "vertex : " << vertex1 << "," << vertex2 << "," << vertex3 << std::endl;
					/*vertex1 -= 1;
					vertex2 -= 1;
					vertex3 -= 1;
					*/
					//  vertex == punt van vertex lijst
					// vertex_buffer -> xyz xyz xyz xyz
					//printf("%f %f %f \n", Vertex_Buffer[3*vertex1], Vertex_Buffer[3*vertex1+1], Vertex_Buffer[3*vertex1+2]);

					Faces_Triangles[triangle_index] = Vertex_Buffer[3 * vertex1];
					Faces_Triangles[triangle_index + 1] = Vertex_Buffer[3 * vertex1 + 1];
					Faces_Triangles[triangle_index + 2] = Vertex_Buffer[3 * vertex1 + 2];
					Faces_Triangles[triangle_index + 3] = Vertex_Buffer[3 * vertex2];
					Faces_Triangles[triangle_index + 4] = Vertex_Buffer[3 * vertex2 + 1];
					Faces_Triangles[triangle_index + 5] = Vertex_Buffer[3 * vertex2 + 2];
					Faces_Triangles[triangle_index + 6] = Vertex_Buffer[3 * vertex3];
					Faces_Triangles[triangle_index + 7] = Vertex_Buffer[3 * vertex3 + 1];
					Faces_Triangles[triangle_index + 8] = Vertex_Buffer[3 * vertex3 + 2];

#if 1
					Normals[triangle_index] = Normal_Buffer[3 * vertex1];
					Normals[triangle_index + 1] = Normal_Buffer[3 * vertex1 + 1];
					Normals[triangle_index + 2] = Normal_Buffer[3 * vertex1 + 2];
					Normals[triangle_index + 3] = Normal_Buffer[3 * vertex2];
					Normals[triangle_index + 4] = Normal_Buffer[3 * vertex2 + 1];
					Normals[triangle_index + 5] = Normal_Buffer[3 * vertex2 + 2];
					Normals[triangle_index + 6] = Normal_Buffer[3 * vertex3];
					Normals[triangle_index + 7] = Normal_Buffer[3 * vertex3 + 1];
					Normals[triangle_index + 8] = Normal_Buffer[3 * vertex3 + 2];
#else

					float coord1[3] = { Faces_Triangles[triangle_index], Faces_Triangles[triangle_index + 1], Faces_Triangles[triangle_index + 2] };
					float coord2[3] = { Faces_Triangles[triangle_index + 3], Faces_Triangles[triangle_index + 4], Faces_Triangles[triangle_index + 5] };
					float coord3[3] = { Faces_Triangles[triangle_index + 6], Faces_Triangles[triangle_index + 7], Faces_Triangles[triangle_index + 8] };
					float *norm = calculateNormal(coord1, coord2, coord3);

					Normals[normal_index] = norm[0];
					Normals[normal_index + 1] = norm[1];
					Normals[normal_index + 2] = norm[2];
					Normals[normal_index + 3] = norm[0];
					Normals[normal_index + 4] = norm[1];
					Normals[normal_index + 5] = norm[2];
					Normals[normal_index + 6] = norm[0];
					Normals[normal_index + 7] = norm[1];
					Normals[normal_index + 8] = norm[2];

					normal_index += 9;
#endif
					triangle_index += 9;
					TotalConnectedTriangles += 1;
					i += 3;
				}
				else{
					std::cout << "NOT suppourt non-triangle now..." << std::endl;
					exit(0);
				}
			}

			std::cout << "Mesh file loading is done..." << std::endl;


			free(Vertex_Buffer);
			free(Normal_Buffer);

			for (int i = 0; i < TotalConnectedTriangles * 3; i++) {
				indices[i] = i;
			}
			fclose(file);

			computeCenter();
		}

		else { printf("File can't be opened\n"); }

		//polyMesh->vertexcount = TotalConnectedPoints;
		polyMesh->vertexcount = TotalConnectedTriangles*3; //actually not very efficent?
		polyMesh->facecount = TotalConnectedTriangles;
		polyMesh->vertexCoords = Faces_Triangles;
		polyMesh->indices = indices;
		polyMesh->vertexNorms = Normals;
	}
	else {
		printf("File does not have a .PLY extension. ");
	}
	return;
}


void PlyMeshReader::computeCenter()
{
	float x = 0, y = 0, z = 0;
	for (int i = 0; i < TotalConnectedTriangles; i++){
		x += Faces_Triangles[9 * i];
		x += Faces_Triangles[9 * i + 3];
		x += Faces_Triangles[9 * i + 6];
		y += Faces_Triangles[9 * i + 1];
		y += Faces_Triangles[9 * i + 4];
		y += Faces_Triangles[9 * i + 7];
		z += Faces_Triangles[9 * i + 2];
		z += Faces_Triangles[9 * i + 5];
		z += Faces_Triangles[9 * i + 8];
	}
	x = x / TotalConnectedTriangles/3;
	y = y / TotalConnectedTriangles/3;
	z = z / TotalConnectedTriangles/3;
	center = make_float3(x,y,z);
}
