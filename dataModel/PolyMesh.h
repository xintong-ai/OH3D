#ifndef POLYMESH_H
#define POLYMESH_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <math.h>
#include <memory>
//using namespace std;

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "ply.h"


typedef struct Vertex {
	float x, y, z;
	float u, v, w;
} Vertex;

typedef struct Face {
	unsigned int count;
	unsigned int *vertices;
	float u, v, w;
} Face;

char* string_list[] = {
	"x", "y", "z", "u", "v", "w", "vertex_indices"
};

float cx, cy, cz;
float x_min, x_max, y_min, y_max, z_min, z_max;


class PolyMesh
{
public:

	Vertex** vertices = 0;
	Face** faces = 0;
	unsigned int vertexcount;
	unsigned int facecount;
	int vertexnormals = 0;
	int facenormals = 0;
	
	
	void read(const char* fname){
		FILE * pFile = fopen(fname, "r");
		if (pFile == NULL) {
			perror("Error opening file");
			exit(EXIT_SUCCESS);
		}
		
		PlyFile* input;

		// get the ply structure and open the file
		input = read_ply(pFile);

		// read in the data
		store_ply(input,
			&vertices, &faces,
			&vertexcount, &facecount,
			&vertexnormals, &facenormals);

		// close the file
		close_ply(input);


		find_center(cx, cy, cz, x_min, x_max,
			y_min, y_max, z_min, z_max);
		printf("geometry center = %f %f %f \n", cx, cy, cz);
		printf("geometry bound = x: %f %f y: %f %f z: %f %f\n",
			x_min, x_max, y_min, y_max, z_min, z_max);
	}




private:
	void store_ply(PlyFile* input, Vertex ***vertices, Face ***faces,
		unsigned int* vertexcount, unsigned int* facecount,
		int* vertexnormals, int* facenormals) {
		int i, j;

		// go through the element types
		for (i = 0; i < input->num_elem_types; i = i + 1) {
			int count;

			// setup the element for reading and get the element count
			char* element = setup_element_read_ply(input, i, &count);

			// get vertices
			if (strcmp("vertex", element) == 0) {
				*vertices = (Vertex**)malloc(sizeof(Vertex)* count);
				*vertexcount = count;

				// run through the properties and store them
				for (j = 0; j < input->elems[i]->nprops; j = j + 1) {
					PlyProperty* property = input->elems[i]->props[j];
					PlyProperty setup;

					if (strcmp("x", property->name) == 0 &&
						property->is_list == PLY_SCALAR) {

						setup.name = string_list[0];
						setup.internal_type = Float32;
						setup.offset = offsetof(Vertex, x);
						setup.count_internal = 0;
						setup.count_offset = 0;

						setup_property_ply(input, &setup);
					}
					else if (strcmp("y", property->name) == 0 &&
						property->is_list == PLY_SCALAR) {

						setup.name = string_list[1];
						setup.internal_type = Float32;
						setup.offset = offsetof(Vertex, y);
						setup.count_internal = 0;
						setup.count_offset = 0;

						setup_property_ply(input, &setup);
					}
					else if (strcmp("z", property->name) == 0 &&
						property->is_list == PLY_SCALAR) {

						setup.name = string_list[2];
						setup.internal_type = Float32;
						setup.offset = offsetof(Vertex, z);
						setup.count_internal = 0;
						setup.count_offset = 0;

						setup_property_ply(input, &setup);
					}
					else if (strcmp("u", property->name) == 0 &&
						property->is_list == PLY_SCALAR) {

						setup.name = string_list[3];
						setup.internal_type = Float32;
						setup.offset = offsetof(Vertex, u);
						setup.count_internal = 0;
						setup.count_offset = 0;

						setup_property_ply(input, &setup);
						*vertexnormals = 1;
					}
					else if (strcmp("v", property->name) == 0 &&
						property->is_list == PLY_SCALAR) {

						setup.name = string_list[4];
						setup.internal_type = Float32;
						setup.offset = offsetof(Vertex, v);
						setup.count_internal = 0;
						setup.count_offset = 0;

						setup_property_ply(input, &setup);
						*vertexnormals = 1;
					}
					else if (strcmp("w", property->name) == 0 &&
						property->is_list == PLY_SCALAR) {

						setup.name = string_list[5];
						setup.internal_type = Float32;
						setup.offset = offsetof(Vertex, w);
						setup.count_internal = 0;
						setup.count_offset = 0;

						setup_property_ply(input, &setup);
						*vertexnormals = 1;
					}
					// dunno what it is
					else {
						fprintf(stderr, "unknown property type found in %s: %s\n",
							element, property->name);
					}
				}

				// do this if you want to grab the other data
				// list_pointer = get_other_properties_ply
				//                (input, offsetof(Vertex, struct_pointer));

				// copy the data
				for (j = 0; j < count; j = j + 1) {
					(*vertices)[j] = (Vertex*)malloc(sizeof(Vertex));

					get_element_ply(input, (void*)((*vertices)[j]));
				}
			}
			// get faces
			else if (strcmp("face", element) == 0) {
				*faces = (Face**)malloc(sizeof(Face)* count);
				*facecount = count;

				// run through the properties and store them
				for (j = 0; j < input->elems[i]->nprops; j = j + 1) {
					PlyProperty* property = input->elems[i]->props[j];
					PlyProperty setup;

					if (strcmp("vertex_indices", property->name) == 0 &&
						property->is_list == PLY_LIST) {

						setup.name = string_list[6];
						setup.internal_type = Uint32;
						setup.offset = offsetof(Face, vertices);
						setup.count_internal = Uint32;
						setup.count_offset = offsetof(Face, count);

						setup_property_ply(input, &setup);
					}
					else if (strcmp("u", property->name) == 0 &&
						property->is_list == PLY_SCALAR) {

						setup.name = string_list[3];
						setup.internal_type = Float32;
						setup.offset = offsetof(Face, u);
						setup.count_internal = 0;
						setup.count_offset = 0;

						setup_property_ply(input, &setup);
						*facenormals = 1;
					}
					else if (strcmp("v", property->name) == 0 &&
						property->is_list == PLY_SCALAR) {

						setup.name = string_list[4];
						setup.internal_type = Float32;
						setup.offset = offsetof(Face, v);
						setup.count_internal = 0;
						setup.count_offset = 0;

						setup_property_ply(input, &setup);
						*facenormals = 1;
					}
					else if (strcmp("w", property->name) == 0 &&
						property->is_list == PLY_SCALAR) {

						setup.name = string_list[5];
						setup.internal_type = Float32;
						setup.offset = offsetof(Face, w);
						setup.count_internal = 0;
						setup.count_offset = 0;

						setup_property_ply(input, &setup);
						*facenormals = 1;
					}
					// dunno what it is
					else {
						fprintf(stderr, "unknown property type found in %s: %s\n",
							element, property->name);
					}
				}

				// do this if you want to grab the other data
				// list_pointer = get_other_properties_ply
				//                (input, offsetof(Face, struct_pointer));

				// copy the data
				for (j = 0; j < count; j = j + 1) {
					(*faces)[j] = (Face*)malloc(sizeof(Face));

					get_element_ply(input, (void*)((*faces)[j]));
				}
			}
			// who knows?
			else {
				fprintf(stderr, "unknown element type found: %s\n", element);
			}
		}

		// if you want to grab the other data do this
		// get_other_element_ply(input);
	}


	void find_center(float& cx, float& cy, float& cz,
		float& minx, float& maxx, float&miny,
		float &maxy, float &minz, float & maxz)
	{
		float x, y, z;
		float min_x = 9999, max_x = -9999, min_y = 9999, max_y = -9999;
		float min_z = 9999, max_z = -9999;

		x = y = z = 0;
		for (int i = 0; i < vertexcount; i++) {
			x += vertices[i]->x;
			y += vertices[i]->y;
			z += vertices[i]->z;
			if (min_x >vertices[i]->x) min_x = vertices[i]->x;
			if (max_x <vertices[i]->x) max_x = vertices[i]->x;
			if (min_y >vertices[i]->y) min_y = vertices[i]->y;
			if (max_y <vertices[i]->y) max_y = vertices[i]->y;
			if (min_z >vertices[i]->z) min_z = vertices[i]->z;
			if (max_z <vertices[i]->z) max_z = vertices[i]->z;
		}
		cx = x / (float)vertexcount;
		cy = y / (float)vertexcount;
		cz = z / (float)vertexcount;
		minx = min_x; maxx = max_x;
		miny = min_y; maxy = max_y;
		minz = min_z; maxz = max_z;
	}
};
#endif