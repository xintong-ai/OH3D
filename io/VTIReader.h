#ifndef VTI_READER_H
#define VTI_READER_H

#include <fstream>
#include <iostream>
#include <memory>
#define _USE_MATH_DEFINES
#include "math.h"
#include "vector_types.h"
class Volume;

class VTIReader
{
public:
	VTIReader(const char* fname, std::shared_ptr<Volume> v);

private:
};



#endif