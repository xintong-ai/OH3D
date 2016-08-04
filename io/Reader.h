#ifndef READER_H
#define READER_H

#include <vector_types.h>

#include <string>
class Reader
{
public:
    Reader(const char* filename){
        datafilename.assign(filename);
    }
	virtual void GetPosRange(float3& posMin, float3& posMax) = 0;
	//virtual void GetValRange(float& vMin, float& vMax) = 0;
protected:
	virtual void Load() = 0;
	std::string datafilename;
};

#endif //READER_H
