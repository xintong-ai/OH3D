#ifndef TRACE_H
#define TRACE_H

#include <vector_types.h>
#include <vector_functions.h>
#include <vector>

class Trace
{
public:
	Trace() {};
	~Trace() {
		pos.clear();
	};

	std::vector<float3> pos;
	void save(const char* filename);
};


class UncertainTrace : public Trace
{
public:

	UncertainTrace() : Trace(){};

	~UncertainTrace() {
		stds.clear();
	};

	std::vector<float3> stds;

	void save(const char* filename);
};


#endif