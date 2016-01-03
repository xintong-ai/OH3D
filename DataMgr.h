#ifndef DATA_MGR_H
#define DATA_MGR_H

#include "cstdlib"
#include <assert.h>
#include <map>

class DataMgr
{
public:
	DataMgr();

	~DataMgr();

	std::string GetConfig(const char* name){ return config[name]; }

	//std::map<std::string, std::string> GetMeshFiles();

private:
	void LoadConfig(const char* filename);
	std::map<std::string, std::string> config;
};


#endif //DATA_MGR_H
