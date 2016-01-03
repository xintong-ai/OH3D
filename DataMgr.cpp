#include "DataMgr.h"
#include "iostream"
#include <fstream>
#include <sstream>

#ifdef __linux__
#include "stdio.h"
#endif


DataMgr::~DataMgr()
{

}

DataMgr::DataMgr()
{
    //SetStride(1, 1, 1);
	LoadConfig("config.txt");
}

//std::map<std::string, std::string> DataMgr::GetMeshFiles()
//{
//	std::map<std::string, std::string> ret;
//	for (auto m : config) {
//		std::string a = m.first.substr(0, 5);
//		if (a == "mesh_") {
//			ret[m.first] = m.second;
//		}
//	}
//	return ret;
//}


void DataMgr::LoadConfig(const char* filename)
{

	std::ifstream is_file(filename);

	std::string line;
	while (std::getline(is_file, line))
	{
		std::istringstream is_line(line);
		std::string key;
		if (std::getline(is_line, key, '='))
		{
			std::string value;
			if (std::getline(is_line, value))
				config[key] = value;
		}
	}
}
