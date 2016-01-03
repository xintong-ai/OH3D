#ifndef READER_H
#define READER_H

#include <string>
class Reader
{
public:
	Reader(const char* filename){ datafilename.assign(filename); }
protected:
	virtual void Load() = 0;
	std::string datafilename;
};

#endif //READER_H