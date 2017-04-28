#ifndef PROCESSOR_H
#define PROCESSOR_H

class Processor
{
public:
	Processor(){};
	~Processor(){};
	virtual bool process(float modelview[16], float projection[16], int winW, int winH){ return false; };

	void resize(int width, int height){ ; }//not implement in each proccessor yet. may need to do this in the future

	bool isActive = true;
};
#endif