

#ifndef GLYPHINTERACTOR_H
#define GLYPHINTERACTOR_H
#include "Interactor.h"

class Particle;
class GlyphInteractor :public Interactor
{
private:


protected:


public:
	GlyphInteractor(){};
	~GlyphInteractor(){};

	//void SetLenses(std::vector<Lens*> *_lenses){ lenses = _lenses; }

	void mousePress(int x, int y, int modifier, int mouseKey = 0) override;
	
};
#endif