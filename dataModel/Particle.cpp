#include <Particle.h>
#include <iostream>
#include <algorithm>    // std::random_shuffle
#include <ctime>        // std::time

#include <helper_cuda.h>
#include <helper_math.h>


Particle::Particle(std::vector<float4> _pos, std::vector<float> _val)
{
	pos = _pos;
	posOrig = _pos;
	val = _val;
	numParticles = pos.size();

	posMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	posMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	float v = 0;
	for (int i = 0; i < pos.size(); i++) {
		v = pos[i].x;
		if (v > posMax.x)
			posMax.x = v;
		if (v < posMin.x)
			posMin.x = v;

		v = pos[i].y;
		if (v > posMax.y)
			posMax.y = v;
		if (v < posMin.y)
			posMin.y = v;

		v = pos[i].z;
		if (v > posMax.z)
			posMax.z = v;
		if (v < posMin.z)
			posMin.z = v;
	}

	valMax = -FLT_MAX;
	valMin = FLT_MAX;
	for (int i = 0; i < val.size(); i++) {
		v = val[i];
		if (v > valMax)
			valMax = v;
		if (v < valMin)
			valMin = v;
	}
}

void Particle::setFeature(std::vector<char> _f)
{
	feature = _f;
	hasFeature = true;

	featureMax = CHAR_MIN;
	featureMin = CHAR_MAX;
	char c;
	for (int i = 0; i < feature.size(); i++) {
		c = feature[i];
		if (c > featureMax)
			featureMax = c;
		if (c < featureMin)
			featureMin = c;
	}
}


void Particle::featureReshuffle()
{
	//since the feature is just tags, we may want to reshuffle the tags due to any reason
	//note! this function do not process value 0. treat it as no feature exists

	std::vector<int> myvector;
	for (int i = 1; i<=featureMax; ++i) myvector.push_back(i); // 1 2 3 4 5 6 7 8 9
	unsigned seed = unsigned(std::time(0));
	//std::srand(seed); //1475426366, 
	std::srand(1475426366); //1475426366, 
	std::cout << "reshuffle seed: " << seed << std::endl;
	std::random_shuffle(myvector.begin(), myvector.end()); 

	for (int i = 0; i < feature.size(); i++){
		char c = feature[i];
		if (c >0)
		{
			feature[i] = myvector[(int)feature[i] - 1];
		}
	}


	////the following is used to generate the image in case study 6.1, to make the colors of the 3 halos differ more
	//for (int i = 0; i < feature.size(); i++)
	//{
	//	char c = feature[i];
	//	if (c ==12)
	//	{
	//		feature[i]  = 30;
	//	}
	//	else if (c == 30)
	//	{
	//		feature[i] = 12;
	//	}
	//	else if (c == 8)
	//	{
	//		feature[i] = 40;
	//	}
	//	else if (c == 40)
	//	{
	//		feature[i] = 8;
	//	}
	//	else if (c == 13)
	//	{
	//		feature[i] = 20;
	//	}
	//	else if (c == 20)
	//	{
	//		feature[i] = 13;
	//	}
	//}
}
