//copied from the website: http://www.andrewnoske.com/wiki/Code_-_heatmaps_and_color_gradients
#ifndef COLOR_GRADIENT_H
#define COLOR_GRADIENT_H
#include <vector>
#include <algorithm>

enum COLOR_MAP{
	RAINBOW,
	RAINBOW_COSMOLOGY,
	SIMPLE_BLUE_RED,
	PU_OR,
	RDYIGN,
	BrBG
};

class ColorGradient
{
private:
	struct ColorPoint  // Internal class used to store colors at different points in the gradient.
	{
		float r, g, b;      // Red, green and blue values of our color.
		float val;        // Position of our color along the gradient (between 0 and 1).
		ColorPoint(float red, float green, float blue, float value)
			: r(red), g(green), b(blue), val(value) {}
	};
	std::vector<ColorPoint> color;      // An array of color points in ascending value.
	bool isReversed = false;

public:
	//-- Default constructor:
	ColorGradient()  { createDefaultHeatMapGradient(); }

	ColorGradient(COLOR_MAP cm, bool _b = false){
		isReversed = _b;
		if (cm == COLOR_MAP::RAINBOW){
			createDefaultRainbowMapGradient();
		}
		else if (cm == COLOR_MAP::SIMPLE_BLUE_RED){ 
			createDefaultHeatMapGradient(); 
		}
		else if (cm == RAINBOW_COSMOLOGY){
			createDefaultRainbowCosmologyMapGradient();
		}
		else if (cm == PU_OR){
			createDefaultPuOrMapGradient();
		}
		else if (cm == RDYIGN){
			createDefaultRDYIGNMapGradient();
		}
		else if (cm == BrBG){
			createDefaultBrBGMapGradient();
		}
		else{
			exit(0);
		}
	}
	//-- Inserts a new color point into its correct position:
	void addColorPoint(float red, float green, float blue, float value)
	{
		for (int i = 0; i<color.size(); i++)  {
			if (value < color[i].val) {
				color.insert(color.begin() + i, ColorPoint(red, green, blue, value));
				return;
			}
		}
		color.push_back(ColorPoint(red, green, blue, value));
	}

	//-- Inserts a new color point into its correct position:
	void clearGradient() { color.clear(); }

	//-- Places a 5 color heapmap gradient into the "color" vector:
	void createDefaultRainbowMapGradient()
	{
		color.clear();
		color.push_back(ColorPoint(0, 0, 1, 0.0f));      // Blue.
		color.push_back(ColorPoint(0, 1, 1, 0.25f));     // Cyan.
		color.push_back(ColorPoint(0, 1, 0, 0.5f));      // Green.
		color.push_back(ColorPoint(1, 1, 0, 0.75f));     // Yellow.
		color.push_back(ColorPoint(1, 0, 0, 1.0f));      // Red.
	}

	void createDefaultRainbowCosmologyMapGradient()
	{
		color.clear();
		color.push_back(ColorPoint(0.3, 0.3, 0.3, 0.0f));    // grey for 0 value particles
		color.push_back(ColorPoint(0, 0, 1, 0.00001f));      // Blue.
		color.push_back(ColorPoint(0, 1, 1, 0.25f));     // Cyan.
		color.push_back(ColorPoint(0, 1, 0, 0.5f));      // Green.
		color.push_back(ColorPoint(1, 1, 0, 0.75f));     // Yellow.
		color.push_back(ColorPoint(1, 0, 0, 1.0f));      // Red.
	}

	void createDefaultHeatMapGradient()
	{
		color.clear();
		color.push_back(ColorPoint(0, 0, 1, 0.0f));      // Blue.
		color.push_back(ColorPoint(1, 1, 1, 0.5f));
		color.push_back(ColorPoint(1, 0, 0, 1.0f));      // Red.
	}

	void createDefaultPuOrMapGradient()
	{
		color.clear();
		color.push_back(ColorPoint(127.0 / 255.0, 59.0 / 255.0, 8 / 255.0, 0.0f));
		color.push_back(ColorPoint(179.0 / 255.0, 88.0 / 255.0, 6 / 255.0, 0.1f));
		color.push_back(ColorPoint(224.0 / 255.0, 130.0 / 255.0, 20 / 255.0, 0.2f));
		color.push_back(ColorPoint(253.0 / 255.0, 184.0 / 255.0, 99 / 255.0, 0.3f));
		color.push_back(ColorPoint(254.0 / 255.0, 224.0 / 255.0, 182 / 255.0, 0.4f));
		color.push_back(ColorPoint(247.0 / 255.0, 247.0 / 255.0, 247.0 / 255.0, 0.5f));
		color.push_back(ColorPoint(216 / 255.0, 238 / 255.0, 235 / 255.0, 0.6f));
		color.push_back(ColorPoint(178 / 255.0, 171 / 255.0, 235 / 210, 0.7f));
		color.push_back(ColorPoint(128 / 255.0, 115 / 255.0, 172 / 255.0, 0.8f));
		color.push_back(ColorPoint(84 / 255.0, 39 / 255.0, 136 / 255.0, 0.9f));
		color.push_back(ColorPoint(45 / 255.0, 0 / 255.0, 75 / 255.0, 1.0f));
	}
	
	void createDefaultRDYIGNMapGradient()
	{
		color.clear();
		color.push_back(ColorPoint(0.647058824, 0, 0.149019608, 0));
		color.push_back(ColorPoint(0.843137255, 0.188235294, 0.152941176, 0.1));
		color.push_back(ColorPoint(0.956862745, 0.42745098, 0.262745098, 0.2));
		color.push_back(ColorPoint(0.992156863, 0.682352941, 0.380392157, 0.3));
		color.push_back(ColorPoint(0.996078431, 0.878431373, 0.545098039, 0.4));
		color.push_back(ColorPoint(1, 1, 0.749019608, 0.5));
		color.push_back(ColorPoint(0.850980392, 0.937254902, 0.545098039, 0.6));
		color.push_back(ColorPoint(0.650980392, 0.850980392, 0.415686275, 0.7));
		color.push_back(ColorPoint(0.4, 0.741176471, 0.388235294, 0.8));
		color.push_back(ColorPoint(0.101960784, 0.596078431, 0.31372549, 0.9));
		color.push_back(ColorPoint(0, 0.407843137, 0.215686275, 1));
	}

	void createDefaultBrBGMapGradient()
	{
		color.clear();
		color.push_back(ColorPoint(0.329411765, 0.188235294, 0.019607843, 0));
		color.push_back(ColorPoint(0.549019608, 0.317647059, 0.039215686, 0.1));
		color.push_back(ColorPoint(0.749019608, 0.505882353, 0.176470588, 0.2));
		color.push_back(ColorPoint(0.874509804, 0.760784314, 0.490196078, 0.3));
		color.push_back(ColorPoint(0.964705882, 0.909803922, 0.764705882, 0.4));
		color.push_back(ColorPoint(0.960784314, 0.960784314, 0.960784314, 0.5));
		color.push_back(ColorPoint(0.780392157, 0.917647059, 0.898039216, 0.6));
		color.push_back(ColorPoint(0.501960784, 0.803921569, 0.756862745, 0.7));
		color.push_back(ColorPoint(0.207843137, 0.592156863, 0.560784314, 0.8));
		color.push_back(ColorPoint(0.003921569, 0.4, 0.368627451, 0.9));
		color.push_back(ColorPoint(0, 0.235294118, 0.188235294, 1));
	}
	
	//-- Inputs a (value) between 0 and 1 and outputs the (red), (green) and (blue)
	//-- values representing that position in the gradient.
	void getColorAtValue(float value, float &red, float &green, float &blue)
	{
		if (color.size() == 0)
			return;
		if (isReversed)
			value = 1 - value;

		for (int i = 0; i<color.size(); i++)
		{
			ColorPoint &currC = color[i];
			if (value < currC.val)
			{
				ColorPoint &prevC = color[std::max(0, i - 1)];
				float valueDiff = (prevC.val - currC.val);
				float fractBetween = (valueDiff == 0) ? 0 : (value - currC.val) / valueDiff;
				red = (prevC.r - currC.r)*fractBetween + currC.r;
				green = (prevC.g - currC.g)*fractBetween + currC.g;
				blue = (prevC.b - currC.b)*fractBetween + currC.b;
				return;
			}
		}
		red = color.back().r;
		green = color.back().g;
		blue = color.back().b;
		return;
	}
};

#endif