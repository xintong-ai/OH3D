#include "VTIReader.h"
#include "Volume.h"

#include <algorithm>

#include <vtkImageData.h>
#include <vtkXMLImageDataReader.h>
#include <vtkSmartPointer.h>
#include <vtkFloatArray.h>

#include <vtkPointData.h>

using namespace std;

VTIReader::VTIReader(const char* fname, std::shared_ptr<Volume> v)
{
	vtkSmartPointer<vtkXMLImageDataReader> reader =
		vtkSmartPointer<vtkXMLImageDataReader>::New();
	reader->SetFileName(fname);
	reader->Update();

	vtkSmartPointer<vtkImageData> img = reader->GetOutput();
	int3 dataSizes;
	img->GetDimensions(&(dataSizes.x));

	v->~Volume();

	v->size = dataSizes;
	v->spacing = make_float3(1, 1, 1);//currently cannot handle other spacing
	v->dataOrigin = make_float3(0, 0, 0);//currently cannot handle other origin

	v->values = new float[dataSizes.x*dataSizes.y*dataSizes.z];


	vtkSmartPointer<vtkFloatArray> array = vtkFloatArray::SafeDownCast(img->GetPointData()->GetArray("XW 2=    CO2     ")); // !!!!!!!!!!!!!!!! currently only for this case !!!!!!!!!!!!!!!


	float minVal = 9999999, maxVal = -999999;

	for (int k = 0; k < dataSizes.z; k++)
	{
		for (int j = 0; j < dataSizes.y; j++)
		{
			for (int i = 0; i < dataSizes.x; i++)
			{
				int ind = k*dataSizes.y * dataSizes.x + j*dataSizes.x + i;
				float p = array->GetValue(ind);
				minVal = min(minVal, p);
				maxVal = max(maxVal, p);
			}
		}
	}

	std::cout << "min max:" << minVal << " " << maxVal << std::endl;

	for (int k = 0; k < dataSizes.z; k++)
	{
		for (int j = 0; j < dataSizes.y; j++)
		{
			for (int i = 0; i < dataSizes.x; i++)
			{
				int ind = k*dataSizes.y * dataSizes.x + j*dataSizes.x + i;
				float p = array->GetValue(ind);
				v->values[ind] = (p - minVal) / (maxVal - minVal);
			}
		}
	}
}
