#include <vector_types.h>
#include <vector>

#include <itkBinaryMorphologicalOpeningImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>

#include "itkBinaryErodeImageFilter.h"
#include "itkGrayscaleMorphologicalOpeningImageFilter.h"
#include "itkConnectedComponentImageFilter.h"

#include "itkImage.h"


#include "MSTAdjList.h"


#define WRITEIMAGETODISK



//for itk
typedef unsigned short PixelType;
typedef itk::Image< PixelType, 3 > ImageType;
typedef itk::ImportImageFilter< PixelType, 3 > ImportFilterType;


void skelComputing(PixelType * localBuffer, int3 dims, float3 spacing, float* skelVolValues, ImageType::Pointer & retImgPointer, int & maxComponentMark)
{

	///////////////import to itk image
	const bool importImageFilterWillOwnTheBuffer = false; //probably can change to true for faster speed?
	typedef itk::BinaryThinningImageFilter3D< ImageType, ImageType > ThinningFilterType;
	ImportFilterType::Pointer importFilter = ImportFilterType::New();

	ImageType::SizeType imsize;
	imsize[0] = dims.x;
	imsize[1] = dims.y;
	imsize[2] = dims.z;
	ImportFilterType::IndexType start;
	start.Fill(0);
	ImportFilterType::RegionType region;
	region.SetIndex(start);
	region.SetSize(imsize);
	importFilter->SetRegion(region);
	const itk::SpacePrecisionType origin[3] = { imsize[0], imsize[1], imsize[2] };
	importFilter->SetOrigin(origin);
	const itk::SpacePrecisionType _spacing[3] = { spacing.x, spacing.y, spacing.z };
	importFilter->SetSpacing(_spacing);
	importFilter->SetImportPointer(localBuffer, dims.x * dims.y * dims.z, importImageFilterWillOwnTheBuffer);
	importFilter->Update();


#ifdef WRITEIMAGETODISK
	typedef itk::ImageFileWriter< ImageType > WriterType;
	WriterType::Pointer writer = WriterType::New();
	writer->SetInput(importFilter->GetOutput());
	writer->SetFileName("channelVol.hdr");
	writer->Update();
#endif

	//////////////////openining
	typedef itk::BinaryBallStructuringElement<ImageType::PixelType, ImageType::ImageDimension>
		StructuringElementType;
	StructuringElementType structuringElement;
	int radius = 4;
	structuringElement.SetRadius(radius);
	structuringElement.CreateStructuringElement();
	typedef itk::GrayscaleMorphologicalOpeningImageFilter< ImageType, ImageType, StructuringElementType > OpeningFilterType;
	OpeningFilterType::Pointer openingFilter = OpeningFilterType::New();
	openingFilter->SetInput(importFilter->GetOutput());
	openingFilter->SetKernel(structuringElement);
	openingFilter->Update();

#ifdef WRITEIMAGETODISK
	writer->SetInput(openingFilter->GetOutput());
	writer->SetFileName("opened.hdr");
	writer->Update();
#endif

	//////////////compute connected components
	typedef itk::ConnectedComponentImageFilter <ImageType, ImageType >
		ConnectedComponentImageFilterType;
	ConnectedComponentImageFilterType::Pointer connected =
		ConnectedComponentImageFilterType::New();
	connected->SetInput(openingFilter->GetOutput());
	connected->Update();
#ifdef WRITEIMAGETODISK
	writer->SetInput(connected->GetOutput());
	writer->SetFileName("cp.hdr");
	writer->Update();
#endif


	////////////////filter out small and boundary componenets
	ImageType::Pointer connectedImg = connected->GetOutput();

	int numObj = connected->GetObjectCount();
	std::cout << "Number of objects: " << connected->GetObjectCount() << std::endl;

	std::vector<bool> atOutside(numObj + 1, 0);
	for (int k = 0; k < dims.z; k += dims.z - 1){
		for (int j = 0; j < dims.y; j++){
			for (int i = 0; i < dims.x; i++){
				ImageType::IndexType pixelIndex;
				pixelIndex[0] = i; // x position
				pixelIndex[1] = j; // y position
				pixelIndex[2] = k; // z position
				int v = connectedImg->GetPixel(pixelIndex);
				if (v>0){
					atOutside[v] = true;
				}
			}
		}
	}
	for (int k = 0; k < dims.z; k++){
		for (int j = 0; j < dims.y; j += dims.y - 1){
			for (int i = 0; i < dims.x; i++){
				ImageType::IndexType pixelIndex;
				pixelIndex[0] = i; // x position
				pixelIndex[1] = j; // y position
				pixelIndex[2] = k; // z position
				int v = connectedImg->GetPixel(pixelIndex);
				if (v>0){
					atOutside[v] = true;
				}
			}
		}
	}
	for (int k = 0; k < dims.z; k++){
		for (int j = 0; j < dims.y; j++){
			for (int i = 0; i < dims.x; i += dims.x - 1){
				ImageType::IndexType pixelIndex;
				pixelIndex[0] = i; // x position
				pixelIndex[1] = j; // y position
				pixelIndex[2] = k; // z position
				int v = connectedImg->GetPixel(pixelIndex);
				if (v>0){
					atOutside[v] = true;
				}
			}
		}
	}

	std::vector<int> objCount(numObj + 1, 0);
	for (int k = 0; k < dims.z; k++){
		for (int j = 0; j < dims.y; j++){
			for (int i = 0; i < dims.x; i++){
				ImageType::IndexType pixelIndex;
				pixelIndex[0] = i; // x position
				pixelIndex[1] = j; // y position
				pixelIndex[2] = k; // z position
				int v = connectedImg->GetPixel(pixelIndex);
				if (v>0){
					objCount[v]++;
				}
			}
		}
	}
	//for (int i = 0; i <= numObj; i++){
	//	std::cout << objCount[i] << std::endl;
	//}

	int thr = 1000;
	for (int k = 0; k < dims.z; k++){
		for (int j = 0; j < dims.y; j++){
			for (int i = 0; i < dims.x; i++){
				ImageType::IndexType pixelIndex;
				pixelIndex[0] = i; // x position
				pixelIndex[1] = j; // y position
				pixelIndex[2] = k; // z position
				int v = connectedImg->GetPixel(pixelIndex);
				if (atOutside[v] || objCount[v] < thr){
					connectedImg->SetPixel(pixelIndex, 0);
				}
			}
		}
	}

#ifdef WRITEIMAGETODISK
	writer->SetInput(connectedImg);
	writer->SetFileName("cpNew.hdr");
	writer->Update();
#endif


	////////////////compute skeleton
	ThinningFilterType::Pointer thinningFilter = ThinningFilterType::New();
	thinningFilter->SetInput(connectedImg); //note that connectedImg is not binary 
	thinningFilter->Update();
#ifdef WRITEIMAGETODISK
	writer->SetInput(thinningFilter->GetOutput());
	writer->SetFileName("skel.hdr");
	writer->Update();
#endif

	/////////////set function output

	ImageType::Pointer skelImg = thinningFilter->GetOutput();
	PixelType* skelRes = skelImg->GetBufferPointer();
	for (int i = 0; i < dims.x*dims.y*dims.z; i++)
	{
		skelVolValues[i] = skelRes[i];
	}



	//////////////componented skeleton image
	for (int k = 0; k < dims.z; k++){
		for (int j = 0; j < dims.y; j++){
			for (int i = 0; i < dims.x; i++){
				ImageType::IndexType pixelIndex;
				pixelIndex[0] = i; // x position
				pixelIndex[1] = j; // y position
				pixelIndex[2] = k; // z position
				if (skelImg->GetPixel(pixelIndex) < 1){
					connectedImg->SetPixel(pixelIndex, 0);
				}
			}
		}
	}
#ifdef WRITEIMAGETODISK
	writer->SetInput(connectedImg);
	writer->SetFileName("skelComponented.hdr");
	writer->Update();
#endif



	retImgPointer = connectedImg;
	maxComponentMark = numObj;
}


void findViews(ImageType::Pointer connectedImg, int maxComponentMark, int3 dims, float3 spacing, std::vector<float3> &views)
{
	////////////order skeletons by length
	std::vector<int> objCount(maxComponentMark + 1, 0);
	std::fill(objCount.begin(), objCount.end(), 0);
	for (int k = 0; k < dims.z; k++){
		for (int j = 0; j < dims.y; j++){
			for (int i = 0; i < dims.x; i++){
				ImageType::IndexType pixelIndex;
				pixelIndex[0] = i; // x position
				pixelIndex[1] = j; // y position
				pixelIndex[2] = k; // z position
				int v = connectedImg->GetPixel(pixelIndex);
				if (v>0){
					objCount[v]++;
				}
			}
		}
	}
	std::vector<size_t> idx(objCount.size());
	std::iota(idx.begin(), idx.end(), 0);
	std::sort(idx.begin(), idx.end(),
		[&objCount](size_t i1, size_t i2) {return objCount[i1] > objCount[i2]; });

	//for (int i = 0; i < objCount.size(); i++){
	//	std::cout << objCount[i] << std::endl;
	//}
	//std::cout << "sorting res:" << std::endl;
	//for (int i = 0; i < objCount.size(); i++){
	//	std::cout << "index: " << idx[i] << " with count: " << objCount[idx[i]] << std::endl;
	//}
	


	/////////do mst for the longest skel
	std::cout << "computing the MST" << std::endl;
	for (int skel = 0; skel < 1; skel++){
		int cpId = idx[skel];
		int numNodes = objCount[cpId];

		struct Graph* graph = createGraph(numNodes);
		
		std::vector<int> voxelIdAssigned(dims.x*dims.y*dims.z, 0); //!!!!NOTE!!! the stored value is the assigned id plus 1, since 0 is used as a mark
		std::vector<float3> posOfId(numNodes);
		int nextAvailableNodeId = 0;
		for (int k = 0; k < dims.z; k++){
			for (int j = 0; j < dims.y; j++){
				for (int i = 0; i < dims.x; i++){
					ImageType::IndexType pixelIndex;
					pixelIndex[0] = i; // x position
					pixelIndex[1] = j; // y position
					pixelIndex[2] = k; // z position

					if (cpId == connectedImg->GetPixel(pixelIndex)){
						int ind = k*dims.y*dims.x + j*dims.x + i;
						if (!voxelIdAssigned[ind]){
							voxelIdAssigned[ind] = nextAvailableNodeId + 1;
							posOfId[nextAvailableNodeId] = make_float3(i,j,k)*spacing;
							nextAvailableNodeId++;
						}
						for (int kk = -1; kk <= 1; kk++){
							for (int jj = -1; jj <= 1; jj++){
								for (int ii = -1; ii <= 1; ii++){
									if ( (ii || jj || kk) //if all 0, skip
										&& i + ii >= 0 && i + ii < dims.x
										&& j + jj >= 0 && j + jj < dims.y
										&& k + kk >= 0 && k + kk < dims.z){


										ImageType::IndexType pixelIndexNei;
										pixelIndexNei[0] = i+ii; // x position
										pixelIndexNei[1] = j+jj; // y position
										pixelIndexNei[2] = k+kk; // z position
										if (cpId == connectedImg->GetPixel(pixelIndexNei)){
											int ind2 = (k + kk)*dims.y*dims.x + (j + jj)*dims.x + (i + ii);
											if (!voxelIdAssigned[ind2]){
												voxelIdAssigned[ind2] = nextAvailableNodeId + 1;
												posOfId[nextAvailableNodeId] = make_float3(i+ii, j+jj, k+kk)*spacing;
												nextAvailableNodeId++;
											}
											int nodeId1 = voxelIdAssigned[ind] - 1, nodeId2 = voxelIdAssigned[ind2] - 1;
											addEdge(graph, nodeId1, nodeId2, (int)(sqrt(1.0*(ii*ii + jj*jj + kk*kk)) * 10));//!!!!! the MST algorithm takes integer weights. so use approximation
										}									
									}
								}
							}
						}
					}

				}
			}
		}
		
		struct Graph* mst = createGraph(numNodes);
		PrimMST(graph, mst);


		//use 2 dfs to find the longest path in the tree
		//http://cs.stackexchange.com/questions/11263/longest-path-in-an-undirected-tree-with-only-one-traversal
		int source = 0;//randomly use node 0 as start
		std::vector<int> toTraverse;
		toTraverse.push_back(source);
		std::vector<int> sourceOfToTraverse;
		sourceOfToTraverse.push_back(-1);
		std::vector<AdjListNode*> nextOfToTraverse;
		nextOfToTraverse.push_back(mst->array[source].head);
		int curLevel = 0;

		int maxDepth = -1;
		int maxDepthNode = -1;

		while (toTraverse.size() > 0){
			int curLevel = toTraverse.size() - 1;

			if (curLevel > maxDepth){
				maxDepth = curLevel;
				maxDepthNode = toTraverse[curLevel];
			}

			AdjListNode* nextNode = nextOfToTraverse[curLevel];
			if (nextNode == NULL){
				toTraverse.pop_back();
				sourceOfToTraverse.pop_back();
				nextOfToTraverse.pop_back();
			}
			else if (nextNode->dest == sourceOfToTraverse[curLevel]){
				nextOfToTraverse[curLevel] = nextNode->next;
			}
			else{
				nextOfToTraverse[curLevel] = nextNode->next;
				toTraverse.push_back(nextNode->dest);
				sourceOfToTraverse.push_back(toTraverse[curLevel]);
				nextOfToTraverse.push_back(mst->array[nextNode->dest].head);
			}
		}

		//the second traverse, which will start from the maxDepthNode, and record the maximum path
		source = maxDepthNode;
		toTraverse.clear();
		toTraverse.push_back(source);
		sourceOfToTraverse.clear();;
		sourceOfToTraverse.push_back(-1);
		nextOfToTraverse.clear();
		nextOfToTraverse.push_back(mst->array[source].head);
		curLevel = 0;
		std::vector<int> maxPath;
		maxDepth = -1;
		maxDepthNode = -1;

		while (toTraverse.size() > 0){
			int curLevel = toTraverse.size() - 1;

			if (curLevel > maxDepth){
				maxDepth = curLevel;
				maxDepthNode = toTraverse[curLevel];
				maxPath = toTraverse;
			}

			AdjListNode* nextNode = nextOfToTraverse[curLevel];
			if (nextNode == NULL){
				toTraverse.pop_back();
				sourceOfToTraverse.pop_back();
				nextOfToTraverse.pop_back();
			}
			else if (nextNode->dest == sourceOfToTraverse[curLevel]){
				nextOfToTraverse[curLevel] = nextNode->next;
			}
			else{
				nextOfToTraverse[curLevel] = nextNode->next;
				toTraverse.push_back(nextNode->dest);
				sourceOfToTraverse.push_back(toTraverse[curLevel]);
				nextOfToTraverse.push_back(mst->array[nextNode->dest].head);
			}
		}

		////////////by maxPath and posOfId, to compute a vector of sample points along the path
		std::vector<float3> posOfPathNode(maxPath.size());
		for (int i = 0; i < maxPath.size(); i++){
			posOfPathNode[i] = posOfId[maxPath[i]];
		}


		int numOfViews = 50;
		int lengthMaxPath = maxPath.size();
		views.clear();
		for (int i = 0; i < numOfViews; i++){
			float pos = 1.0 * (lengthMaxPath - 1) * i / numOfViews;
			int n1 = floor(pos), n2 = n1 + 1;
			views.push_back(posOfPathNode[n1] * (n2 - pos) + posOfPathNode[n2] * (pos - n1));
		}
	}

	return;
}