/**********************************************
File		:	lcsUtility.h
Author		:	Mingcheng Chen
Last Update	:	September 25th, 2012
***********************************************/

#ifndef __LCS_UTILITY_H
#define __LCS_UTILITY_H

#include <cstdio>
#include <vector>
#include <string>
#include <CL/opencl.h>

namespace lcs {

////////////////////////////////////////////////
void Error(const char *str);

void ConsumeChar(char goal, FILE *fin);

bool IsFloatChar(char ch);

int GPUExclusiveScanForInt(int workGroupSize, int numOfBanks,
			   cl_kernel scanKernel, cl_kernel reverseUpdateKernel, cl_mem globalArray, int length,
			   cl_command_queue commandQueue);

void CheckIntArrayInDevice(const char *fileName, cl_command_queue commandQueue, cl_mem intArr, int length);

void CheckFloatArrayInDevice(const char *fileName, cl_command_queue commandQueue, cl_mem floatArr, int length);

void GetOrignalUnorderedIntArrayFromPartialSum(const char *fileName, cl_command_queue commandQueue,
					       cl_mem intArr, int length);
	
////////////////////////////////////////////////
class Configure {
public:
	Configure(const char *fileName);
	int GetNumOfFrames() const;
	int GetSharedMemoryKilobytes() const;
	int GetBoundingBoxXRes() const;
	int GetBoundingBoxYRes() const;
	int GetBoundingBoxZRes() const;
	int GetNumOfBanks() const;
	double GetTimeStep() const;
	double GetBlockSize() const;
	double GetTimeInterval() const;
	double GetEpsilonForTetBlkIntersection() const;
	double GetEpsilon() const;
	double GetBoundingBoxMinX() const;
	double GetBoundingBoxMaxX() const;
	double GetBoundingBoxMinY() const;
	double GetBoundingBoxMaxY() const;
	double GetBoundingBoxMinZ() const;
	double GetBoundingBoxMaxZ() const;
	std::string GetFileName() const;
	std::string GetDataFilePrefix() const;
	std::string GetDataFileSuffix() const;
	std::string GetIntegration() const;
	std::vector<double> GetTimePoints() const;
	std::vector<std::string> GetDataFileIndices() const;
	bool UseDouble() const;
	bool UseUnitTestForTetBlkIntersection() const;
	bool UseUnitTestForInitialCellLocation() const;

private:
	void DefaultSetting();

	std::string fileName;
	int numOfFrames;
	int sharedMemoryKilobytes;
	int boundingBoxXRes;
	int boundingBoxYRes;
	int boundingBoxZRes;
	int numOfBanks;
	std::vector<double> timePoints;
	std::string dataFilePrefix;
	std::string dataFileSuffix;
	std::vector<std::string> dataFileIndices;
	std::string integration;
	double timeStep;
	double blockSize;
	double timeInterval;
	double epsilonForTetBlkIntersection;
	double epsilon;
	double boundingBoxMinX;
	double boundingBoxMaxX;
	double boundingBoxMinY;
	double boundingBoxMaxY;
	double boundingBoxMinZ;
	double boundingBoxMaxZ;
	bool useDouble;
	bool unitTestForTetBlkIntersection;
	bool unitTestForInitialCellLocation;
};

}

#endif
