/**********************************************
File		:	main.cpp
Author		:	Mingcheng Chen
Last Update	:	September 28th, 2012
***********************************************/

#include "lcs.h"
#include "lcsUtility.h"
#include "lcsUnitTest.h"
#include "lcsGeometry.h"

#include <CL/opencl.h>
#include <ctime>
#include <string>
#include <algorithm>

const char *configurationFile = "RungeKutta4.conf";

const char *tetrahedronBlockIntersectionKernel = "lcsTetrahedronBlockIntersectionKernel.cl";
const char *initialCellLocationKernel = "lcsInitialCellLocationKernel.cl";
const char *bigBlockInitializationForPositionsKernel = "lcsBigBlockInitializationForPositionsKernel.cl";
const char *bigBlockInitializationForVelocitiesKernel = "lcsBigBlockInitializationForVelocitiesKernel.cl";
const char *exclusiveScanForIntKernels = "lcsExclusiveScanForIntKernels.cl";
const char *collectActiveParticlesForNewIntervalKernels = "lcsCollectActiveParticlesForNewIntervalKernels.cl";
const char *collectActiveParticlesForNewRunKernels = "lcsCollectActiveParticlesForNewRunKernels.cl";
const char *redistributeParticlesKernels = "lcsRedistributeParticlesKernels.cl";
const char *collectEveryKElementKernel = "lcsGetStartOffsetInParticlesKernel.cl";
const char *assignWorkGroupsKernels = "lcsGetGroupsForBlocksKernels.cl";

const char *blockedTracingKernelPrefix = "lcsBlockedTracingKernelOf";
const char *blockedTracingKernelSuffix = ".cl";

const char *lastPositionFile = "lcsLastPositions.txt";

lcs::Configure *configure;

lcs::Frame **frames;
int numOfFrames;

int *tetrahedralConnectivities, *tetrahedralLinks;
void *vertexPositions;
int globalNumOfCells, globalNumOfPoints;
double globalMinX, globalMaxX, globalMinY, globalMaxY, globalMinZ, globalMaxZ;

double blockSize;
int numOfBlocksInX, numOfBlocksInY, numOfBlocksInZ;

// For tetrahedron-block intersection
int *xLeftBound, *xRightBound, *yLeftBound, *yRightBound, *zLeftBound, *zRightBound;
int numOfQueries;
int *queryTetrahedron, *queryBlock;
char *queryResults; // Whether certain tetrahedron intersects with certain block

// For blocks
int numOfBlocks, numOfInterestingBlocks, numOfBigBlocks;
lcs::BlockRecord **blocks;
bool *canFitInSharedMemory;
int *startOffsetInCell, *startOffsetInPoint;
int *startOffsetInCellForBig, *startOffsetInPointForBig;

// For initial cell location
int *initialCellLocations;

// For tracing
lcs::ParticleRecord **particleRecords;
int *exitCells;
int numOfInitialActiveParticles;

// OpenCL variables

// error, platform, device, context and command queue
cl_int err;
cl_uint numOfPlatforms, numOfDevices;
cl_platform_id *platformIDs;
cl_device_id *deviceIDs;
cl_context context;
cl_command_queue commandQueue;

// Host memory for global geometry
cl_mem h_tetrahedralConnectivities, h_tetrahedralLinks, h_vertexPositions;

// Host memory for tetrahedron-block intersection queries
cl_mem h_queryTetrahedron, h_queryBlock;

// Device memory for exclusive scan for int
cl_mem d_exclusiveScanArrayForInt;

// Device memory for interesting block map
cl_mem d_interestingBlockMap;

// Device memory for (tet, blk) to local tet ID map
cl_mem d_startOffsetsInLocalIDMap;
cl_mem d_blocksOfTets;
cl_mem d_localIDsOfTets;

// Device memory for particle redistribution
cl_mem d_numOfParticlesByStageInBlocks; // It depends on the maximum stage number of the integration method.
cl_mem d_interestingBlockMarks;
cl_mem d_particleOrders; // The local order number in (block, stage) group
cl_mem d_blockLocations;

// Device memory for global geometry
cl_mem d_tetrahedralConnectivities, d_tetrahedralLinks, d_vertexPositions;
cl_mem d_queryTetrahedron, d_queryBlock;
cl_mem d_queryResults;

// Device memory for cell locations of particles
cl_mem d_cellLocations;

// Device memory for local geometry in blocks
cl_mem d_localConnectivities, d_localLinks;
cl_mem d_globalCellIDs, d_globalPointIDs;
cl_mem d_startOffsetInCell, d_startOffsetInPoint;

// Device memory for particle
cl_mem d_activeBlockOfParticles;
cl_mem d_localTetIDs;
cl_mem d_exitCells;
cl_mem d_activeParticles[2];

int currActiveParticleArray;

cl_mem d_stages;
cl_mem d_lastPositionForRK4;
cl_mem d_k1ForRK4, d_k2ForRK4, d_k3ForRK4;
cl_mem d_pastTimes;
cl_mem d_placesOfInterest;

// Device memory for velocities
cl_mem d_velocities[2];

// Device memory for big blocks
cl_mem d_bigBlocks;
cl_mem d_startOffsetInCellForBig, d_startOffsetInPointForBig;
cl_mem d_vertexPositionsForBig, d_startVelocitiesForBig, d_endVelocitiesForBig;

// Device memory for canFitInSharedMemory flags
cl_mem d_canFitInSharedMemory;

// Device memory for active block list
cl_mem d_activeBlocks;
cl_mem d_activeBlockIndices;
cl_mem d_numOfActiveBlocks;

// Device memory for tracing work groups distribution
cl_mem d_numOfGroupsForBlocks;
cl_mem d_blockOfGroups;
cl_mem d_offsetInBlocks;

// Device memory for start offsets of particles in active blocks
cl_mem d_startOffsetInParticles;

// Device memory for particles grouped in blocks
cl_mem d_blockedActiveParticles;

int GetBlockID(int x, int y, int z) {
	return (x * numOfBlocksInY + y) * numOfBlocksInZ + z;
}

void GetXYZFromBlockID(int blockID, int &x, int &y, int &z) {
	z = blockID % numOfBlocksInZ;
	blockID /= numOfBlocksInZ;
	y = blockID % numOfBlocksInY;
	x = blockID / numOfBlocksInY;
}

void GetXYZFromPosition(const lcs::Vector &position, int &x, int &y, int &z) {
	x = (int)((position.GetX() - globalMinX) / blockSize);
	y = (int)((position.GetY() - globalMinY) / blockSize);
	z = (int)((position.GetZ() - globalMinZ) / blockSize);
}

void SystemTest() {
	printf("sizeof(double) = %d\n", sizeof(double));
	printf("sizeof(float) = %d\n", sizeof(float));
	printf("sizeof(int) = %d\n", sizeof(int));
	printf("sizeof(int *) = %d\n", sizeof(int *));
	printf("sizeof(char) = %d\n", sizeof(char));

	printf("sizeof(cl_float) = %d\n", sizeof(cl_float));
	printf("sizeof(cl_double) = %d\n", sizeof(cl_double));

	printf("\n");
}

void ReadConfFile() {
	configure = new lcs::Configure(configurationFile);
	if (configure->GetIntegration() == "FE") lcs::ParticleRecord::SetDataType(lcs::ParticleRecord::FE);
	if (configure->GetIntegration() == "RK4") lcs::ParticleRecord::SetDataType(lcs::ParticleRecord::RK4);
	if (configure->GetIntegration() == "RK45") lcs::ParticleRecord::SetDataType(lcs::ParticleRecord::RK45);
	printf("\n");
}

void LoadFrames() {
	numOfFrames = configure->GetNumOfFrames();
	frames = new lcs::Frame *[numOfFrames];

	for (int i = 0; i < numOfFrames; i++) {
		double timePoint = configure->GetTimePoints()[i];
		std::string dataFileName = configure->GetDataFilePrefix() + configure->GetDataFileIndices()[i] + 
					   "." + configure->GetDataFileSuffix();
		printf("Loading frame %d (file = %s) ... ", i, dataFileName.c_str());
		frames[i] = new lcs::Frame(timePoint, dataFileName.c_str());
		printf("Done.\n");
	}
	printf("\n");
}

void GetTopologyAndGeometry() {
	globalNumOfCells = frames[0]->GetTetrahedralGrid()->GetNumOfCells();
	globalNumOfPoints = frames[0]->GetTetrahedralGrid()->GetNumOfVertices();

	tetrahedralConnectivities = new int [globalNumOfCells * 4];
	tetrahedralLinks = new int [globalNumOfCells * 4];

	if (configure->UseDouble())
		vertexPositions = new double [globalNumOfPoints * 3];
	else
		vertexPositions = new float [globalNumOfPoints * 3];

	frames[0]->GetTetrahedralGrid()->ReadConnectivities(tetrahedralConnectivities);
	frames[0]->GetTetrahedralGrid()->ReadLinks(tetrahedralLinks);

	if (configure->UseDouble())
		frames[0]->GetTetrahedralGrid()->ReadPositions((double *)vertexPositions);
	else
		frames[0]->GetTetrahedralGrid()->ReadPositions((float *)vertexPositions);
}

void GetGlobalBoundingBox() {
	lcs::Vector firstPoint = frames[0]->GetTetrahedralGrid()->GetVertex(0);

	globalMaxX = globalMinX = firstPoint.GetX();
	globalMaxY = globalMinY = firstPoint.GetY();
	globalMaxZ = globalMinZ = firstPoint.GetZ();

	for (int i = 1; i < globalNumOfPoints; i++) {
		lcs::Vector point = frames[0]->GetTetrahedralGrid()->GetVertex(i);

		globalMaxX = std::max(globalMaxX, point.GetX());
		globalMinX = std::min(globalMinX, point.GetX());
		
		globalMaxY = std::max(globalMaxY, point.GetY());
		globalMinY = std::min(globalMinY, point.GetY());

		globalMaxZ = std::max(globalMaxZ, point.GetZ());
		globalMinZ = std::min(globalMinZ, point.GetZ());
	}

	printf("Global Bounding Box\n");
	printf("X: [%lf, %lf], length = %lf\n", globalMinX, globalMaxX, globalMaxX - globalMinX);
	printf("Y: [%lf, %lf], length = %lf\n", globalMinY, globalMaxY, globalMaxY - globalMinY);
	printf("Z: [%lf, %lf], length = %lf\n", globalMinZ, globalMaxZ, globalMaxZ - globalMinZ);
	printf("\n");
}

void CalculateNumOfBlocksInXYZ() {
	blockSize = configure->GetBlockSize();

	numOfBlocksInX = (int)((globalMaxX - globalMinX) / blockSize) + 1;
	numOfBlocksInY = (int)((globalMaxY - globalMinY) / blockSize) + 1;
	numOfBlocksInZ = (int)((globalMaxZ - globalMinZ) / blockSize) + 1;
}

void PrepareTetrahedronBlockIntersectionQueries() {
	// Get the bounding box for every tetrahedral cell
	xLeftBound = new int [globalNumOfCells];
	xRightBound = new int [globalNumOfCells];
	yLeftBound = new int [globalNumOfCells];
	yRightBound = new int [globalNumOfCells];
	zLeftBound = new int [globalNumOfCells];
	zRightBound = new int [globalNumOfCells];

	numOfQueries = 0;
	for (int i = 0; i < globalNumOfCells; i++) {
		lcs::Tetrahedron tetrahedron = frames[0]->GetTetrahedralGrid()->GetTetrahedron(i);
		lcs::Vector firstPoint = tetrahedron.GetVertex(0);
		double localMinX, localMaxX, localMinY, localMaxY, localMinZ, localMaxZ;
		localMaxX = localMinX = firstPoint.GetX();
		localMaxY = localMinY = firstPoint.GetY();
		localMaxZ = localMinZ = firstPoint.GetZ();
		for (int j = 1; j < 4; j++) {
			lcs::Vector point = tetrahedron.GetVertex(j);
			localMaxX = std::max(localMaxX, point.GetX());
			localMinX = std::min(localMinX, point.GetX());
			localMaxY = std::max(localMaxY, point.GetY());
			localMinY = std::min(localMinY, point.GetY());
			localMaxZ = std::max(localMaxZ, point.GetZ());
			localMinZ = std::min(localMinZ, point.GetZ());
		}

		xLeftBound[i] = (int)((localMinX - globalMinX) / blockSize);
		xRightBound[i] = (int)((localMaxX - globalMinX) / blockSize);
		yLeftBound[i] = (int)((localMinY - globalMinY) / blockSize);
		yRightBound[i] = (int)((localMaxY - globalMinY) / blockSize);
		zLeftBound[i] = (int)((localMinZ - globalMinZ) / blockSize);
		zRightBound[i] = (int)((localMaxZ - globalMinZ) / blockSize);

		numOfQueries += (xRightBound[i] - xLeftBound[i] + 1) *
				(yRightBound[i] - yLeftBound[i] + 1) *
				(zRightBound[i] - zLeftBound[i] + 1);
	}

	// Prepare host input and output arrays
	queryTetrahedron = new int [numOfQueries];
	queryBlock = new int [numOfQueries];
	queryResults = new char [numOfQueries];

	int currQuery = 0;

	for (int i = 0; i < globalNumOfCells; i++)
		for (int xItr = xLeftBound[i]; xItr <= xRightBound[i]; xItr++)
			for (int yItr = yLeftBound[i]; yItr <= yRightBound[i]; yItr++)
				for (int zItr = zLeftBound[i]; zItr <= zRightBound[i]; zItr++) {
					queryTetrahedron[currQuery] = i;
					queryBlock[currQuery] = GetBlockID(xItr, yItr, zItr);
					currQuery++;
				}

	// Release bounding box arrays
	delete [] xLeftBound;
	delete [] xRightBound;
	delete [] yLeftBound;
	delete [] yRightBound;
	delete [] zLeftBound;
	delete [] zRightBound;
}

cl_program CreateProgram(const char *kernelFile, const char *kernelName) {

	/// DEBUG ///
	bool debug = !strcmp(kernelName, "blocked tracing");


	// Load the kernel code
	FILE *fin = fopen(kernelFile, "r");
	if (fin == NULL) {
		char str[100];
		sprintf(str, "Fail to load the %s kernel", kernelName);
		lcs::Error(str);
	}

	std::string kernelCode = "";

	if (!configure->UseDouble()) kernelCode = "#define double float\n\n";

	char ch;
	for (; (ch = fgetc(fin)) != EOF; kernelCode += ch);

	fclose(fin);

	size_t codeLength = kernelCode.length() + 1; // Consider the tailing 0
	const char *codeString = kernelCode.c_str();

	// Create a program based on the kernel code
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&codeString, &codeLength, &err);
	if (err) {
		char str[100];
		sprintf(str, "Fail to create a program for the %s kernel", kernelName);
		lcs::Error(str);
	}

	// Build the program and output the build information

	/// DEBUG ///
	/*if (debug)
		err = clBuildProgram(program, 0, NULL, "-cl-opt-disable", NULL, NULL);
	else*/
		err = clBuildProgram(program, 0, NULL, "", NULL, NULL);


	bool compilationFailure = err;

	size_t lengthOfBuildInfo;
	err = clGetProgramBuildInfo(program, deviceIDs[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &lengthOfBuildInfo);
	if (err) lcs::Error("Fail to get the length of the program build information");

	char *buildInfo = new char [lengthOfBuildInfo + 1];
	buildInfo[lengthOfBuildInfo] = 0;
	err = clGetProgramBuildInfo(program, deviceIDs[0], CL_PROGRAM_BUILD_LOG, lengthOfBuildInfo, buildInfo, NULL);
	if (err) lcs::Error("Fail to get the program build information");

	printf("The program build information is as follows.\n\n");

	if (buildInfo[0] == '\n') printf("Successful Compilation\n\n");
	else printf("%s\n", buildInfo);
	printf("* End of Build Information *\n");
	delete [] buildInfo;
	printf("\n");
	if (compilationFailure) {
		char str[100];
		sprintf(str, "Fail to build the program of the %s kernel", kernelName);
		lcs::Error(str);
	}

	return program;
}

void LaunchGPUforIntersectionQueries() {
	printf("Start to use GPU to process tetrahedron-block intersection queries ...\n");
	printf("\n");

	int startTime = clock();

	// Get platform information
	err = clGetPlatformIDs(0, NULL, &numOfPlatforms);
	if (err) lcs::Error("Fail to get the number of platforms");
	printf("The machine has %d platform(s) for OpenCL.\n", numOfPlatforms);

	platformIDs = new cl_platform_id [numOfPlatforms];
	err = clGetPlatformIDs(numOfPlatforms, platformIDs, NULL);
	if (err) lcs::Error("Fail to get the platform list");

	int cudaPlatformID = -1;

	for (int i = 0; i < numOfPlatforms; i++) {
		char platformName[50];
		err = clGetPlatformInfo(platformIDs[i], CL_PLATFORM_NAME, 50, platformName, NULL);
		if (err) lcs::Error("Fail to get the platform name");
		printf("Platform %d is %s\n", i + 1, platformName);
		if (!strcmp(platformName, "NVIDIA CUDA")) cudaPlatformID = i;
	}
	printf("\n");

	if (cudaPlatformID == -1) lcs::Error("Fail to find an NVIDIA CUDA platform");

	printf("Platform %d (NVIDIA CUDA) is chosen for use.\n", cudaPlatformID + 1);
	printf("\n");

	// Get device information
	err = clGetDeviceIDs(platformIDs[cudaPlatformID], CL_DEVICE_TYPE_GPU, 0, NULL, &numOfDevices);
	if (err) lcs::Error("Fail to get the number of devices");
	printf("CUDA platform has %d device(s).\n", numOfDevices);

	deviceIDs = new cl_device_id [numOfDevices];
	err = clGetDeviceIDs(platformIDs[cudaPlatformID], CL_DEVICE_TYPE_GPU, numOfDevices, deviceIDs, NULL);
	if (err) lcs::Error("Fail to get the device list");
	for (int i = 0; i < numOfDevices; i++) {
		char deviceName[50];
		err = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_NAME, 50, deviceName, NULL);
		if (err) lcs::Error("Fail to get the device name");
		printf("Device %d is %s\n", i + 1, deviceName);
	}
	printf("\n");

	// Create a context
	context = clCreateContext(NULL, numOfDevices, deviceIDs, NULL, NULL, &err);
	if (err) lcs::Error("Fail to create a context");

	printf("Device 1 is chosen for use.\n");
	printf("\n");

	// Create a command queue for the first device
	commandQueue = clCreateCommandQueue(context, deviceIDs[0],
					    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err);
	if (err) lcs::Error("Fail to create a command queue");

	// create the program
	cl_program program = CreateProgram(tetrahedronBlockIntersectionKernel, "tetrahedron-block intersection");
	
	// Create OpenCL buffer pointing to the host tetrahedralConnectivities
	h_tetrahedralConnectivities = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
						     sizeof(int) * globalNumOfCells * 4, tetrahedralConnectivities, &err);
	if (err) lcs::Error("Fail to create a buffer for host tetrahedralConnectivities");

	// Create OpenCL buffer pointing to the host tetrahedralLinks
	h_tetrahedralLinks = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
					    sizeof(int) * globalNumOfCells * 4, tetrahedralLinks, &err);
	if (err) lcs::Error("Fail to create a buffer for host tetrahedralLinks");

	// Create OpenCL buffer pointing to the host vertexPositions
	if (configure->UseDouble())
		h_vertexPositions = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
						   sizeof(double) * globalNumOfPoints * 3, vertexPositions, &err);
	else
		h_vertexPositions = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
						   sizeof(float) * globalNumOfPoints * 3, vertexPositions, &err);
	if (err) lcs::Error("Fail to create a buffer for host vertexPositions");

	// Create OpenCL buffer pointing to the host queryTetrahedron
	h_queryTetrahedron = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
					    sizeof(int) * numOfQueries, queryTetrahedron, &err);
	if (err) lcs::Error("Fail to create a buffer for host queryTetrahedron");

	// Create OpenCL buffer pointing to the host queryBlock
	h_queryBlock = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				      sizeof(int) * numOfQueries, queryBlock, &err);
	if (err) lcs::Error("Fail to create a buffer for host queryBlock");

	// Create OpenCL buffer pointing to the device tetrahedralConnectivities
	d_tetrahedralConnectivities = clCreateBuffer(context, CL_MEM_READ_ONLY,
						     sizeof(int) * globalNumOfCells * 4, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device tetrahedralConnectivities");

	// Create OpenCL buffer pointing to the device vertexPositions
	if (configure->UseDouble())
		d_vertexPositions = clCreateBuffer(context, CL_MEM_READ_ONLY,
						   sizeof(double) * globalNumOfPoints * 3, NULL, &err);
	else
		d_vertexPositions = clCreateBuffer(context, CL_MEM_READ_ONLY,
						   sizeof(float) * globalNumOfPoints * 3, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device vertexPositions");

	// Create OpenCL buffer pointing to the device queryTetrahedron
	d_queryTetrahedron = clCreateBuffer(context, CL_MEM_READ_ONLY,
					    sizeof(int) * numOfQueries, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device queryTetrahedron");

	// Create OpenCL buffer pointing to the device queryBlock
	d_queryBlock = clCreateBuffer(context, CL_MEM_READ_ONLY,
				      sizeof(int) * numOfQueries, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device queryBlock");

	// Create OpenCL buffer pointing to the device queryResults (output)
	d_queryResults = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
					sizeof(char) * numOfQueries, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device queryResults");

	// Copy from host to device
	cl_event copyHConnToDConn;
	cl_event copyHPosiToDPosi;
	cl_event copyHqueryTetToDqueryTet;
	cl_event copyHqueryBlkToDqueryBlk;

	err = clEnqueueCopyBuffer(commandQueue, h_tetrahedralConnectivities, d_tetrahedralConnectivities, 0, 0,
				  sizeof(int) * globalNumOfCells * 4, 0, NULL, &copyHConnToDConn);
	if (err) lcs::Error("Fail to enqueue copyHConnToDConn");
	
	if (configure->UseDouble())
		err = clEnqueueCopyBuffer(commandQueue, h_vertexPositions, d_vertexPositions, 0, 0,
					  sizeof(double) * globalNumOfPoints * 3, 0, NULL, &copyHPosiToDPosi);
	else
		err = clEnqueueCopyBuffer(commandQueue, h_vertexPositions, d_vertexPositions, 0, 0,
					  sizeof(float) * globalNumOfPoints * 3, 0, NULL, &copyHPosiToDPosi);
	if (err) lcs::Error("Fail to enqueue copyHPosiToDPosi");

	err = clEnqueueCopyBuffer(commandQueue, h_queryTetrahedron, d_queryTetrahedron, 0, 0,
				  sizeof(int) * numOfQueries, 0, NULL, &copyHqueryTetToDqueryTet);
	if (err) lcs::Error("Fail to enqueue copyHqueryTetToDqueryTet");

	err = clEnqueueCopyBuffer(commandQueue, h_queryBlock, d_queryBlock, 0, 0,
				  sizeof(int) * numOfQueries, 0, NULL, &copyHqueryBlkToDqueryBlk);
	if (err) lcs::Error("Fail to enqueue copyHqueryBlkToDqueryBlk");

	// Create the kernel
	cl_kernel kernel = clCreateKernel(program, "TetrahedronBlockIntersection", &err);
	if (err) lcs::Error("Fail to create the kernel for Tetrahedron-Block Intersection");

	// Get the work group size
	size_t maxWorkGroupSize;
	err = clGetKernelWorkGroupInfo(kernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSize, NULL);
	printf("maxWorkGroupSize = %d\n", maxWorkGroupSize);
	size_t workGroupSize = 1;
	for (; workGroupSize * 2 <= maxWorkGroupSize; workGroupSize <<= 1);
	printf("workGroupSize = %d\n", workGroupSize);
	printf("\n");

	// Set the argument values for the kernel
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_vertexPositions);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_tetrahedralConnectivities);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_queryTetrahedron);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_queryBlock);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_queryResults);
	
	cl_int cl_numOfBlocksInY = numOfBlocksInY;
	cl_int cl_numOfBlocksInZ = numOfBlocksInZ;
	
	clSetKernelArg(kernel, 5, sizeof(cl_int), &cl_numOfBlocksInY);
	clSetKernelArg(kernel, 6, sizeof(cl_int), &cl_numOfBlocksInZ);

	void *floatNumbers;

	if (configure->UseDouble()) {
		floatNumbers = new double [5];
		((double *)floatNumbers)[0] = globalMinX;
		((double *)floatNumbers)[1] = globalMinY;
		((double *)floatNumbers)[2] = globalMinZ;
		((double *)floatNumbers)[3] = blockSize;
		((double *)floatNumbers)[4] = configure->GetEpsilon();
		for (int i = 0; i < 5; i++)
			clSetKernelArg(kernel, 7 + i, sizeof(cl_double), (double *)floatNumbers + i);
	} else {
		floatNumbers = new float [5];
		((float *)floatNumbers)[0] = (float)globalMinX;
		((float *)floatNumbers)[1] = (float)globalMinY;
		((float *)floatNumbers)[2] = (float)globalMinZ;
		((float *)floatNumbers)[3] = (float)blockSize;
		((float *)floatNumbers)[4] = (float)configure->GetEpsilonForTetBlkIntersection();

		for (int i = 0; i < 5; i++)
			clSetKernelArg(kernel, 7 + i, sizeof(cl_float), (float *)floatNumbers + i);
	}

	if (configure->UseDouble())
		delete [] (double *)floatNumbers;
	else
		delete [] (float *)floatNumbers;

	cl_int cl_numOfQueries = numOfQueries;
	clSetKernelArg(kernel, 12, sizeof(cl_int), &cl_numOfQueries);

	// Set local / global work size
	size_t localWorkSize[] = {workGroupSize};
	size_t globalWorkSize[] = {((numOfQueries - 1) / workGroupSize + 1) * workGroupSize};

	// Enqueue the kernel event
	cl_event kernelEvent;
	cl_event eventList[] = {copyHConnToDConn, copyHPosiToDPosi, copyHqueryTetToDqueryTet, copyHqueryBlkToDqueryBlk};

	err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize,
				     sizeof(eventList) / sizeof(cl_event), eventList, &kernelEvent);

	if (err) lcs::Error("Fail to enqueue tetrahedron-block intersection kernel");

	// Copy from device to host
	cl_event copyDresTHres;
	err = clEnqueueReadBuffer(commandQueue, d_queryResults, CL_FALSE, 0, sizeof(char) * numOfQueries,
				  queryResults, 1, &kernelEvent, &copyDresTHres);
	if (err) lcs::Error("Fail to enqueue copyDresTHres");

	// Synchronization Point
	clFinish(commandQueue);

	// Release some resources
	clReleaseMemObject(d_queryTetrahedron);
	clReleaseMemObject(d_queryBlock);
	clReleaseMemObject(d_queryResults);

	clReleaseEvent(copyHConnToDConn);
	clReleaseEvent(copyHPosiToDPosi);
	clReleaseEvent(copyHqueryTetToDqueryTet);
	clReleaseEvent(copyHqueryBlkToDqueryBlk);
	clReleaseEvent(kernelEvent);
	clReleaseEvent(copyDresTHres);

	int endTime = clock();

	printf("First 10 results: ");
	for (int i = 0; i < 10; i++)
		printf("%d", queryResults[i]);
	printf("\n\n");

	printf("The GPU Kernel for tetrahedron-block intersection queries cost %lf sec.\n",
	       (endTime - startTime) * 1.0 / CLOCKS_PER_SEC);
	printf("\n");

	// Unit Test for Tetrahedron-Block Intersection Kernel
	startTime = clock();

	if (configure->UseUnitTestForTetBlkIntersection()) {
		lcs::UnitTestForTetBlkIntersection(frames[0]->GetTetrahedralGrid(),
						   blockSize, globalMinX, globalMinY, globalMinZ,
						   numOfBlocksInY, numOfBlocksInZ,
						   queryTetrahedron, queryBlock, queryResults,
						   numOfQueries, configure->GetEpsilon());
		printf("\n");
	}

	endTime = clock();

	printf("The unit test cost %lf sec.\n", (endTime - startTime) * 1.0 / CLOCKS_PER_SEC);
	printf("\n");
}

void DivisionProcess() {
	// Filter out empty blocks and build interestingBlockMap
	numOfBlocks = numOfBlocksInX * numOfBlocksInY * numOfBlocksInZ;
	int *interestingBlockMap = new int [numOfBlocks];
	memset(interestingBlockMap, 255, sizeof(int) * numOfBlocks);

	d_interestingBlockMap = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * numOfBlocks, NULL, &err);
	if (err) lcs::Error("Fail to create device interestingBlockMap");

	numOfInterestingBlocks = 0;
	for (int i = 0; i < numOfQueries; i++)
		if (queryResults[i]) {
			int blockID = queryBlock[i];
			if (interestingBlockMap[blockID] != -1) continue;
			interestingBlockMap[blockID] = numOfInterestingBlocks++;
		}

	err = clEnqueueWriteBuffer(commandQueue, d_interestingBlockMap, CL_TRUE, 0, sizeof(int) * numOfBlocks,
				   interestingBlockMap, 0, NULL, NULL);
	if (err) lcs::Error("Fail to write to device interestingBlockMap");

	// Count the numbers of tetrahedrons in non-empty blocks and the numbers of blocks of tetrahedrons
	int sizeOfHashMap = 0;

	int *numOfTetrahedronsInBlock, *numOfBlocksOfTetrahedron;
	int **cellsInBlock;

	numOfTetrahedronsInBlock = new int [numOfInterestingBlocks];
	memset(numOfTetrahedronsInBlock, 0, sizeof(int) * numOfInterestingBlocks);

	numOfBlocksOfTetrahedron = new int [globalNumOfCells];
	memset(numOfBlocksOfTetrahedron, 0, sizeof(int) * globalNumOfCells);

	for (int i = 0; i < numOfQueries; i++)
		if (queryResults[i]) {
			numOfTetrahedronsInBlock[interestingBlockMap[queryBlock[i]]]++;
			numOfBlocksOfTetrahedron[queryTetrahedron[i]]++;
			sizeOfHashMap++;
		}

	// Initialize device arrays
	d_startOffsetsInLocalIDMap = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (globalNumOfCells + 1), NULL, &err);
	if (err) lcs::Error("Fail to create device startOffsetsInLocalMap");

	d_blocksOfTets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * sizeOfHashMap, NULL, &err);
	if (err) lcs::Error("Fail to create device blocksOfTets");

	d_localIDsOfTets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * sizeOfHashMap, NULL, &err);
	if (err) lcs::Error("Fail to create device localIDsOfTets");

	// Initialize some work arrays
	int *startOffsetsInLocalIDMap = new int [globalNumOfCells + 1];
	
	startOffsetsInLocalIDMap[0] = 0;
	for (int i = 1; i <= globalNumOfCells; i++)
		startOffsetsInLocalIDMap[i] = startOffsetsInLocalIDMap[i - 1] + numOfBlocksOfTetrahedron[i - 1];

	int *topOfCells = new int [globalNumOfCells];
	memset(topOfCells, 0, sizeof(int) * globalNumOfCells);

	int *blocksOfTets = new int [sizeOfHashMap];
	int *localIDsOfTets = new int [sizeOfHashMap];

	// Fill cellsInblock and build local cell ID map
	cellsInBlock = new int * [numOfInterestingBlocks];
	for (int i = 0; i < numOfInterestingBlocks; i++)
		cellsInBlock[i] = new int [numOfTetrahedronsInBlock[i]];
	int *heads = new int [numOfInterestingBlocks];
	memset(heads, 0, sizeof(int) * numOfInterestingBlocks);
	for (int i = 0; i < numOfQueries; i++)
		if (queryResults[i]) {
			int tetrahedronID = queryTetrahedron[i];
			int blockID = interestingBlockMap[queryBlock[i]];

			int positionInHashMap = startOffsetsInLocalIDMap[tetrahedronID] + topOfCells[tetrahedronID];
			blocksOfTets[positionInHashMap] = queryBlock[i];
			localIDsOfTets[positionInHashMap] = heads[blockID];
			topOfCells[tetrahedronID]++;

			cellsInBlock[blockID][heads[blockID]++] = tetrahedronID;
		}
	delete [] heads;

	/// DEBUG ///
	for (int i = 0; i < globalNumOfCells; i++)
		if (startOffsetsInLocalIDMap[i] >= startOffsetsInLocalIDMap[i + 1]) {
			printf("%d %d\n", i, startOffsetsInLocalIDMap[i]);
			lcs::Error("local ID Map error");
		}

	printf("hash size = %d\n", startOffsetsInLocalIDMap[globalNumOfCells]);
	printf("sizeOfHashMap = %d\n", sizeOfHashMap);

	printf("globalNumOfCells = %d\n", globalNumOfCells);

	// Fill some device arrays
	err = clEnqueueWriteBuffer(commandQueue, d_startOffsetsInLocalIDMap, CL_TRUE, 0, sizeof(int) * (globalNumOfCells + 1),
				   startOffsetsInLocalIDMap, 0, NULL, NULL);
	if (err) lcs::Error("Fail to write to device startOffsetsInLocalIDMap");

	err = clEnqueueWriteBuffer(commandQueue, d_blocksOfTets, CL_TRUE, 0, sizeof(int) * sizeOfHashMap,
				   blocksOfTets, 0, NULL, NULL);
	if (err) lcs::Error("Fail to write to device blocksOfTets");

	err = clEnqueueWriteBuffer(commandQueue, d_localIDsOfTets, CL_TRUE, 0, sizeof(int) * sizeOfHashMap,
				   localIDsOfTets, 0, NULL, NULL);
	if (err) lcs::Error("Fail to write to device localIDsOfTets");

	// Delete some work arrays
	delete [] startOffsetsInLocalIDMap;
	delete [] topOfCells;
	delete [] blocksOfTets;
	delete [] localIDsOfTets;
	delete [] interestingBlockMap;

	// Initialize blocks and release cellsInBlock and numOfTetrahedronsInBlock
	blocks = new lcs::BlockRecord * [numOfInterestingBlocks];
	for (int i = 0; i < numOfInterestingBlocks; i++) {
		blocks[i] = new lcs::BlockRecord();
		blocks[i]->SetLocalNumOfCells(numOfTetrahedronsInBlock[i]);
		blocks[i]->CreateGlobalCellIDs(cellsInBlock[i]);
		delete [] cellsInBlock[i];
	}
	delete [] cellsInBlock;
	delete [] numOfTetrahedronsInBlock;

	// Initialize work arrays
	int *cellMarks = new int [globalNumOfCells];
	int *pointMarks = new int [globalNumOfPoints];
	int *localPointIDs = new int [globalNumOfPoints];
	int *localCellIDs = new int [globalNumOfCells];
	int *pointList = new int [globalNumOfPoints];
	int *tempConnectivities = new int [globalNumOfCells * 4];
	int *tempLinks = new int [globalNumOfCells * 4];
	int markCount = 0;

	memset(cellMarks, 0, sizeof(int) * globalNumOfCells);
	memset(pointMarks, 0, sizeof(int) * globalNumOfPoints);
	
	// Process blocks
	int smallEnoughBlocks = 0;

	canFitInSharedMemory = new bool [numOfInterestingBlocks];

	for (int i = 0; i < numOfInterestingBlocks; i++) {
		markCount++;
		int population = 0;

		// Get local points
		for (int j = 0; j < blocks[i]->GetLocalNumOfCells(); j++) {
			int globalCellID = blocks[i]->GetGlobalCellID(j);
			cellMarks[globalCellID] = markCount;
			localCellIDs[globalCellID] = j;

			for (int k = 0; k < 4; k++) {
				int globalPointID = tetrahedralConnectivities[(globalCellID << 2) + k];
				if (globalPointID == -1 || pointMarks[globalPointID] == markCount) continue;
				pointMarks[globalPointID] = markCount;
				localPointIDs[globalPointID] = population;
				pointList[population++] = globalPointID;
			}
		}

		blocks[i]->SetLocalNumOfPoints(population);
		blocks[i]->CreateGlobalPointIDs(pointList);

		// Mark whether the block can fit into the shared memory
		int currentBlockMemoryCost = blocks[i]->EvaluateNumOfBytes();

		if (currentBlockMemoryCost <= configure->GetSharedMemoryKilobytes() * 1024) {
			smallEnoughBlocks++;
			canFitInSharedMemory[i] = true;
		} else
			canFitInSharedMemory[i] = false;

		// Calculate the local connectivity and link
		for (int j = 0; j < blocks[i]->GetLocalNumOfCells(); j++) {
			int globalCellID = blocks[i]->GetGlobalCellID(j);

			// Fill tempConnectivities
			for (int k = 0; k < 4; k++) {
				int globalPointID = tetrahedralConnectivities[(globalCellID << 2) + k];
				int localPointID;
				if (globalPointID != -1 && pointMarks[globalPointID] == markCount)
					localPointID = localPointIDs[globalPointID];
				else localPointID = -1;
				tempConnectivities[(j << 2) + k] = localPointID;
			}

			// Fill tempLinks
			for (int k = 0; k < 4; k++) {
				int globalNeighborID = tetrahedralLinks[(globalCellID << 2) + k];
				int localNeighborID;
				if (globalNeighborID != -1 && cellMarks[globalNeighborID] == markCount)
					localNeighborID = localCellIDs[globalNeighborID];
				else localNeighborID = -1;
				tempLinks[(j << 2) + k] = localNeighborID;
			}
		}

		blocks[i]->CreateLocalConnectivities(tempConnectivities);
		blocks[i]->CreateLocalLinks(tempLinks);
	}
	
	printf("Division is done. smallEnoughBlocks = %d\n", smallEnoughBlocks);
	printf("\n");

	// Select big blocks
	int *bigBlocks = new int [numOfInterestingBlocks];
	numOfBigBlocks = 0;
	for (int i = 0; i < numOfInterestingBlocks; i++)
		if (!canFitInSharedMemory[i])
			bigBlocks[numOfBigBlocks++] = i;

	d_bigBlocks = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * numOfBigBlocks, NULL, &err);
	if (err) lcs::Error("Fail to create device bigBlocks");

	err = clEnqueueWriteBuffer(commandQueue, d_bigBlocks, CL_TRUE, 0, sizeof(int) * numOfBigBlocks, bigBlocks, 0, NULL, NULL);
	if (err) lcs::Error("Fail to write to d_bigBlockFail to write to d_bigBlocks");

	delete [] bigBlocks;

	// Release work arrays
	delete [] cellMarks;
	delete [] pointMarks;
	delete [] localPointIDs;
	delete [] localCellIDs;
	delete [] pointList;
	delete [] tempConnectivities;
	delete [] tempLinks;

	// Some statistics
	int minPos = globalNumOfCells, maxPos = 0;
	int numOfUnder100 = 0, numOfUnder200 = 0;

	for (int i = 0; i < numOfInterestingBlocks; i++) {
		maxPos = std::max(maxPos, blocks[i]->GetLocalNumOfCells());
		minPos = std::min(minPos, blocks[i]->GetLocalNumOfCells());
		numOfUnder100 += blocks[i]->GetLocalNumOfCells() < 100;
		numOfUnder200 += blocks[i]->GetLocalNumOfCells() < 200;
	}
	
	printf("Statistics\n");
	printf("The number of blocks is %d.\n", numOfBlocks);
	printf("The number of non-zero blocks is %d.\n", numOfInterestingBlocks);
	printf("The number of under-100 blocks is %d.\n", numOfUnder100);
	printf("The number of under-200 blocks is %d.\n", numOfUnder200);
	printf("The maximum number of tetrahedrons in a block is %d.\n", maxPos);
	printf("The minimum non-zero number of tetrahedrons in a block is %d.\n", minPos);
	printf("\n");
}

void StoreBlocksInDevice() {
	// Initialize start offsets in cells and points
	startOffsetInCell = new int [numOfInterestingBlocks + 1];
	startOffsetInPoint = new int [numOfInterestingBlocks + 1];
	startOffsetInCell[0] = 0;
	startOffsetInPoint[0] = 0;

	// Initialize startOffsetInCellForBig and startOffsetInPointForBig
	startOffsetInCellForBig = new int [numOfInterestingBlocks + 1];
	startOffsetInPointForBig = new int [numOfInterestingBlocks + 1];
	startOffsetInCellForBig[0] = 0;
	startOffsetInPointForBig[0] = 0;

	// Calculate start offsets
	int maxNumOfCells = 0, maxNumOfPoints = 0;
	for (int i = 0; i < numOfInterestingBlocks; i++) {
		startOffsetInCell[i + 1] = startOffsetInCell[i] + blocks[i]->GetLocalNumOfCells();
		startOffsetInPoint[i + 1] = startOffsetInPoint[i] + blocks[i]->GetLocalNumOfPoints();

		if (blocks[i]->EvaluateNumOfBytes() > configure->GetSharedMemoryKilobytes() * 1024) {
			startOffsetInCellForBig[i + 1] = startOffsetInCellForBig[i] + blocks[i]->GetLocalNumOfCells();
			startOffsetInPointForBig[i + 1] = startOffsetInPointForBig[i] + blocks[i]->GetLocalNumOfPoints();

			maxNumOfCells += blocks[i]->GetLocalNumOfCells();
			maxNumOfPoints += blocks[i]->GetLocalNumOfPoints();
		} else {
			startOffsetInCellForBig[i + 1] = startOffsetInCellForBig[i];
			startOffsetInPointForBig[i + 1] = startOffsetInPointForBig[i];
		}
	}

	printf("Total number of cells in all the blocks is %d.\n", startOffsetInCell[numOfInterestingBlocks]);
	printf("Total number of points in all the blocks is %d.\n", startOffsetInPoint[numOfInterestingBlocks]);
	printf("\n");

	//Create d_canFitInSharedMemory
	d_canFitInSharedMemory = clCreateBuffer(context, CL_MEM_READ_ONLY,
						sizeof(bool) * numOfInterestingBlocks, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device canFitInSharedMemory");

	// Create d_vertexPositionsForBig
	if (configure->UseDouble())
		d_vertexPositionsForBig = clCreateBuffer(context, CL_MEM_READ_WRITE,
							 sizeof(double) * 3 * maxNumOfPoints, NULL, &err);
	else
		d_vertexPositionsForBig = clCreateBuffer(context, CL_MEM_READ_WRITE,
							 sizeof(float) * 3 * maxNumOfPoints, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device vertexPositionsForBig");
	
	// Create d_startVelocitiesForBig
	if (configure->UseDouble())
		d_startVelocitiesForBig = clCreateBuffer(context, CL_MEM_READ_WRITE,
							 sizeof(double) * 3 * maxNumOfPoints, NULL, &err);
	else
		d_startVelocitiesForBig = clCreateBuffer(context, CL_MEM_READ_WRITE,
							 sizeof(float) * 3 * maxNumOfPoints, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device startVelocitiesForBig");

	// Create d_endVelocitiesForBig
	if (configure->UseDouble())
		d_endVelocitiesForBig = clCreateBuffer(context, CL_MEM_READ_WRITE,
						       sizeof(double) * 3 * maxNumOfPoints, NULL, &err);
	else
		d_endVelocitiesForBig = clCreateBuffer(context, CL_MEM_READ_WRITE,
						       sizeof(float) * 3 * maxNumOfPoints, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device endVelocitiesForBig");

	// Create d_startOffsetInCell
	d_startOffsetInCell = clCreateBuffer(context, CL_MEM_READ_ONLY,
					     sizeof(int) * (numOfInterestingBlocks + 1), NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device startOffsetInCell");

	// Create d_startOffsetInCellForBig
	d_startOffsetInCellForBig = clCreateBuffer(context, CL_MEM_READ_ONLY,
						   sizeof(int) * (numOfInterestingBlocks + 1), NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device startOffsetInCellForBig");

	// Create d_startOffsetInPoint
	d_startOffsetInPoint = clCreateBuffer(context, CL_MEM_READ_ONLY,
					      sizeof(int) * (numOfInterestingBlocks + 1), NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device startOffsetInPoint");

	// Create d_startOffsetInPointForBig
	d_startOffsetInPointForBig = clCreateBuffer(context, CL_MEM_READ_ONLY,
						    sizeof(int) * (numOfInterestingBlocks + 1), NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device startOffsetInPointForBig");

	// Create d_localConnectivities
	d_localConnectivities = clCreateBuffer(context, CL_MEM_READ_ONLY,
					       sizeof(int) * startOffsetInCell[numOfInterestingBlocks] * 4, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device localConnectivities");

	// Create d_localLinks
	d_localLinks = clCreateBuffer(context, CL_MEM_READ_ONLY,
				      sizeof(int) * startOffsetInCell[numOfInterestingBlocks] * 4, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device localLinks");

	// Create d_globalCellIDs
	d_globalCellIDs = clCreateBuffer(context, CL_MEM_READ_ONLY,
					 sizeof(int) * startOffsetInCell[numOfInterestingBlocks], NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device globalCellIDs");

	// Create d_globalPointIDs
	d_globalPointIDs = clCreateBuffer(context, CL_MEM_READ_ONLY,
					  sizeof(int) * startOffsetInPoint[numOfInterestingBlocks], NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device globalPointIDs");

	// Fill d_canFitInSharedMemory
	err = clEnqueueWriteBuffer(commandQueue, d_canFitInSharedMemory, CL_FALSE, 0, sizeof(bool) * numOfInterestingBlocks,
				   canFitInSharedMemory, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue write-to-device for d_canFitInSharedMemory");

	// Fill d_startOffsetInCell
	err = clEnqueueWriteBuffer(commandQueue, d_startOffsetInCell, CL_FALSE, 0, sizeof(int) * (numOfInterestingBlocks + 1),
				   startOffsetInCell, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue write-to-device for d_startOffsetInCell");

	// Fill d_startOffsetInCellForBig
	err = clEnqueueWriteBuffer(commandQueue, d_startOffsetInCellForBig, CL_FALSE, 0, sizeof(int) * (numOfInterestingBlocks + 1),
				   startOffsetInCellForBig, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue write-to-device for d_startOffsetInCellForBig");

	// Fill d_startOffsetInPoint
	err = clEnqueueWriteBuffer(commandQueue, d_startOffsetInPoint, CL_FALSE, 0, sizeof(int) * (numOfInterestingBlocks + 1),
				   startOffsetInPoint, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue write-to-device for d_startOffsetInPoint");

	// Fill d_startOffsetInPointForBig
	err = clEnqueueWriteBuffer(commandQueue, d_startOffsetInPointForBig, CL_FALSE, 0, sizeof(int) * (numOfInterestingBlocks + 1),
				   startOffsetInPointForBig, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue write-to-device for d_startOffsetInPointForBig");

	// Fill d_localConnectivities
	for (int i = 0; i < numOfInterestingBlocks; i++) {
		int length = startOffsetInCell[i + 1] - startOffsetInCell[i];

		if (!length) continue;

		int *currLocalConnectivities = blocks[i]->GetLocalConnectivities();

		// Enqueue write-to-device
		err = clEnqueueWriteBuffer(commandQueue, d_localConnectivities, CL_FALSE,
					   startOffsetInCell[i] * 4 * sizeof(int), length * 4 * sizeof(int),
					   currLocalConnectivities, 0, NULL, NULL);
		if (err) lcs::Error("Fail to enqueue write-to-device for d_localConnectivities");
	}

	// Fill d_localLinks
	for (int i = 0; i < numOfInterestingBlocks; i++) {
		int length = startOffsetInCell[i + 1] - startOffsetInCell[i];

		if (!length) continue;

		int *currLocalLinks = blocks[i]->GetLocalLinks();

		// Enqueue write-to-device
		err = clEnqueueWriteBuffer(commandQueue, d_localLinks, CL_FALSE,
					   startOffsetInCell[i] * 4 * sizeof(int), length * 4 * sizeof(int),
					   currLocalLinks, 0, NULL, NULL);
		if (err) lcs::Error("Fail to enqueue write-to-device for d_localLinks");
	}

	// Fill d_globalCellIDs
	for (int i = 0; i < numOfInterestingBlocks; i++) {
		int length = startOffsetInCell[i + 1] - startOffsetInCell[i];

		if (!length) continue;

		int *currGlobalCellIDs = blocks[i]->GetGlobalCellIDs();

		// Enqueue write-to-device
		err = clEnqueueWriteBuffer(commandQueue, d_globalCellIDs, CL_FALSE,
					   startOffsetInCell[i] * sizeof(int), length * sizeof(int),
					   currGlobalCellIDs, 0, NULL, NULL);
		if (err) lcs::Error("Fail to enqueue write-to-device for d_globalCellIDs");
	}

	// Fill d_globalPointIDs
	for (int i = 0; i < numOfInterestingBlocks; i++) {
		int length = startOffsetInPoint[i + 1] - startOffsetInPoint[i];

		if (!length) continue;

		int *currGlobalPointIDs = blocks[i]->GetGlobalPointIDs();

		// Enqueue write-to-device
		err = clEnqueueWriteBuffer(commandQueue, d_globalPointIDs, CL_FALSE,
					   startOffsetInPoint[i] * sizeof(int), length * sizeof(int),
					   currGlobalPointIDs, 0, NULL, NULL);
		if (err) lcs::Error("Fail to enqueue write-to-device for d_globalPointIDs");
	}

	clFinish(commandQueue);
}

void Division() {
	// Prepare queries
	PrepareTetrahedronBlockIntersectionQueries();

	// Launch GPU to solve queries
	LaunchGPUforIntersectionQueries();
	
	// Main process of division
	DivisionProcess();

	// Store blocks in the global memory of device
	StoreBlocksInDevice();
}

void InitialCellLocation() {
	printf("Start to use GPU to process initial cell location ...\n");
	printf("\n");

	int startTime = clock();

	double minX = configure->GetBoundingBoxMinX();
	double maxX = configure->GetBoundingBoxMaxX();
	double minY = configure->GetBoundingBoxMinY();
	double maxY = configure->GetBoundingBoxMaxY();
	double minZ = configure->GetBoundingBoxMinZ();
	double maxZ = configure->GetBoundingBoxMaxZ();

	int xRes = configure->GetBoundingBoxXRes();
	int yRes = configure->GetBoundingBoxYRes();
	int zRes = configure->GetBoundingBoxZRes();

	double dx = (maxX - minX) / xRes;
	double dy = (maxY - minY) / yRes;
	double dz = (maxZ - minZ) / zRes;

	int numOfGridPoints = (xRes + 1) * (yRes + 1) * (zRes + 1);
	initialCellLocations = new int [numOfGridPoints];
	memset(initialCellLocations, 255, sizeof(int) * numOfGridPoints);

	// Create the program
	cl_program program = CreateProgram(initialCellLocationKernel, "initial cell location");

	// Create OpenCL buffer pointing to the device cellLocations (output)
	d_cellLocations = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
					 sizeof(int) * numOfGridPoints, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device cellLocations");

	// Initialize d_cellLocations to -1 arrays
	cl_event copyHlocTDloc;
	err = clEnqueueWriteBuffer(commandQueue, d_cellLocations, CL_FALSE, 0, sizeof(int) * numOfGridPoints,
				   initialCellLocations, 0, NULL, &copyHlocTDloc);
	if (err) lcs::Error("Fail to enqueue copyHlocTDloc");

	// Create the kernel
	cl_kernel kernel = clCreateKernel(program, "InitialCellLocation", &err);
	if (err) lcs::Error("Fail to create the kernel for Initial Cell Location");

	// Get the work group size
	size_t maxWorkGroupSize;
	err = clGetKernelWorkGroupInfo(kernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSize, NULL);
	printf("maxWorkGroupSize = %d\n", maxWorkGroupSize);
	size_t workGroupSize = 1;
	for (; workGroupSize * 2 <= maxWorkGroupSize; workGroupSize <<= 1);
	printf("workGroupSize = %d\n", workGroupSize);
	printf("\n");

	// Set the argument values for the kernel
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_vertexPositions);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_tetrahedralConnectivities);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_cellLocations);
	
	cl_int cl_xRes = xRes;
	cl_int cl_yRes = yRes;
	cl_int cl_zRes = zRes;
	
	clSetKernelArg(kernel, 3, sizeof(cl_int), &cl_xRes);
	clSetKernelArg(kernel, 4, sizeof(cl_int), &cl_yRes);
	clSetKernelArg(kernel, 5, sizeof(cl_int), &cl_zRes);

	void *floatNumbers;

	if (configure->UseDouble()) {
		floatNumbers = new double [7];
		((double *)floatNumbers)[0] = minX;
		((double *)floatNumbers)[1] = minY;
		((double *)floatNumbers)[2] = minZ;
		((double *)floatNumbers)[3] = dx;
		((double *)floatNumbers)[4] = dy;
		((double *)floatNumbers)[5] = dz;
		((double *)floatNumbers)[6] = configure->GetEpsilon();
		for (int i = 0; i < 7; i++)
			clSetKernelArg(kernel, 6 + i, sizeof(cl_double), (double *)floatNumbers + i);
	} else {
		floatNumbers = new float [7];
		((float *)floatNumbers)[0] = (float)minX;
		((float *)floatNumbers)[1] = (float)minY;
		((float *)floatNumbers)[2] = (float)minZ;
		((float *)floatNumbers)[3] = (float)dx;
		((float *)floatNumbers)[4] = (float)dy;
		((float *)floatNumbers)[5] = (float)dz;
		((float *)floatNumbers)[6] = (float)configure->GetEpsilon();
		for (int i = 0; i < 7; i++)
			clSetKernelArg(kernel, 6 + i, sizeof(cl_float), (float *)floatNumbers + i);
	}

	if (configure->UseDouble())
		delete [] (double *)floatNumbers;
	else
		delete [] (float *)floatNumbers;

	cl_int cl_numOfCells = globalNumOfCells;
	clSetKernelArg(kernel, 13, sizeof(cl_int), &cl_numOfCells);

	// Set local / global work size
	size_t localWorkSize[] = {workGroupSize};
	size_t globalWorkSize[] = {((globalNumOfCells - 1) / workGroupSize + 1) * workGroupSize};

	// Enqueue the kernel event
	cl_event kernelEvent;
	err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize,
				     1, &copyHlocTDloc, &kernelEvent);

	if (err) lcs::Error("Fail to enqueue initial cell location kernel");

	// Copy from device to host
	cl_event copyDlocTHloc;
	err = clEnqueueReadBuffer(commandQueue, d_cellLocations, CL_FALSE, 0, sizeof(int) * numOfGridPoints,
				  initialCellLocations, 1, &kernelEvent, &copyDlocTHloc);

	if (err) lcs::Error("Fail to enqueue copyDlocTHloc");

	// Synchronization Point
	clFinish(commandQueue);

	// Release some resources
	clReleaseMemObject(d_cellLocations);

	clReleaseEvent(copyHlocTDloc);
	clReleaseEvent(kernelEvent);
	clReleaseEvent(copyDlocTHloc);

	/// DEBUG ///
	FILE *locationFile = fopen("lcsInitialLocations.txt", "w");
	for (int i = 0; i < numOfGridPoints; i++)
		if (initialCellLocations[i] != -1)
			fprintf(locationFile, "%d %d\n", i, initialCellLocations[i]);
	fclose(locationFile);

	int endTime = clock();

	printf("First 10 results: ");
	for (int i = 0; i < 10; i++) {
		if (i) printf(" ");
		printf("%d", initialCellLocations[i]);
	}
	printf("\n\n");

	printf("The GPU Kernel for initial cell locations cost %lf sec.\n", (endTime - startTime) * 1.0 / CLOCKS_PER_SEC);
	printf("\n");

	// Unit Test for Initial Cell Location Kernel
	startTime = clock();

	if (configure->UseUnitTestForInitialCellLocation()) {
		lcs::UnitTestForInitialCellLocations(frames[0]->GetTetrahedralGrid(),
						     xRes, yRes, zRes,
						     minX, minY, minZ,
						     dx, dy, dz,
						     initialCellLocations,
						     configure->GetEpsilon());
		printf("\n");
	}

	endTime = clock();

	printf("The unit test cost %lf sec.\n", (endTime - startTime) * 1.0 / CLOCKS_PER_SEC);
	printf("\n");
}

void InitializeParticleRecordsInDevice() {
	// Initialize stage
	int *stage = new int [numOfInitialActiveParticles];
	memset(stage, 0, sizeof(int) * numOfInitialActiveParticles);

	// Initialize pastTimes
	void *pastTimes;
	if (configure->UseDouble()) {
		pastTimes = new double [numOfInitialActiveParticles];
		memset(pastTimes, 0, sizeof(double) * numOfInitialActiveParticles);
	} else {
		pastTimes = new float [numOfInitialActiveParticles];
		memset(pastTimes, 0, sizeof(float) * numOfInitialActiveParticles);
	}

	// Initialize activeBlockOfParticles
	d_activeBlockOfParticles = clCreateBuffer(context, CL_MEM_READ_WRITE,
						  sizeof(int) * numOfInitialActiveParticles, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device activeBlockOfParticles");

	// Initialize localTetIDs
	d_localTetIDs = clCreateBuffer(context, CL_MEM_READ_WRITE,
				       sizeof(int) * numOfInitialActiveParticles, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device localTetIDs");

	// Initialize particleOrders
	d_particleOrders = clCreateBuffer(context, CL_MEM_READ_WRITE,
					  sizeof(int) * numOfInitialActiveParticles, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device particleOrders");

	// Initialize blockLocations
	d_blockLocations = clCreateBuffer(context, CL_MEM_READ_WRITE,
					  sizeof(int) * numOfInitialActiveParticles, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device blockLocations");

	// Initialize d_placesOfInterest (Another part is in lastPositions initialization)
	if (configure->UseDouble())
		d_placesOfInterest = clCreateBuffer(context, CL_MEM_READ_WRITE,
						    sizeof(double) * 3 * numOfInitialActiveParticles, NULL, &err);
	else
		d_placesOfInterest = clCreateBuffer(context, CL_MEM_READ_WRITE,
						    sizeof(float) * 3 * numOfInitialActiveParticles, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device placesOfInterest");	

	// Initialize d_activeParticles[2]
	for (int i = 0; i < 2; i++) {
		d_activeParticles[i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
						      sizeof(int) * numOfInitialActiveParticles, NULL, &err);
		if (err) lcs::Error("Fail to create a buffer for device activeParticles");
	}

	// Initialize d_exitCells
	d_exitCells = clCreateBuffer(context, CL_MEM_READ_WRITE,
				     sizeof(int) * numOfInitialActiveParticles, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device exitCells");

	err = clEnqueueWriteBuffer(commandQueue, d_exitCells, CL_FALSE, 0, sizeof(int) * numOfInitialActiveParticles,
				   exitCells, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue write-to-device for d_exitCells");

	// Initialize d_stage
	d_stages = clCreateBuffer(context, CL_MEM_READ_WRITE,
				  sizeof(int) * numOfInitialActiveParticles, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device stages");

	err = clEnqueueWriteBuffer(commandQueue, d_stages, CL_FALSE, 0, sizeof(int) * numOfInitialActiveParticles,
				   stage, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue write-to-device for d_stage");

	// Initialize d_pastTimes
	if (configure->UseDouble())
		d_pastTimes = clCreateBuffer(context, CL_MEM_READ_WRITE,
					     sizeof(double) * numOfInitialActiveParticles, NULL, &err);
	else
		d_pastTimes = clCreateBuffer(context, CL_MEM_READ_WRITE,
					     sizeof(float) * numOfInitialActiveParticles, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device pastTimes");

	if (configure->UseDouble())
		err = clEnqueueWriteBuffer(commandQueue, d_pastTimes, CL_FALSE, 0, sizeof(double) * numOfInitialActiveParticles,
					   pastTimes, 0, NULL, NULL);
	else
		err = clEnqueueWriteBuffer(commandQueue, d_pastTimes, CL_FALSE, 0, sizeof(float) * numOfInitialActiveParticles,
					   pastTimes, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue write-to-device for d_pastTimes");

	// Initialize some integration-specific device arrays
	switch (lcs::ParticleRecord::GetDataType()) {
	case lcs::ParticleRecord::RK4: {	
		// Initialize d_lastPositionForRK4
		void *lastPosition;
		if (configure->UseDouble())
			lastPosition = new double [numOfInitialActiveParticles * 3];
		else
			lastPosition = new float [numOfInitialActiveParticles * 3];
		for (int i = 0; i < numOfInitialActiveParticles; i++) {
			lcs::ParticleRecordDataForRK4 *data = (lcs::ParticleRecordDataForRK4 *)particleRecords[i]->GetData();
			lcs::Vector point = data->GetLastPosition();
			double x = point.GetX();
			double y = point.GetY();
			double z = point.GetZ();
			if (configure->UseDouble()) {
				((double *)lastPosition)[i * 3] = x;
				((double *)lastPosition)[i * 3 + 1] = y;
				((double *)lastPosition)[i * 3 + 2] = z;
			} else {
				((float *)lastPosition)[i * 3] = x;
				((float *)lastPosition)[i * 3 + 1] = y;
				((float *)lastPosition)[i * 3 + 2] = z;
			}
		}

		if (configure->UseDouble())
			d_lastPositionForRK4 = clCreateBuffer(context, CL_MEM_READ_WRITE,
							      sizeof(double) * 3 * numOfInitialActiveParticles, NULL, &err);
		else
			d_lastPositionForRK4 = clCreateBuffer(context, CL_MEM_READ_WRITE,
							      sizeof(float) * 3 * numOfInitialActiveParticles, NULL, &err);
		if (err) lcs::Error("Fail to create a buffer for device lastPosition for RK4");

		if (configure->UseDouble())
			err = clEnqueueWriteBuffer(commandQueue, d_lastPositionForRK4, CL_FALSE, 0,
						   sizeof(double) * 3 * numOfInitialActiveParticles, lastPosition, 0, NULL, NULL);
		else
			err = clEnqueueWriteBuffer(commandQueue, d_lastPositionForRK4, CL_FALSE, 0,
						   sizeof(float) * 3 * numOfInitialActiveParticles, lastPosition, 0, NULL, NULL);
		if (err) lcs::Error("Fail to enqueue write-to-device for d_lastPositionForRK4");

		// Additional work of placesOfInterest initialization
		if (configure->UseDouble())
			err = clEnqueueWriteBuffer(commandQueue, d_placesOfInterest, CL_FALSE, 0,
						   sizeof(double) * 3 * numOfInitialActiveParticles, lastPosition, 0, NULL, NULL);
		else
			err = clEnqueueWriteBuffer(commandQueue, d_placesOfInterest, CL_FALSE, 0,
						   sizeof(float) * 3 * numOfInitialActiveParticles, lastPosition, 0, NULL, NULL);
		if (err) lcs::Error("Fail to enqueue write-to-device for d_placesOfInterest");

		// Initialize d_k1ForRK4
		if (configure->UseDouble())
			d_k1ForRK4 = clCreateBuffer(context, CL_MEM_READ_WRITE,
						    sizeof(double) * 3 * numOfInitialActiveParticles, NULL, &err);
		else
			d_k1ForRK4 = clCreateBuffer(context, CL_MEM_READ_WRITE,
						    sizeof(float) * 3 * numOfInitialActiveParticles, NULL, &err);
		if (err) lcs::Error("Fail to create a buffer for device k1 for RK4");

		// Initialize d_k2ForRK4
		if (configure->UseDouble())
			d_k2ForRK4 = clCreateBuffer(context, CL_MEM_READ_WRITE,
						    sizeof(double) * 3 * numOfInitialActiveParticles, NULL, &err);
		else
			d_k2ForRK4 = clCreateBuffer(context, CL_MEM_READ_WRITE,
						    sizeof(float) * 3 * numOfInitialActiveParticles, NULL, &err);
		if (err) lcs::Error("Fail to create a buffer for device k2 for RK4");

		// Initialize d_k3ForRK4
		if (configure->UseDouble())
			d_k3ForRK4 = clCreateBuffer(context, CL_MEM_READ_WRITE,
						    sizeof(double) * 3 * numOfInitialActiveParticles, NULL, &err);
		else
			d_k3ForRK4 = clCreateBuffer(context, CL_MEM_READ_WRITE,
						    sizeof(float) * 3 * numOfInitialActiveParticles, NULL, &err);
		if (err) lcs::Error("Fail to create a buffer for device k3 for RK4");
	} break;
	}

	// Release some arrays
	delete [] stage;
	delete [] exitCells;

	if (configure->UseDouble())
		delete [] (double *)pastTimes;
	else
		delete [] (float *)pastTimes;

	clFinish(commandQueue);
}

void BigBlockInitializationForPositions() {
	// create the program
	cl_program program = CreateProgram(bigBlockInitializationForPositionsKernel, "big block initialization for positions");

	// Create the kernel
	cl_kernel kernel = clCreateKernel(program, "BigBlockInitializationForPositions", &err);
	if (err) lcs::Error("Fail to create the kernel for Big Block Initialization for Positions");

	// Get the work group size
	size_t maxWorkGroupSize;
	err = clGetKernelWorkGroupInfo(kernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSize, NULL);
	printf("maxWorkGroupSize = %d\n", maxWorkGroupSize);
	size_t workGroupSize = 1;
	for (; workGroupSize * 2 <= maxWorkGroupSize; workGroupSize <<= 1);
	printf("workGroupSize = %d\n", workGroupSize);
	printf("\n");

	// Set the argument values for the kernel
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_vertexPositions);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_globalPointIDs);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_startOffsetInPoint);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_startOffsetInPointForBig);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_vertexPositionsForBig);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_bigBlocks);
	
	// Set local / global work size
	size_t localWorkSize[] = {workGroupSize};
	size_t globalWorkSize[] = {workGroupSize * numOfBigBlocks};

	// Enqueue the kernel event
	err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue big block initialization for positions kernel");

	// Synchronization Point
	clFinish(commandQueue);
}

void BigBlockInitializationForVelocities(int currStartVIndex) {
	// create the program
	cl_program program = CreateProgram(bigBlockInitializationForVelocitiesKernel, "big block initialization for velocities");

	// Create the kernel
	cl_kernel kernel = clCreateKernel(program, "BigBlockInitializationForVelocities", &err);
	if (err) lcs::Error("Fail to create the kernel for Big Block Initialization for Velocities");

	// Get the work group size
	size_t maxWorkGroupSize;
	err = clGetKernelWorkGroupInfo(kernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSize, NULL);
	printf("maxWorkGroupSize = %d\n", maxWorkGroupSize);
	size_t workGroupSize = 1;
	for (; workGroupSize * 2 <= maxWorkGroupSize; workGroupSize <<= 1);
	printf("workGroupSize = %d\n", workGroupSize);
	printf("\n");

	// Set the argument values for the kernel
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_velocities[currStartVIndex]);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_velocities[1 - currStartVIndex]);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_globalPointIDs);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_startOffsetInPoint);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_startOffsetInPointForBig);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_startVelocitiesForBig);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_endVelocitiesForBig);
	clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_bigBlocks);
	
	// Set local / global work size
	size_t localWorkSize[] = {workGroupSize};
	size_t globalWorkSize[] = {workGroupSize * numOfBigBlocks};

	// Enqueue the kernel event
	err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue big block initialization for velocities kernel");

	// Synchronization Point
	clFinish(commandQueue);
}

/// DEBUG ///
double kernelSum;

void LaunchBlockedTracingKernel(cl_kernel kernel, size_t workGroupSize, int numOfEvents, cl_event *events,
				int currStartVIndex, int numOfWorkGroups, double beginTime, double finishTime) {
	int starTime;

	printf("Start to use GPU to process blocked tracing ...\n");
	printf("\n");

	int startTime = clock();

	// Set the argument values for the kernel
	switch (lcs::ParticleRecord::GetDataType()) {
	case lcs::ParticleRecord::RK4: {
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_velocities[currStartVIndex]);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_velocities[1 - currStartVIndex]);

		if (configure->UseDouble()) {
			cl_double d_startTime = beginTime;
			cl_double d_endTime = finishTime;
			
			clSetKernelArg(kernel, 32, sizeof(cl_double), &d_startTime);
			clSetKernelArg(kernel, 33, sizeof(cl_double), &d_endTime);
		} else {
			cl_float d_startTime = beginTime;
			cl_float d_endTime = finishTime;

			clSetKernelArg(kernel, 32, sizeof(cl_float), &d_startTime);
			clSetKernelArg(kernel, 33, sizeof(cl_float), &d_endTime);
		}
	} break;
	}

	// Set local / global work size
	size_t localWorkSize[] = {workGroupSize};
	size_t globalWorkSize[] = {numOfWorkGroups * workGroupSize};

	/// DEBUG ///
	printf("workGroupSize = %d\n", workGroupSize);
	printf("numOfWorkGroups = %d\n", numOfWorkGroups);

	// Enqueue the kernel event
	cl_event kernelEvent;
	err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize,
				     numOfEvents, events, &kernelEvent);

	/// DEBUG ///
	printf("err = %d\n", err);

	if (err) lcs::Error("Fail to enqueue blocked tracing kernel");

	/// DEBUG ///
	err = clFinish(commandQueue);
	printf("blocked tracing kernel clFinish = %d\n", err);

	// Release some resources
	clReleaseEvent(kernelEvent);

	int endTime = clock();

	/// DEBUG ///
	kernelSum += (double)(endTime - startTime) / CLOCKS_PER_SEC;

	printf("The GPU Kernel for blocked tracing cost %lf sec.\n", (endTime - startTime) * 1.0 / CLOCKS_PER_SEC);
	printf("\n");
}

void InitializeInitialActiveParticles() {
	// Initialize particleRecord
	double minX = configure->GetBoundingBoxMinX();
	double maxX = configure->GetBoundingBoxMaxX();
	double minY = configure->GetBoundingBoxMinY();
	double maxY = configure->GetBoundingBoxMaxY();
	double minZ = configure->GetBoundingBoxMinZ();
	double maxZ = configure->GetBoundingBoxMaxZ();

	int xRes = configure->GetBoundingBoxXRes();
	int yRes = configure->GetBoundingBoxYRes();
	int zRes = configure->GetBoundingBoxZRes();

	double dx = (maxX - minX) / xRes;
	double dy = (maxY - minY) / yRes;
	double dz = (maxZ - minZ) / zRes;

	int numOfGridPoints = (xRes + 1) * (yRes + 1) * (zRes + 1);

	// Get numOfInitialActiveParticles
	numOfInitialActiveParticles = 0;
	for (int i = 0; i < numOfGridPoints; i++)
		if (initialCellLocations[i] != -1) numOfInitialActiveParticles++;

	if (!numOfInitialActiveParticles)
		lcs::Error("There is no initial active particle for tracing.");

	// Initialize particleRecords
	particleRecords = new lcs::ParticleRecord * [numOfInitialActiveParticles];

	int idx = -1, activeIdx = -1;
	for (int i = 0; i <= xRes; i++)
		for (int j = 0; j <= yRes; j++)
			for (int k = 0; k <= zRes; k++) {
				idx++;

				if (initialCellLocations[idx] == -1) continue;

				activeIdx++;

				switch (lcs::ParticleRecord::GetDataType()) {
				case lcs::ParticleRecord::RK4: {
					lcs::ParticleRecordDataForRK4 *data = new lcs::ParticleRecordDataForRK4();
					data->SetLastPosition(lcs::Vector(minX + i * dx, minY + j * dy, minZ + k * dz));
					particleRecords[activeIdx] = new
								     lcs::ParticleRecord(lcs::ParticleRecordDataForRK4::COMPUTING_K1,
								     idx, data);
				} break;
				}
			}

	// Initialize exitCells
	exitCells = new int [numOfInitialActiveParticles];
	for (int i = 0; i < numOfInitialActiveParticles; i++)
		exitCells[i] = initialCellLocations[particleRecords[i]->GetGridPointID()];

	// Initialize particle records in device
	InitializeParticleRecordsInDevice();
}

void InitializeVelocityData(void **velocities) {
	// Initialize velocity data
	for (int i = 0; i < 2; i++)
		if (configure->UseDouble())
			velocities[i] = new double [globalNumOfPoints * 3];
		else
			velocities[i] = new float [globalNumOfPoints * 3];

	// Read velocities[0]
	if (configure->UseDouble())
		frames[0]->GetTetrahedralGrid()->ReadVelocities((double *)velocities[0]);
	else
		frames[0]->GetTetrahedralGrid()->ReadVelocities((float *)velocities[0]);

	// Create d_velocities[2]
	for (int i = 0; i < 2; i++) {
		if (configure->UseDouble())
			d_velocities[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
							 sizeof(double) * 3 * globalNumOfPoints, NULL, &err);
		else
			d_velocities[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
							 sizeof(float) * 3 * globalNumOfPoints, NULL, &err);
		if (err) lcs::Error("Fail to create buffers for d_velocities[2]");
	}

	// Initialize d_velocities[0]
	if (configure->UseDouble())
		err = clEnqueueWriteBuffer(commandQueue, d_velocities[0], CL_TRUE, 0, sizeof(double) * 3 * globalNumOfPoints,
					   velocities[0], 0, NULL, NULL);
	else
		err = clEnqueueWriteBuffer(commandQueue, d_velocities[0], CL_TRUE, 0, sizeof(float) * 3 * globalNumOfPoints,
					   velocities[0], 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue copy for d_velocities[0]");
}

cl_event LoadVelocities(void *velocities, cl_mem d_velocities, int frameIdx) {
	// Read velocities
	if (configure->UseDouble())
		frames[frameIdx]->GetTetrahedralGrid()->ReadVelocities((double *)velocities);
	else
		frames[frameIdx]->GetTetrahedralGrid()->ReadVelocities((float *)velocities);

	// Enqueue write for d_velocities[frameIdx]
	cl_event writeEvent;
	if (configure->UseDouble())
		err = clEnqueueWriteBuffer(commandQueue, d_velocities, CL_FALSE, 0, sizeof(double) * 3 * globalNumOfPoints,
					   velocities, 0, NULL, &writeEvent);
	else
		err = clEnqueueWriteBuffer(commandQueue, d_velocities, CL_FALSE, 0, sizeof(float) * 3 * globalNumOfPoints,
					   velocities, 0, NULL, &writeEvent);
	if (err) lcs::Error("Fail to enqueue copy for d_velocities");
	return writeEvent;
}

void InitializeExclusiveScanKernel(cl_program &scanProgram, cl_kernel &scanKernel, cl_kernel &reverseUpdateKernel,
				   int &numOfBanks, int &maxArrSize, int &workGroupSize) {
	scanProgram = CreateProgram(exclusiveScanForIntKernels, "exclusive scan");

	scanKernel = clCreateKernel(scanProgram, "Scan", &err);
	if (err) lcs::Error("Fail to create the kernel for Scan");

	reverseUpdateKernel = clCreateKernel(scanProgram, "ReverseUpdate", &err);
	if (err) lcs::Error("Fail to create the kernel for reverse update");

	numOfBanks = configure->GetNumOfBanks();
	maxArrSize = std::max(numOfInterestingBlocks, numOfInitialActiveParticles);

	size_t maxWorkGroupSizeForScan;
	err = clGetKernelWorkGroupInfo(scanKernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSizeForScan, NULL);
	if (err) lcs::Error("Fail to get the maximum work group size for scan kernel");

	size_t maxWorkGroupSizeForReverseUpdate;
	err = clGetKernelWorkGroupInfo(reverseUpdateKernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSizeForReverseUpdate, NULL);
	if (err) lcs::Error("Fail to get the maximum work group size for reverse update kernel");

	int upperBound = std::min(maxWorkGroupSizeForScan, maxWorkGroupSizeForReverseUpdate);

	workGroupSize = 1;
	for (; workGroupSize * 2 <= upperBound; workGroupSize <<= 1);

	d_exclusiveScanArrayForInt = clCreateBuffer(context, CL_MEM_READ_WRITE,
						    sizeof(int) * maxArrSize, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device exclusive scan array for int");
}

void InitializeCollectActiveParticlesForNewIntervalKernel(cl_program &collect1Program, cl_kernel &collect1InitKernel,
							  cl_kernel &collect1PickKernel, int &workGroupSize) {
	collect1Program = CreateProgram(collectActiveParticlesForNewIntervalKernels, "1st collect scan");

	collect1InitKernel = clCreateKernel(collect1Program, "InitializeScanArray", &err);
	if (err) lcs::Error("Fail to create the kernel for collect 1 init kernel");

	collect1PickKernel = clCreateKernel(collect1Program, "CollectActiveParticles", &err);
	if (err) lcs::Error("Fail to create the kernel for collect 1 pick kernel");

	size_t maxWorkGroupSize1;
	err = clGetKernelWorkGroupInfo(collect1InitKernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSize1, NULL);
	if (err) lcs::Error("Fail to get the maximum work group size for collect 1 init kernel");

	size_t maxWorkGroupSize2;
	err = clGetKernelWorkGroupInfo(collect1PickKernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSize2, NULL);
	if (err) lcs::Error("Fail to get the maximum work group size for collect 1 pick kernel");

	int upperBound = std::min(maxWorkGroupSize1, maxWorkGroupSize2);

	workGroupSize = 1;
	for (; workGroupSize * 2 <= upperBound; workGroupSize <<= 1);
}

void InitializeCollectActiveParticlesForNewRunKernel(cl_program &collect2Program, cl_kernel &collect2InitKernel,
						     cl_kernel &collect2PickKernel, int &workGroupSize) {
	collect2Program = CreateProgram(collectActiveParticlesForNewRunKernels, "2nd collect scan");

	collect2InitKernel = clCreateKernel(collect2Program, "InitializeScanArray", &err);
	if (err) lcs::Error("Fail to create the kernel for collect 2 init kernel");

	collect2PickKernel = clCreateKernel(collect2Program, "CollectActiveParticles", &err);
	if (err) lcs::Error("Fail to create the kernel for collect 2 pick kernel");

	size_t maxWorkGroupSize1;
	err = clGetKernelWorkGroupInfo(collect2InitKernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSize1, NULL);
	if (err) lcs::Error("Fail to get the maximum work group size for collect 2 init kernel");

	size_t maxWorkGroupSize2;
	err = clGetKernelWorkGroupInfo(collect2PickKernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSize2, NULL);
	if (err) lcs::Error("Fail to get the maximum work group size for collect 2 pick kernel");

	int upperBound = std::min(maxWorkGroupSize1, maxWorkGroupSize2);

	workGroupSize = 1;
	for (; workGroupSize * 2 <= upperBound; workGroupSize <<= 1);
}

int CollectActiveParticlesForNewInterval(cl_kernel collect1InitKernel, cl_kernel collect1PickKernel, int collect1WorkGroupSize,
					 cl_kernel scanKernel, cl_kernel reverseUpdateKernel, int scanWorkGroupSize,
					 int numOfBanks, cl_mem d_activeParticles) {
	// Prepare for exclusive scan
	clSetKernelArg(collect1InitKernel, 0, sizeof(cl_mem), &d_exitCells);
	clSetKernelArg(collect1InitKernel, 1, sizeof(cl_mem), &d_exclusiveScanArrayForInt);
	cl_int length = numOfInitialActiveParticles;
	clSetKernelArg(collect1InitKernel, 2, sizeof(cl_int), &length);

	size_t localWorkSize = collect1WorkGroupSize;
	size_t globalWorkSize = ((length - 1) / localWorkSize + 1) * localWorkSize;

	err = clEnqueueNDRangeKernel(commandQueue, collect1InitKernel, 1, NULL,
				     &globalWorkSize, &localWorkSize, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue collect1InitKernel");
	
	clFinish(commandQueue);

	// Launch exclusive scan
	int sum;
	sum = lcs::GPUExclusiveScanForInt(scanWorkGroupSize, numOfBanks, scanKernel, reverseUpdateKernel,
					  d_exclusiveScanArrayForInt, numOfInitialActiveParticles, commandQueue);

	// Compaction
	clSetKernelArg(collect1PickKernel, 0, sizeof(cl_mem), &d_exitCells);
	clSetKernelArg(collect1PickKernel, 1, sizeof(cl_mem), &d_exclusiveScanArrayForInt);
	clSetKernelArg(collect1PickKernel, 2, sizeof(cl_mem), &d_activeParticles);
	clSetKernelArg(collect1PickKernel, 3, sizeof(cl_int), &length);

	err = clEnqueueNDRangeKernel(commandQueue, collect1PickKernel, 1, NULL,
				     &globalWorkSize, &localWorkSize, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue collect1PickKernel");

	clFinish(commandQueue);

	// Return number of active particles
	return sum;
}

int CollectActiveParticlesForNewRun(cl_kernel collect2InitKernel, cl_kernel collect2PickKernel, int collect2WorkGroupSize,
				    cl_kernel scanKernel, cl_kernel reverseUpdateKernel, int scanWorkGroupSize,
				    int numOfBanks, cl_mem d_oldActiveParticles, cl_mem d_newActiveParticles, cl_int length) {
	// Prepare for exclusive scan
	clSetKernelArg(collect2InitKernel, 0, sizeof(cl_mem), &d_exitCells);
	clSetKernelArg(collect2InitKernel, 1, sizeof(cl_mem), &d_oldActiveParticles);
	clSetKernelArg(collect2InitKernel, 2, sizeof(cl_mem), &d_exclusiveScanArrayForInt);
	clSetKernelArg(collect2InitKernel, 3, sizeof(cl_int), &length);

	size_t localWorkSize = collect2WorkGroupSize;
	size_t globalWorkSize = ((length - 1) / localWorkSize + 1) * localWorkSize;

	err = clEnqueueNDRangeKernel(commandQueue, collect2InitKernel, 1, NULL,
				     &globalWorkSize, &localWorkSize, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue collect2InitKernel");
	
	clFinish(commandQueue);

	// Launch exclusive scan
	int sum;
	sum = lcs::GPUExclusiveScanForInt(scanWorkGroupSize, numOfBanks, scanKernel, reverseUpdateKernel,
					  d_exclusiveScanArrayForInt, length, commandQueue);

	/// DEBUG ///
	printf("CollectActiveParticlesForNewRun(): length = %d, sum = %d\n", length, sum);

	// Compaction
	clSetKernelArg(collect2PickKernel, 0, sizeof(cl_mem), &d_exitCells);
	clSetKernelArg(collect2PickKernel, 1, sizeof(cl_mem), &d_oldActiveParticles);
	clSetKernelArg(collect2PickKernel, 2, sizeof(cl_mem), &d_exclusiveScanArrayForInt);
	clSetKernelArg(collect2PickKernel, 3, sizeof(cl_mem), &d_newActiveParticles);
	clSetKernelArg(collect2PickKernel, 4, sizeof(cl_int), &length);

	err = clEnqueueNDRangeKernel(commandQueue, collect2PickKernel, 1, NULL,
				     &globalWorkSize, &localWorkSize, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue collect2PickKernel");

	clFinish(commandQueue);

	// Return number of active particles
	return sum;
}

void InitializeInterestingBlockMarks() {
	int *marks = new int [numOfInterestingBlocks];
	memset(marks, 255, sizeof(int) * numOfInterestingBlocks);

	d_interestingBlockMarks = clCreateBuffer(context, CL_MEM_READ_WRITE,
						 sizeof(int) * numOfInterestingBlocks, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device interestingBlockMarks");

	err = clEnqueueWriteBuffer(commandQueue, d_interestingBlockMarks, CL_TRUE, 0,
				   sizeof(int) * numOfInterestingBlocks, marks, 0, NULL, NULL);

	delete [] marks;
}

void InitializeRedistributeParticlesKernel(cl_program &redistributeProgram,
					   cl_kernel &collectBlocksKernel, cl_kernel &collectParticlesKernel,
					   cl_kernel &statisticsKernel,
					   int &collectBlocksWorkGroupSize, int &collectParticlesWorkGroupSize,
					   int &statisticsWorkGroupSize,
					   cl_int maxNumOfStages, double epsilon) {
	redistributeProgram = CreateProgram(redistributeParticlesKernels, "redistribute");

	collectBlocksKernel = clCreateKernel(redistributeProgram, "CollectActiveBlocks", &err);
	if (err) lcs::Error("Fail to create the kernel for collect active blocks kernel");

	collectParticlesKernel = clCreateKernel(redistributeProgram, "CollectParticlesToBlocks", &err);
	if (err) lcs::Error("Fail to create the kernel for collect particles kernel");

	statisticsKernel = clCreateKernel(redistributeProgram, "GetNumOfParticlesByStageInBlocks", &err);
	if (err) lcs::Error("Fail to create the kernel for get number of particles by stage in blocks");

	size_t maxWorkGroupSize1;
	err = clGetKernelWorkGroupInfo(collectBlocksKernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSize1, NULL);
	if (err) lcs::Error("Fail to get the maximum work group size for collect active blocks kernel");

	collectBlocksWorkGroupSize = maxWorkGroupSize1;

	size_t maxWorkGroupSize2;
	err = clGetKernelWorkGroupInfo(collectParticlesKernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSize2, NULL);
	if (err) lcs::Error("Fail to get the maximum work group size for collect particles kernel");

	collectParticlesWorkGroupSize = maxWorkGroupSize2;

	size_t maxWorkGroupSize3;
	err = clGetKernelWorkGroupInfo(statisticsKernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSize3, NULL);
	if (err) lcs::Error("Fail to get the maximum work group size for get the number of particles by stage in blocks");

	statisticsWorkGroupSize = maxWorkGroupSize3;

	// Set collectBlocksKernel parameters
	//clSetKernelArg(collectBlocksKernel, 0, sizeof(cl_mem), &d_activeParticles);
	clSetKernelArg(collectBlocksKernel, 1, sizeof(cl_mem), &d_exitCells);
	clSetKernelArg(collectBlocksKernel, 2, sizeof(cl_mem), &d_placesOfInterest);
	clSetKernelArg(collectBlocksKernel, 3, sizeof(cl_mem), &d_localTetIDs);
	clSetKernelArg(collectBlocksKernel, 4, sizeof(cl_mem), &d_blockLocations);
	clSetKernelArg(collectBlocksKernel, 5, sizeof(cl_mem), &d_interestingBlockMap);
	clSetKernelArg(collectBlocksKernel, 6, sizeof(cl_mem), &d_startOffsetsInLocalIDMap);
	clSetKernelArg(collectBlocksKernel, 7, sizeof(cl_mem), &d_blocksOfTets);
	clSetKernelArg(collectBlocksKernel, 8, sizeof(cl_mem), &d_localIDsOfTets);
	clSetKernelArg(collectBlocksKernel, 9, sizeof(cl_mem), &d_interestingBlockMarks);
	clSetKernelArg(collectBlocksKernel, 10, sizeof(cl_mem), &d_activeBlocks);
	clSetKernelArg(collectBlocksKernel, 11, sizeof(cl_mem), &d_activeBlockIndices);
	clSetKernelArg(collectBlocksKernel, 12, sizeof(cl_mem), &d_numOfActiveBlocks);
	//clSetKernelArg(collectBlocksKernel, 13, sizeof(cl_mem), &d_activeBlockOfParticles);

	cl_int numInX = numOfBlocksInX, numInY = numOfBlocksInY, numInZ = numOfBlocksInZ;
	clSetKernelArg(collectBlocksKernel, 15, sizeof(cl_int), &numInX);
	clSetKernelArg(collectBlocksKernel, 16, sizeof(cl_int), &numInY);
	clSetKernelArg(collectBlocksKernel, 17, sizeof(cl_int), &numInZ);

	if (configure->UseDouble()) {
		clSetKernelArg(collectBlocksKernel, 18, sizeof(cl_double), &globalMinX);
		clSetKernelArg(collectBlocksKernel, 19, sizeof(cl_double), &globalMinY);
		clSetKernelArg(collectBlocksKernel, 20, sizeof(cl_double), &globalMinZ);
		clSetKernelArg(collectBlocksKernel, 21, sizeof(cl_double), &blockSize);
		clSetKernelArg(collectBlocksKernel, 22, sizeof(cl_double), &epsilon);
	} else {
		cl_float f_globalMinX = globalMinX;
		cl_float f_globalMinY = globalMinY;
		cl_float f_globalMinZ = globalMinZ;
		cl_float f_blockSize = blockSize;
		cl_float f_epsilon = epsilon;
		clSetKernelArg(collectBlocksKernel, 18, sizeof(cl_float), &f_globalMinX);
		clSetKernelArg(collectBlocksKernel, 19, sizeof(cl_float), &f_globalMinY);
		clSetKernelArg(collectBlocksKernel, 20, sizeof(cl_float), &f_globalMinZ);
		clSetKernelArg(collectBlocksKernel, 21, sizeof(cl_float), &f_blockSize);
		clSetKernelArg(collectBlocksKernel, 22, sizeof(cl_float), &f_epsilon);
	}

	// Set collectParticlesKernel parameters
	clSetKernelArg(collectParticlesKernel, 0, sizeof(cl_mem), &d_numOfParticlesByStageInBlocks);
	clSetKernelArg(collectParticlesKernel, 1, sizeof(cl_mem), &d_particleOrders);
	clSetKernelArg(collectParticlesKernel, 2, sizeof(cl_mem), &d_stages);
	//clSetKernelArg(collectParticlesKernel, 3, sizeof(cl_mem), &d_activeParticles);
	clSetKernelArg(collectParticlesKernel, 4, sizeof(cl_mem), &d_blockLocations);
	clSetKernelArg(collectParticlesKernel, 5, sizeof(cl_mem), &d_activeBlockIndices);
	clSetKernelArg(collectParticlesKernel, 6, sizeof(cl_mem), &d_blockedActiveParticles);
	clSetKernelArg(collectParticlesKernel, 7, sizeof(cl_int), &maxNumOfStages);

	// Set statisticsKernel parameters
	clSetKernelArg(statisticsKernel, 0, sizeof(cl_mem), &d_numOfParticlesByStageInBlocks);
	clSetKernelArg(statisticsKernel, 1, sizeof(cl_mem), &d_particleOrders);
	clSetKernelArg(statisticsKernel, 2, sizeof(cl_mem), &d_stages);
	//clSetKernelArg(statisticsKernel, 3, sizeof(cl_mem), &d_activeParticles);
	//clSetKernelArg(statisticsKernel, 4, sizeof(cl_mem), &d_activeBlockOfParticles);
	clSetKernelArg(statisticsKernel, 4, sizeof(cl_mem), &d_blockLocations);
	clSetKernelArg(statisticsKernel, 5, sizeof(cl_mem), &d_activeBlockIndices);
	clSetKernelArg(statisticsKernel, 6, sizeof(cl_int), &maxNumOfStages);
	//clSetKernelArg(statisticsKernel, 6, sizeof(cl_mem), &numOfActiveParticles);
}

int RedistributeParticles(cl_kernel collectBlocksKernel, cl_kernel collectParticlesKernel, cl_kernel statisticsKernel,
			  size_t workGroupSize1, size_t workGroupSize2, size_t workGroupSize3,
			  cl_mem d_activeParticles, cl_int numOfActiveParticles, cl_int iBMCount,
			  int numOfStages, cl_kernel scanKernel, cl_kernel reverseUpdateKernel, int scanWorkGroupSize,
			  int numOfBanks) {
	/// DEBUG ///
	err = clFinish(commandQueue);
	printf("Before collect blocks Kernel, err = %d\n", err);

	/// DEBUG ///
	//lcs::CheckFloatArrayInDevice("placesOfInterest.txt", commandQueue, d_placesOfInterest, numOfInitialActiveParticles * 3);
	printf("iBMCount = %d\n", iBMCount);

	// Intialize d_numOfActiveBlocks
	int zero = 0;

	err = clEnqueueWriteBuffer(commandQueue, d_numOfActiveBlocks, CL_TRUE, 0, sizeof(int),
				   &zero, 0, NULL, NULL);
	if (err) lcs::Error("Fail to write d_numOfActiveBlocks");

	// Launch collectActiveBlocksKernel
	clSetKernelArg(collectBlocksKernel, 0, sizeof(cl_mem), &d_activeParticles);
	clSetKernelArg(collectBlocksKernel, 13, sizeof(cl_int), &iBMCount);
	clSetKernelArg(collectBlocksKernel, 14, sizeof(cl_int), &numOfActiveParticles);

	size_t globalWorkSize = ((numOfActiveParticles - 1) / workGroupSize1 + 1) * workGroupSize1;

	err = clEnqueueNDRangeKernel(commandQueue, collectBlocksKernel, 1, NULL,
				     &globalWorkSize, &workGroupSize1, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue collectBlocksKernel");

	/// DEBUG ///
	err = clFinish(commandQueue);
	printf("err = %d\n", err);

	if (err) lcs::Error("non-zero err value");

	// Get the number of active blocks
	int numOfActiveBlocks;

	err = clEnqueueReadBuffer(commandQueue, d_numOfActiveBlocks, CL_TRUE, 0, sizeof(int),
				  &numOfActiveBlocks, 0, NULL, NULL);
	if (err) lcs::Error("Fail to read d_numOfActiveBlocks");

	/// DEBUG ///
	printf("numOfActiveBlocks = %d\n", numOfActiveBlocks);

	/// DEBUG ///
	//lcs::CheckIntArrayInDevice("blockLocations.txt", commandQueue, d_blockLocations, numOfInitialActiveParticles);

	// Get the number of particles by stage in blocks
	static int *zeroArray = NULL;
	if (!zeroArray) {
		zeroArray = new int [numOfInterestingBlocks * numOfStages];
		memset(zeroArray, 0, sizeof(int) * numOfInterestingBlocks * numOfStages);
	}

	err = clEnqueueWriteBuffer(commandQueue, d_numOfParticlesByStageInBlocks, CL_TRUE, 0,
				   numOfActiveBlocks * numOfStages * sizeof(int), zeroArray, 0, NULL, NULL);
	if (err) lcs::Error("Fail to write d_numOfParticlesByStageInBlocks");

	clSetKernelArg(statisticsKernel, 3, sizeof(cl_mem), &d_activeParticles);
	clSetKernelArg(statisticsKernel, 7, sizeof(cl_int), &numOfActiveParticles);

	globalWorkSize = ((numOfActiveParticles - 1) / workGroupSize3 + 1) * workGroupSize3;

	err = clEnqueueNDRangeKernel(commandQueue, statisticsKernel, 1, NULL,
				     &globalWorkSize, &workGroupSize3, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue statisticsKernel");

	clFinish(commandQueue);

	// Prefix scan for d_numOfParticlesByStageInBlocks
	int sum;
	sum = lcs::GPUExclusiveScanForInt(scanWorkGroupSize, numOfBanks, scanKernel, reverseUpdateKernel,
					  d_numOfParticlesByStageInBlocks, numOfActiveBlocks * numOfStages, commandQueue);

	/// DEBUG ///
	printf("sum = %d\n", sum);

	// Collect particles to blocks
	clSetKernelArg(collectParticlesKernel, 3, sizeof(cl_mem), &d_activeParticles);
	clSetKernelArg(collectParticlesKernel, 8, sizeof(cl_int), &numOfActiveParticles);

	globalWorkSize = ((numOfActiveParticles - 1) / workGroupSize2 + 1) * workGroupSize2;

	err = clEnqueueNDRangeKernel(commandQueue, collectParticlesKernel, 1, NULL,
				     &globalWorkSize, &workGroupSize2, 0, NULL, NULL);

	/// DEBUG ///
	printf("err = %d\n", err);
	if (err) lcs::Error("Fail to enqueue collectParticlesKernel");

	/// DEBUG ///
	err = clFinish(commandQueue);
	if (err) lcs::Error("collectParticleKernel execution error");

	// return
	return numOfActiveBlocks;
}

void InitializeCollectEveryKElementKernel(cl_program &everyKElementProgram, cl_kernel &everyKElementKernel,
					  int &everyKElementWorkGroupSize, cl_int maxNumOfStages) {
	everyKElementProgram = CreateProgram(collectEveryKElementKernel, "collect every k element");

	everyKElementKernel = clCreateKernel(everyKElementProgram, "CollectEveryKElement", &err);
	if (err) lcs::Error("Fail to create the kernel for collect every k element kernel");

	size_t maxWorkGroupSize;
	err = clGetKernelWorkGroupInfo(everyKElementKernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSize, NULL);
	if (err) lcs::Error("Fail to get the maximum work group size for collect every k element kernel");

	everyKElementWorkGroupSize = maxWorkGroupSize;

	clSetKernelArg(everyKElementKernel, 0, sizeof(cl_mem), &d_numOfParticlesByStageInBlocks);
	clSetKernelArg(everyKElementKernel, 1, sizeof(cl_mem), &d_startOffsetInParticles);
	clSetKernelArg(everyKElementKernel, 2, sizeof(cl_int), &maxNumOfStages);
}

void GetStartOffsetInParticles(cl_kernel everyKElementKernel, cl_int numOfActiveBlocks, size_t workGroupSize,
			       int numOfActiveParticles) {
	/// DEBUG ///
	printf("In GetStartOffsetInParticles()\n");
	printf("numOfActiveBlocks = %d\n", numOfActiveBlocks);
	printf("numOfActiveParticles = %d\n", numOfActiveParticles);


	err = clEnqueueWriteBuffer(commandQueue, d_startOffsetInParticles, CL_TRUE, sizeof(int) * numOfActiveBlocks, sizeof(int),
				   &numOfActiveParticles, 0, NULL, NULL);
	if (err) lcs::Error("Fail to write to d_startOffsetInParticles");

	clSetKernelArg(everyKElementKernel, 3, sizeof(cl_int), &numOfActiveBlocks);

	size_t globalWorkSize = ((numOfActiveBlocks - 1) / workGroupSize + 1) * workGroupSize;

	err = clEnqueueNDRangeKernel(commandQueue, everyKElementKernel, 1, NULL,
				     &globalWorkSize, &workGroupSize, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue everyKElementKernel");

	clFinish(commandQueue);
}

int AssignWorkGroups(cl_kernel getNumKernel, cl_kernel assignKernel, 
		      size_t workGroupSize1, size_t workGroupSize2, cl_int numOfActiveBlocks,
		      int scanWorkGroupSize, int numOfBanks, cl_kernel scanKernel, cl_kernel reverseUpdateKernel) {
	// Get numOfGroupsForBlocks
	clSetKernelArg(getNumKernel, 2, sizeof(cl_int), &numOfActiveBlocks);

	size_t globalWorkSize = ((numOfActiveBlocks - 1) / workGroupSize1 + 1) * workGroupSize1;

	err = clEnqueueNDRangeKernel(commandQueue, getNumKernel, 1, NULL, &globalWorkSize, &workGroupSize1, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue getNumKernel");

	clFinish(commandQueue);

	// Exclusive scan of numOfGroupsForBlocks
	int sum;
	sum = lcs::GPUExclusiveScanForInt(scanWorkGroupSize, numOfBanks, scanKernel, reverseUpdateKernel,
					  d_numOfGroupsForBlocks, numOfActiveBlocks, commandQueue);

	// Fill in the sum
	err = clEnqueueWriteBuffer(commandQueue, d_numOfGroupsForBlocks, CL_TRUE,
				   sizeof(int) * numOfActiveBlocks, sizeof(int), &sum, 0, NULL, NULL);
	if (err) lcs::Error("Fail to write to d_numOfGroupsForBlocks");

	// Assign groups
	globalWorkSize = numOfActiveBlocks * workGroupSize2;

	err = clEnqueueNDRangeKernel(commandQueue, assignKernel, 1, NULL, &globalWorkSize, &workGroupSize2, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue assignKernel");

	/// DEBUG ///
	err = clFinish(commandQueue);
	if (err) lcs::Error("Get non-zero er in AssignWorkGroups()");

	return sum;
}

void InitializeAssignGroupsKernel(cl_program &assignGroupsProgram, cl_kernel &getNumKernel, cl_kernel &assignKernel,
				  int &getNumWorkGroupSize, int &assignWorkGroupSize, cl_int tracingWorkGroupSize) {
	assignGroupsProgram = CreateProgram(assignWorkGroupsKernels, "assign work groups");

	getNumKernel = clCreateKernel(assignGroupsProgram, "GetNumOfGroupsForBlocks", &err);
	if (err) lcs::Error("Fail to create the kernel for get number of groups for blocks kernel");

	assignKernel = clCreateKernel(assignGroupsProgram, "AssignGroups", &err);
	if (err) lcs::Error("Fail to create the kernel for assign groups kernel");

	size_t maxWorkGroupSize1;
	err = clGetKernelWorkGroupInfo(getNumKernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSize1, NULL);
	if (err) lcs::Error("Fail to get the maximum work group size for get number of groups for blocks kernel");

	getNumWorkGroupSize = maxWorkGroupSize1;

	size_t maxWorkGroupSize2;
	err = clGetKernelWorkGroupInfo(assignKernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSize2, NULL);
	if (err) lcs::Error("Fail to get the maximum work group size for assign groups kernel");

	assignWorkGroupSize = maxWorkGroupSize2;

	// Set getNumKernel parameters
	clSetKernelArg(getNumKernel, 0, sizeof(cl_mem), &d_startOffsetInParticles);
	clSetKernelArg(getNumKernel, 1, sizeof(cl_mem), &d_numOfGroupsForBlocks);
	
	clSetKernelArg(getNumKernel, 3, sizeof(cl_int), &tracingWorkGroupSize);

	// Set assignKernel parameters
	clSetKernelArg(assignKernel, 0, sizeof(cl_mem), &d_numOfGroupsForBlocks);
	clSetKernelArg(assignKernel, 1, sizeof(cl_mem), &d_blockOfGroups);
	clSetKernelArg(assignKernel, 2, sizeof(cl_mem), &d_offsetInBlocks);
}

void InitializeTracingKernel(cl_program &tracingProgram, cl_kernel &tracingKernel, int &workGroupSize, double epsilon) {
	static char kernelName[100];
	switch (lcs::ParticleRecord::GetDataType()) {
	case lcs::ParticleRecord::RK4: {
		sprintf(kernelName, "%sRK4%s", blockedTracingKernelPrefix, blockedTracingKernelSuffix);
	} break;
	}

	tracingProgram = CreateProgram(kernelName, "blocked tracing");

	tracingKernel = clCreateKernel(tracingProgram, "BlockedTracing", &err);
	if (err) lcs::Error("Fail to create the kernel for tracing");

	size_t maxWorkGroupSize;
	err = clGetKernelWorkGroupInfo(tracingKernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
				       sizeof(size_t), &maxWorkGroupSize, NULL);
	if (err) lcs::Error("Fail to get the maximum work group size for get number of groups for blocks kernel");

	workGroupSize = maxWorkGroupSize;

	clSetKernelArg(tracingKernel, 0, sizeof(cl_mem), &d_vertexPositions);
	//clSetKernelArg(tracingKernel, 1, sizeof(cl_mem), &d_startVelocities);
	//clSetKernelArg(tracingKernel, 2, sizeof(cl_mem), &d_endVelocities);
	clSetKernelArg(tracingKernel, 3, sizeof(cl_mem), &d_tetrahedralConnectivities);
	clSetKernelArg(tracingKernel, 4, sizeof(cl_mem), &d_tetrahedralLinks);

	clSetKernelArg(tracingKernel, 5, sizeof(cl_mem), &d_startOffsetInCell);
	clSetKernelArg(tracingKernel, 6, sizeof(cl_mem), &d_startOffsetInPoint);

	clSetKernelArg(tracingKernel, 7, sizeof(cl_mem), &d_startOffsetInCellForBig);
	clSetKernelArg(tracingKernel, 8, sizeof(cl_mem), &d_startOffsetInPointForBig);
	clSetKernelArg(tracingKernel, 9, sizeof(cl_mem), &d_vertexPositionsForBig);
	clSetKernelArg(tracingKernel, 10, sizeof(cl_mem), &d_startVelocitiesForBig);
	clSetKernelArg(tracingKernel, 11, sizeof(cl_mem), &d_endVelocitiesForBig);
	
	clSetKernelArg(tracingKernel, 12, sizeof(cl_mem), &d_canFitInSharedMemory);

	clSetKernelArg(tracingKernel, 13, sizeof(cl_mem), &d_localConnectivities);
	clSetKernelArg(tracingKernel, 14, sizeof(cl_mem), &d_localLinks);
	clSetKernelArg(tracingKernel, 15, sizeof(cl_mem), &d_globalCellIDs);
	clSetKernelArg(tracingKernel, 16, sizeof(cl_mem), &d_globalPointIDs);

	clSetKernelArg(tracingKernel, 17, sizeof(cl_mem), &d_activeBlocks); // Map active block ID to interesting block ID

	clSetKernelArg(tracingKernel, 18, sizeof(cl_mem), &d_blockOfGroups);
	clSetKernelArg(tracingKernel, 19, sizeof(cl_mem), &d_offsetInBlocks);

	clSetKernelArg(tracingKernel, 20, sizeof(cl_mem), &d_stages);
	clSetKernelArg(tracingKernel, 21, sizeof(cl_mem), &d_lastPositionForRK4);
	clSetKernelArg(tracingKernel, 22, sizeof(cl_mem), &d_k1ForRK4);
	clSetKernelArg(tracingKernel, 23, sizeof(cl_mem), &d_k2ForRK4);
	clSetKernelArg(tracingKernel, 24, sizeof(cl_mem), &d_k3ForRK4);
	clSetKernelArg(tracingKernel, 25, sizeof(cl_mem), &d_pastTimes);

	clSetKernelArg(tracingKernel, 26, sizeof(cl_mem), &d_placesOfInterest);
	
	clSetKernelArg(tracingKernel, 27, sizeof(cl_mem), &d_startOffsetInParticles);
	clSetKernelArg(tracingKernel, 28, sizeof(cl_mem), &d_blockedActiveParticles);
	clSetKernelArg(tracingKernel, 29, sizeof(cl_mem), &d_localTetIDs);

	clSetKernelArg(tracingKernel, 30, sizeof(cl_mem), &d_exitCells);

	clSetKernelArg(tracingKernel, 31, configure->GetSharedMemoryKilobytes() * 1024, NULL);
	
	if (configure->UseDouble()) {
		cl_double d_timeStep = configure->GetTimeStep();
		clSetKernelArg(tracingKernel, 34, sizeof(cl_double), &d_timeStep);
		clSetKernelArg(tracingKernel, 35, sizeof(cl_double), &epsilon);
	} else {
		cl_float f_timeStep = configure->GetTimeStep();
		cl_float f_epsilon = epsilon;
		clSetKernelArg(tracingKernel, 34, sizeof(cl_float), &f_timeStep);
		clSetKernelArg(tracingKernel, 35, sizeof(cl_float), &f_epsilon);
	}
}

/// DEBUG ///
void GetFinalPositions();
	


void Tracing() {
	// Initialize d_tetrahedralLinks
	d_tetrahedralLinks = clCreateBuffer(context, CL_MEM_READ_ONLY,
					    sizeof(int) * globalNumOfCells * 4, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device tetrahedralLinks");

	err = clEnqueueCopyBuffer(commandQueue, h_tetrahedralLinks, d_tetrahedralLinks, 0, 0,
				  sizeof(int) * globalNumOfCells * 4, 0, NULL, NULL);
	if (err) lcs::Error("Fail to enqueue copy for d_tetrahedralLinks");

	clFinish(commandQueue);

	// Initialize initial active particle data
	InitializeInitialActiveParticles();

	// Initialize velocity data
	void *velocities[2];
	int currStartVIndex = 1;
	InitializeVelocityData(velocities);

	// Create some dynamic device arrays
	d_blockOfGroups = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * numOfInitialActiveParticles, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device blockOfGroups");

	d_offsetInBlocks = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * numOfInitialActiveParticles, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device offsetInBlocks");

	d_numOfGroupsForBlocks = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * numOfInterestingBlocks, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device numOfGroupsForBlocks");

	d_activeBlocks = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * numOfInterestingBlocks, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device activeBlocks");

	d_activeBlockIndices = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * numOfInterestingBlocks, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device activeBlockIndices");

	d_numOfActiveBlocks = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device numOfActiveBlocks");

	d_startOffsetInParticles = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * (numOfInterestingBlocks + 1), NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device startOffsetInParticles");

	d_blockedActiveParticles = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * numOfInitialActiveParticles, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device blockedAciveParticles");

	// Initialize interestingBlockMarks to {-1}
	InitializeInterestingBlockMarks();
	int iBMCount = 0;

	// Initialize numOfParticlesByStageInBlocks
	int maxNumOfStages;
	switch (lcs::ParticleRecord::GetDataType()) {
	case lcs::ParticleRecord::RK4: maxNumOfStages = 4; break;
	}

	d_numOfParticlesByStageInBlocks = clCreateBuffer(context, CL_MEM_READ_WRITE,
							 sizeof(int) * numOfInterestingBlocks * 4, NULL, &err);
	if (err) lcs::Error("Fail to create a buffer for device numOfParticlesByStageInBlocks");

	// Initialize blocked tracing kernel
	cl_program tracingProgram;
	cl_kernel tracingKernel;
	int tracingWorkGroupSize;

	InitializeTracingKernel(tracingProgram, tracingKernel, tracingWorkGroupSize, configure->GetEpsilon());

	printf("tracingWorkGroupSize = %d\n", tracingWorkGroupSize);

	// Initialize assign groups kernel
	cl_program assignGroupsProgram;
	cl_kernel getNumKernel, assignKernel;
	int getNumWorkGroupSize, assignWorkGroupSize;

	InitializeAssignGroupsKernel(assignGroupsProgram, getNumKernel, assignKernel,
				     getNumWorkGroupSize, assignWorkGroupSize, tracingWorkGroupSize);

	// Initialize redistribute particles kernel
	cl_program redistributeProgram;
	cl_kernel collectBlocksKernel, collectParticlesKernel, statisticsKernel;
	int collectBlocksWorkGroupSize, collectParticlesWorkGroupSize, statisticsWorkGroupSize;

	InitializeRedistributeParticlesKernel(redistributeProgram,
					      collectBlocksKernel, collectParticlesKernel, statisticsKernel,
					      collectBlocksWorkGroupSize, collectParticlesWorkGroupSize, statisticsWorkGroupSize,
					      maxNumOfStages, configure->GetEpsilon());

	// Initialzie collect every k element kernel
	cl_program everyKElementProgram;
	cl_kernel everyKElementKernel;
	int everyKElementWorkGroupSize;

	InitializeCollectEveryKElementKernel(everyKElementProgram, everyKElementKernel, everyKElementWorkGroupSize, maxNumOfStages);

	// Initialize exclusive scan kernel
	cl_program scanProgram;
	cl_kernel scanKernel, reverseUpdateKernel;
	int scanWorkGroupSize, numOfBanks, maxArrSize;

	InitializeExclusiveScanKernel(scanProgram, scanKernel, reverseUpdateKernel, numOfBanks, maxArrSize, scanWorkGroupSize);

	// Initialize collect active particles for new interval kernel
	cl_program collect1Program;
	cl_kernel collect1InitKernel, collect1PickKernel;
	int collect1WorkGroupSize;
	InitializeCollectActiveParticlesForNewIntervalKernel(collect1Program, collect1InitKernel, collect1PickKernel,
							     collect1WorkGroupSize);

	// Initialize collect active particles for new interval run
	cl_program collect2Program;
	cl_kernel collect2InitKernel, collect2PickKernel;
	int collect2WorkGroupSize;
	InitializeCollectActiveParticlesForNewRunKernel(collect2Program, collect2InitKernel, collect2PickKernel,
							collect2WorkGroupSize);

	// Initialize point positions in big blocks
	BigBlockInitializationForPositions();

	// Some start setting
	currActiveParticleArray = 0;
	double currTime = 0;
	double interval = configure->GetTimeInterval();
	cl_event events[10];

	// Main loop for blocked tracing
	int startTime = clock();

	/// DEBUG ///
	kernelSum = 0;
	int numOfKernelCalls = 0;

	for (int frameIdx = 0; frameIdx + 1 < numOfFrames; frameIdx++, currTime += interval) {
		printf("*********Tracing between frame %d and frame %d*********\n", frameIdx, frameIdx + 1);
		printf("\n");

		/// DEBUG ///
		int startTime;
		startTime = clock();

		currStartVIndex = 1 - currStartVIndex;

		// Collect active particles
		int lastNumOfActiveParticles;

		lastNumOfActiveParticles = CollectActiveParticlesForNewInterval(collect1InitKernel, collect1PickKernel,
										collect1WorkGroupSize, scanKernel,
										reverseUpdateKernel, scanWorkGroupSize,
										numOfBanks, d_activeParticles[currActiveParticleArray]);
		
		/// DEBUG ///
		//printf("lastNumOfActiveParticles = %d\n", lastNumOfActiveParticles);
		//int *activeParticles = new int [lastNumOfActiveParticles];
		//err = clEnqueueReadBuffer(commandQueue, d_activeParticles[currActiveParticleArray], CL_TRUE, 0, sizeof(int) * lastNumOfActiveParticles, activeParticles, 0, NULL, NULL);
		//if (err) lcs::Error("Fail to read d_activeParticles");
		//for (int i = 0; i < lastNumOfActiveParticles; i++)
		//	printf("%d\t", activeParticles[i]);
		//delete [] activeParticles;

		/// DEBUG ///
		printf("CollectActiveParticlesForNewInterval done.\n");

		// Load end velocities
		cl_event loadVelocities = LoadVelocities(velocities[1 - currStartVIndex],
							 d_velocities[1 - currStartVIndex], frameIdx + 1);

		clFinish(commandQueue);

		// Initialize big blocks
		BigBlockInitializationForVelocities(currStartVIndex);

		/// DEBUG ///
		printf("BigBlockInitializationForVelocities done.\n");

		while (true) {
			// Get active particles
			currActiveParticleArray = 1 - currActiveParticleArray;

			int numOfActiveParticles;

			numOfActiveParticles = CollectActiveParticlesForNewRun(collect2InitKernel, collect2PickKernel,
									       collect2WorkGroupSize, scanKernel,
									       reverseUpdateKernel, scanWorkGroupSize,
									       numOfBanks,
									       d_activeParticles[1 - currActiveParticleArray],
									       d_activeParticles[currActiveParticleArray],
									       lastNumOfActiveParticles);

			/// DEBUG ///
			printf("CollectActiveParticlesForNewRun done.\n");

			//lcs::CheckIntArrayInDevice("activeParticles.txt", commandQueue, d_activeParticles[currActiveParticleArray], numOfActiveParticles);

			/// DEBUG ///
			printf("numOfActiveParticles = %d\n", numOfActiveParticles);
			printf("\n");

			lastNumOfActiveParticles = numOfActiveParticles;

			if (!numOfActiveParticles) break;

			/// DEBUG ///
			numOfKernelCalls++;

			int numOfActiveBlocks = RedistributeParticles(collectBlocksKernel, collectParticlesKernel,
								      statisticsKernel,
								      collectBlocksWorkGroupSize, collectParticlesWorkGroupSize,
								      statisticsWorkGroupSize,
								      d_activeParticles[currActiveParticleArray],
								      numOfActiveParticles, iBMCount++, maxNumOfStages,
								      scanKernel, reverseUpdateKernel, scanWorkGroupSize,
								      numOfBanks);	

			/// DEBUG ///
			printf("RedistributeParticles done.\n");

			GetStartOffsetInParticles(everyKElementKernel, numOfActiveBlocks, everyKElementWorkGroupSize,
						  numOfActiveParticles);

			/// DEBUG ///
			//lcs::CheckIntArrayInDevice("numOfParticlesByStageInBlocks.txt", commandQueue, d_numOfParticlesByStageInBlocks, numOfActiveBlocks * 4 + 1);
			//lcs::CheckIntArrayInDevice("startOffsetInParticles.txt", commandQueue, d_startOffsetInParticles, numOfActiveBlocks + 1);
			//lcs::GetOrignalUnorderedIntArrayFromPartialSum("numOfParticlesInBlocks.txt", commandQueue,
			//		       			       d_startOffsetInParticles, numOfActiveBlocks);
			//lcs::CheckIntArrayInDevice("localTetID.txt", commandQueue, d_localTetIDs, numOfInitialActiveParticles);	
	
			int numOfWorkGroups = AssignWorkGroups(getNumKernel, assignKernel, 
							       getNumWorkGroupSize, assignWorkGroupSize, numOfActiveBlocks,
							       scanWorkGroupSize, numOfBanks, scanKernel, reverseUpdateKernel);

			/// DEBUG ///
			//lcs::CheckIntArrayInDevice("blockOfGroups.txt", commandQueue, d_blockOfGroups, numOfWorkGroups);
			//lcs::CheckIntArrayInDevice("offsetInBlocks.txt", commandQueue, d_offsetInBlocks, numOfWorkGroups);
			//lcs::CheckFloatArrayInDevice("initialLastPositions.txt", commandQueue, d_lastPositionForRK4, numOfInitialActiveParticles * 3);
			//lcs::CheckIntArrayInDevice("blockedActiveParticles.txt", commandQueue, d_blockedActiveParticles, numOfInitialActiveParticles);
			//lcs::CheckIntArrayInDevice("stages.txt", commandQueue, d_stages, numOfInitialActiveParticles);

			printf("numOfWorkGroups = %d\n", numOfWorkGroups);	

			LaunchBlockedTracingKernel(tracingKernel, tracingWorkGroupSize, 0, NULL,
						   currStartVIndex, numOfWorkGroups, currTime, currTime + interval);

			/// DEBUG ///
			//lcs::CheckIntArrayInDevice("exitCells.txt", commandQueue, d_exitCells, numOfInitialActiveParticles);
			//lcs::CheckFloatArrayInDevice("lastPositions.txt", commandQueue, d_lastPositionForRK4, numOfInitialActiveParticles * 3);
			//GetFinalPositions();
	
			//break;

			/// DEBUG ///
			//double oldTime;

			//LaunchBlockedTracingKernel(blockedTracingKernelForTest, workGroupSize, numOfEvents, events, currStartVIndex, numOfActiveBlocks, currTime, currTime + interval);

			//oldTime = kernelSum;
			//LaunchBlockedTracingKernel(blockedTracingKernel, workGroupSize, numOfEvents, events, currStartVIndex, numOfActiveBlocks, currTime, currTime + interval);
			//kernelSum = oldTime;

			// Update necessary active particle data
			//switch (lcs::ParticleRecord::GetDataType()) {
			//case lcs::ParticleRecord::RK4: {
			//	UpdateSqueezedArraysForRK4(squeezedLastPositionForRK4,
			//							   squeezedK1ForRK4, squeezedK2ForRK4, squeezedK3ForRK4, squeezedStage, squeezedExitCells,
			//							   numOfActiveParticles);
			//	UpdateActiveParticleDataForRK4(blockedActiveParticleIDList, squeezedLastPositionForRK4,
			//								   squeezedK1ForRK4, squeezedK2ForRK4, squeezedK3ForRK4, squeezedStage, squeezedExitCells,
			//								   numOfActiveParticles);
			//							   } break;
			//}

			// Release events
			//clReleaseEvent(initDActiveBlockList);
			//clReleaseEvent(initDStartOffsetInParticles);
			//clReleaseEvent(initDBlockedActiveParticles);
			//clReleaseEvent(initDBlockedCellLocations);
		}

		int endTime = clock();
		printf("This interval cost %lf sec.\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
		printf("\n");

		/// DEBUG ///
		//if (frameIdx == 8) {
		//	lcs::CheckFloatArrayInDevice("lastPositions.txt", commandQueue, d_lastPositionForRK4, numOfInitialActiveParticles * 3);
		//	GetFinalPositions();

			//break;
		//}
		//printf("curr frameIdx = %d\n", frameIdx);
	}

	// Release device resources
	clReleaseMemObject(d_exclusiveScanArrayForInt);

	/// DEBUG ///
	printf("kernelSum = %lf\n", kernelSum);
	printf("numOfKernelCalls = %d\n", numOfKernelCalls);

	/// DEBUG ///
	int endTime = clock();
	printf("The total tracing time is %lf sec.\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
	printf("\n");
}

void GetFinalPositions() {
	void *finalPositions;

	if (configure->UseDouble())
		finalPositions = new double [numOfInitialActiveParticles * 3];
	else
		finalPositions = new float [numOfInitialActiveParticles * 3];

	switch (lcs::ParticleRecord::GetDataType()) {
	case lcs::ParticleRecord::RK4: {
		if (configure->UseDouble())
			clEnqueueReadBuffer(commandQueue, d_lastPositionForRK4, CL_TRUE, 0, sizeof(double) * 3 * numOfInitialActiveParticles, finalPositions, 0, NULL, NULL);
		else
			clEnqueueReadBuffer(commandQueue, d_lastPositionForRK4, CL_TRUE, 0, sizeof(float) * 3 * numOfInitialActiveParticles, finalPositions, 0, NULL, NULL);
								   } break;
	}

	FILE *fout = fopen(lastPositionFile, "w");
	for (int i = 0; i < numOfInitialActiveParticles; i++) {
		int gridPointID = particleRecords[i]->GetGridPointID();
		int z = gridPointID % (configure->GetBoundingBoxZRes() + 1);
		int temp = gridPointID / (configure->GetBoundingBoxZRes() + 1);
		int y = temp % (configure->GetBoundingBoxYRes() + 1);
		int x = temp / (configure->GetBoundingBoxYRes() + 1);
		fprintf(fout, "%d %d %d:", x, y, z);
		for (int j = 0; j < 3; j++)
			if (configure->UseDouble())
				fprintf(fout, " %lf", ((double *)finalPositions)[i * 3 + j]);
			else
				fprintf(fout, " %lf", ((float *)finalPositions)[i * 3 + j]);
		fprintf(fout, "\n");
	}
	fclose(fout);

	if (configure->UseDouble())
		delete [] (double *)finalPositions;
	else
		delete [] (float *)finalPositions;
}

int main() {
	// Test the system
	SystemTest();

	// Read the configure file
	ReadConfFile();

	// Load all the frames
	LoadFrames();

	// Put both topological and geometrical data into arrays
	GetTopologyAndGeometry();

	// Get the global bounding box
	GetGlobalBoundingBox();

	// Calculate the number of blocks in X, Y and Z
	CalculateNumOfBlocksInXYZ();

	// Divide the flow domain into blocks
	Division();

	// Initially locate global tetrahedral cells for interesting Cartesian grid points
	InitialCellLocation();

	// Main Tracing Process
	Tracing();

	// Get final positions for initial active particles
	GetFinalPositions();

	return 0;
}
