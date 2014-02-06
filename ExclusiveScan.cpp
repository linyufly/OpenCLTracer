#include <CL/opencl.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>

const char *exclusiveScanKernels = "ExclusiveScanKernels.cl";

void Error(const char *str) {
	printf("%s\n", str);
	system("pause");
	exit(0);
}

cl_int err;
cl_uint numOfPlatforms, numOfDevices;
cl_platform_id *platformIDs;
cl_device_id *deviceIDs;
cl_context context;
cl_command_queue commandQueue;

cl_program CreateProgram(const char *kernelFile, const char *kernelName) {
	// Load the kernel code
	FILE *fin = fopen(kernelFile, "r");
	if (fin == NULL) {
		char str[100];
		sprintf(str, "Fail to load the %s kernel", kernelName);
		Error(str);
	}

	std::string kernelCode = "";

	char ch;
	for (; (ch = fgetc(fin)) != EOF; kernelCode += ch);
	size_t codeLength = kernelCode.length() + 1; // Consider the tailing 0
	const char *codeString = kernelCode.c_str();

	// Create a program based on the kernel code
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&codeString, &codeLength, &err);
	if (err) {
		char str[100];
		sprintf(str, "Fail to create a program for the %s kernel", kernelName);
		Error(str);
	}

	// Build the program and output the build information
	err = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	bool compilationFailure = err;

	size_t lengthOfBuildInfo;
	err = clGetProgramBuildInfo(program, deviceIDs[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &lengthOfBuildInfo);
	if (err) Error("Fail to get the length of the program build information");

	char *buildInfo = new char [lengthOfBuildInfo + 1];
	buildInfo[lengthOfBuildInfo] = 0;
	err = clGetProgramBuildInfo(program, deviceIDs[0], CL_PROGRAM_BUILD_LOG, lengthOfBuildInfo, buildInfo, NULL);
	if (err) Error("Fail to get the program build information");

	printf("The program build information is as follows.\n\n");

	if (buildInfo[0] == '\n') printf("Successful Compilation\n\n");
	else printf("%s\n", buildInfo);
	printf("** end of build information **\n");
	delete [] buildInfo;
	printf("\n");
	if (compilationFailure) {
		char str[100];
		sprintf(str, "Fail to build the program of the %s kernel", kernelName);
		Error(str);
	}

	return program;
}

void CheckValues(int length, cl_mem d_arr) {
	int *GPUResult = new int [length];
	memset(GPUResult, 0, sizeof(int) * length);
	err = clEnqueueReadBuffer(commandQueue, d_arr, CL_TRUE, 0, sizeof(int) * length, GPUResult, 0, NULL, NULL);
	printf("err = %d\n", err);
	if (err) Error("Fail to read d_arr");

	for (int i = 0; i < length; i++)
		printf("%d: %d\t", i, GPUResult[i]);
	printf("\n");

	scanf("%*c");
}

int main() {
	// Get platform information
	err = clGetPlatformIDs(0, NULL, &numOfPlatforms);
	if (err) Error("Fail to get the number of platforms");
	printf("The machine has %d platform(s) for OpenCL.\n", numOfPlatforms);

	platformIDs = new cl_platform_id [numOfPlatforms];
	err = clGetPlatformIDs(numOfPlatforms, platformIDs, NULL);
	if (err) Error("Fail to get the platform list");

	int cudaPlatformID = -1;

	for (int i = 0; i < numOfPlatforms; i++) {
		char platformName[50];
		err = clGetPlatformInfo(platformIDs[i], CL_PLATFORM_NAME, 50, platformName, NULL);
		if (err) Error("Fail to get the platform name");
		printf("Platform %d is %s\n", i + 1, platformName);
		if (!strcmp(platformName, "NVIDIA CUDA")) cudaPlatformID = i;
	}
	printf("\n");

	if (cudaPlatformID == -1) Error("Fail to find an NVIDIA CUDA platform");

	printf("Platform %d (NVIDIA CUDA) is chosen for use.\n", cudaPlatformID + 1);
	printf("\n");

	// Get device information
	err = clGetDeviceIDs(platformIDs[cudaPlatformID], CL_DEVICE_TYPE_GPU, 0, NULL, &numOfDevices);
	if (err) Error("Fail to get the number of devices");
	printf("CUDA platform has %d device(s).\n", numOfDevices);

	deviceIDs = new cl_device_id [numOfDevices];
	err = clGetDeviceIDs(platformIDs[cudaPlatformID], CL_DEVICE_TYPE_GPU, numOfDevices, deviceIDs, NULL);
	if (err) Error("Fail to get the device list");
	for (int i = 0; i < numOfDevices; i++) {
		char deviceName[50];
		err = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_NAME, 50, deviceName, NULL);
		if (err) Error("Fail to get the device name");
		printf("Device %d is %s\n", i + 1, deviceName);
	}
	printf("\n");

	// Create a context
	context = clCreateContext(NULL, numOfDevices, deviceIDs, NULL, NULL, &err);
	if (err) Error("Fail to create a context");

	printf("Device 1 is chosen for use.\n");
	printf("\n");

	// Create a command queue for the first device
	commandQueue = clCreateCommandQueue(context, deviceIDs[0],
									    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err);
	if (err) Error("Fail to create a command queue");

	// create the program
	cl_program program = CreateProgram(exclusiveScanKernels, "exclusive scan");

	// create two kernels
	cl_kernel scanKernel = clCreateKernel(program, "Scan", &err);
	if (err) Error("Fail to create the kernel for scan");

	cl_kernel reverseUpdateKernel = clCreateKernel(program, "ReverseUpdate", &err);
	if (err) Error("Fail to create the kernel for reverse update");

	// Get the work group size
	size_t maxWorkGroupSize;
	err = clGetKernelWorkGroupInfo(scanKernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
								   sizeof(size_t), &maxWorkGroupSize, NULL);
	printf("maxWorkGroupSize = %d\n", maxWorkGroupSize);

	err = clGetKernelWorkGroupInfo(reverseUpdateKernel, deviceIDs[0], CL_KERNEL_WORK_GROUP_SIZE,
								   sizeof(size_t), &maxWorkGroupSize, NULL);
	printf("maxWorkGroupSize = %d\n", maxWorkGroupSize);

	// Set work group size to 64

	int workGroupSize = 512;

	int length = 2048000;
	int *arr = new int [length];
	for (int i = 0; i < length; i++)
		arr[i] = rand() % 100;

	int *prefixSum = new int [length];
	prefixSum[0] = 0;

	int t0 = clock();

	for (int i = 1; i < length; i++)
		prefixSum[i] = prefixSum[i - 1] + arr[i - 1];

	int t1 = clock();

	printf("time1: %lf\n", (t1 - t0) * 1.0 / CLOCKS_PER_SEC);

	cl_mem d_arr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * length, NULL, &err);
	if (err) Error("Fail to create d_arr");

	err = clEnqueueWriteBuffer(commandQueue, d_arr, CL_TRUE, 0, sizeof(int) * length, arr, 0, NULL, NULL);
	if (err) Error("Fail to write d_arr");

	clSetKernelArg(scanKernel, 0, sizeof(cl_mem), &d_arr);
	cl_int d_length = length;
	clSetKernelArg(scanKernel, 1, sizeof(cl_int), &d_length);
	cl_int d_step = 1;
	clSetKernelArg(scanKernel, 2, sizeof(cl_int), &d_step);
	clSetKernelArg(scanKernel, 3, sizeof(cl_int) * (workGroupSize * 2 + workGroupSize * 2 / 16 + 1), NULL);

	int problemSize = length;
	int records[10];
	int num = 0;

	int t2 = clock();

	for (; problemSize > 1; problemSize = (problemSize - 1) / (workGroupSize * 2) + 1) {

		if (num) d_step *= workGroupSize * 2;

		printf("d_step = %d\n", d_step);

		records[num++] = problemSize;

		printf("problemSize = %d\n", problemSize);

		clSetKernelArg(scanKernel, 2, sizeof(cl_int), &d_step);

		size_t globalWorkSize = ((problemSize - 1) / (workGroupSize * 2) + 1) * workGroupSize;
		size_t localWorkSize = workGroupSize;

		err = clEnqueueNDRangeKernel(commandQueue, scanKernel, 1, NULL, &globalWorkSize, &localWorkSize,
								 0, NULL, NULL);
		if (err) Error("Fail to enqueue scan");
		clFinish(commandQueue);
	}

	//CheckValues(length, d_arr);

	int zero = 0;
	clEnqueueWriteBuffer(commandQueue, d_arr, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL);

	printf("d_step = %d\n", d_step);

	//scanf("%*c");

	clSetKernelArg(reverseUpdateKernel, 0, sizeof(cl_mem), &d_arr);
	clSetKernelArg(reverseUpdateKernel, 1, sizeof(cl_int), &d_length);

	for (int i = num - 1; i >= 0; i--, d_step /= workGroupSize * 2) {
		printf("d_step = %d\n", d_step);

		clSetKernelArg(reverseUpdateKernel, 2, sizeof(cl_int), &d_step);
		size_t globalWorkSize = ((records[i] - 1) / (workGroupSize * 2) + 1) * workGroupSize;
		size_t localWorkSize = workGroupSize;

		printf("globalWorkSize = %d, localWorkSize = %d\n", globalWorkSize, localWorkSize);

		err = clEnqueueNDRangeKernel(commandQueue, reverseUpdateKernel, 1, NULL, &globalWorkSize, &localWorkSize,
								 0, NULL, NULL);
		if (err) Error("Fail to enqueue scan");
		clFinish(commandQueue);
	}

	int t3 = clock();

	printf("time: %lf\n", (t3 - t2) * 1.0 / CLOCKS_PER_SEC);

	int *GPUResult = new int [length];
	memset(GPUResult, 0, sizeof(int) * length);
	err = clEnqueueReadBuffer(commandQueue, d_arr, CL_TRUE, 0, sizeof(int) * length, GPUResult, 0, NULL, NULL);
	printf("err = %d\n", err);
	if (err) Error("Fail to read d_arr");

	for (int i = 0; i < length; i++)
		if (GPUResult[i] != prefixSum[i]) printf("at i = %d, GPUResult[%d] = %d, prefixSum[%d] = %d\n", i, i, GPUResult[i], i, prefixSum[i]);

	system("pause");
	return 0;
}