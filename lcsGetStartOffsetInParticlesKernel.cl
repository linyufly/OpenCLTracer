/**************************************************************
File		:	lcsGetStartOffsetInParticlesKernel.cl
Author		:	Mingcheng Chen
Last Update	:	September 9th, 2012
***************************************************************/

__kernel void CollectEveryKElement(__global int* input, __global int* output, int k, int length) {
	int globalID = get_global_id(0);
	if (globalID < length)
		output[globalID] = input[globalID * k];
}
