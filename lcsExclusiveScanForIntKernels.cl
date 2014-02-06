/******************************************************************
File			:		lcsExclusiveScanKernels.cl
Author			:		Mingcheng Chen
Last Update		:		September 24th, 2012
*******************************************************************/

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#define CONFLICT_FREE_OFFSET(n) ((n) >> (LOG_NUM_BANKS))
#define POSI(n) ((n) + CONFLICT_FREE_OFFSET(n))

__kernel void Scan(__global int *globalArray, int length, int step, __local int *localArray) {
	int localID = get_local_id(0);
	int groupID = get_group_id(0);
	int groupSize = get_local_size(0);
	int startOffset = (groupSize << 1) * groupID * step;
	
	int posi1 = startOffset + localID * step;
	int posi2 = posi1 + groupSize * step;
	
	localArray[POSI(localID)] = posi1 < length ? globalArray[posi1] : 0;
	localArray[POSI(localID + groupSize)] = posi2 < length ? globalArray[posi2] : 0;
	
	// Up-sweep
	for (int stride = 1, d = groupSize; stride <= groupSize; stride <<= 1, d >>= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if (localID < d) {
			posi1 = stride * ((localID << 1) + 1) - 1;
			posi2 = posi1 + stride;
			localArray[POSI(posi2)] += localArray[POSI(posi1)];
		}
	}
	
	// Down-sweep
	for (int stride = groupSize, d = 1; stride >= 1; stride >>= 1, d <<= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if (localID < d) {
			posi1 = stride * ((localID << 1) + 1) - 1;
			posi2 = POSI(posi1 + stride);
			posi1 = POSI(posi1);
			
			int t = localArray[posi1];
			localArray[posi1] = localArray[posi2];
			localArray[posi2] = localArray[posi2] * !!localID + t;
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Write to global memory
	posi1 = startOffset + localID * step;
	posi2 = posi1 + groupSize * step;
	
	if (posi1 < length) globalArray[posi1] = localArray[POSI(localID)];
	if (posi2 < length) globalArray[posi2] = localArray[POSI(localID + groupSize)];
}

__kernel void ReverseUpdate(__global int *globalArray, int length, int step) {
	int localID = get_local_id(0);
	int groupID = get_group_id(0);
	int groupSize = get_local_size(0);
	int startOffset = groupID * (groupSize << 1) * step;
	
	if (groupID) {
		int value = globalArray[startOffset];
		int posi1 = startOffset + localID * step;
		int posi2 = posi1 + groupSize * step;
		if (posi1 < length && localID) globalArray[posi1] += value;
		if (posi2 < length) globalArray[posi2] += value;
	}
}
