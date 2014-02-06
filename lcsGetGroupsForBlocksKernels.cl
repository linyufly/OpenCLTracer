/*********************************************************
File		:	lcsGetGroupsForBlocksKernels.cl
Author		:	Mingcheng Chen
Last Update	:	September 18th, 2012
**********************************************************/

__kernel void GetNumOfGroupsForBlocks(__global int *startOffsetInParticles,
				      __global int *numOfGroupsForBlocks,
				      int numOfActiveBlocks, int groupSize) {
	int globalID = get_global_id(0);
	if (globalID < numOfActiveBlocks) {
		int numOfParticles = startOffsetInParticles[globalID + 1] - startOffsetInParticles[globalID];
		numOfGroupsForBlocks[globalID] = (numOfParticles - 1) / groupSize + 1;
	}
}

__kernel void AssignGroups(__global int *numOfGroupsForBlocks, // It should be the prefix sum now.
			   __global int *blockOfGroups,
			   __global int *offsetInBlocks) {
	int blockID = get_group_id(0);
	int threadID = get_local_id(0);
	int workSize = get_local_size(0);
	int startOffset = numOfGroupsForBlocks[blockID];
	int numOfGroups = numOfGroupsForBlocks[blockID + 1] - startOffset;
	//int numOfGroupsPerThread = (numOfGroups - 1) / workSize + 1;

	for (int i = threadID; i < numOfGroups; i += workSize) {
		blockOfGroups[startOffset + i] = blockID;
		offsetInBlocks[startOffset + i] = i;
	}
}
