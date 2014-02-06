/*************************************************************************
File		:	lcsCollectActiveParticlesForNewIntervalKernels.cl
Author		:	Mingcheng Chen
Last Update	:	September 3rd, 2012
**************************************************************************/

__kernel void InitializeScanArray(__global int *exitCells, __global int *scanArray, int length) {
	int globalID = get_global_id(0);
	if (globalID < length) {
		if (exitCells[globalID] < -1) exitCells[globalID] = -(exitCells[globalID] + 2);
		scanArray[globalID] = exitCells[globalID] == -1 ? 0 : 1;
	}
}

__kernel void CollectActiveParticles(__global int *exitCells, __global int *scanArray, __global int *activeParticles, int length) {
	int globalID = get_global_id(0);
	if (globalID < length) {
		if (exitCells[globalID] != -1)
			activeParticles[scanArray[globalID]] = globalID;
	}
}
