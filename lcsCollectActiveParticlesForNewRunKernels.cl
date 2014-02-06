/************************************************************************
File		:	lcsCollectActiveParticlesForNewRunKernels.cl
Author		:	Mingcheng Chen
Last Update	:	September 27th, 2012
*************************************************************************/

__kernel void InitializeScanArray(__global int *exitCells, __global int *oldActiveParticles, __global int *scanArray, int length) {
	int globalID = get_global_id(0);
	if (globalID < length)
		scanArray[globalID] = exitCells[oldActiveParticles[globalID]] < 0 ? 0 : 1;
}

__kernel void CollectActiveParticles(__global int *exitCells, __global int *oldActiveParticles,__global int *scanArray,
				     __global int *newActiveParticles, int length) {
	int globalID = get_global_id(0);
	if (globalID < length)
		if (exitCells[oldActiveParticles[globalID]] >= 0)
			newActiveParticles[scanArray[globalID]] = oldActiveParticles[globalID];
}
