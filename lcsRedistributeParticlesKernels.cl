/******************************************************************
File		:	lcsRedistributeParticlesKernels.cl
Author		:	Mingcheng Chen
Last Update	:	September 25th, 2012
*******************************************************************/

inline int Sign(double a, double eps) {
	return a < -eps ? -1 : a > eps;
}

inline int GetLocalTetID(int blockID, int tetID,
			 __global int *startOffsetsInLocalIDMap,
			 __global int *blocksOfTets,
			 __global int *localIDsOfTets) { // blockID and tetID are all global IDs.
	int offset = startOffsetsInLocalIDMap[tetID];
	int endOffset = -1;
	while (1) {
		if (blocksOfTets[offset] == blockID) return localIDsOfTets[offset];
		if (endOffset == -1) endOffset = startOffsetsInLocalIDMap[tetID + 1];

		offset++;
		if (offset >= endOffset) return -1;
	}
}

inline int GetBlockID(int x, int y, int z, int numOfBlocksInY, int numOfBlocksInZ) {
	return (x * numOfBlocksInY + y) * numOfBlocksInZ + z;
}

__kernel void CollectActiveBlocks(__global int *activeParticles,
				  __global int *exitCells,
				  //__global int *stages,
				  __global double *placesOfInterest,

				  //__global int *particleOrders,
				  __global int *localTetIDs,
				  __global int *blockLocations,

				  __global int *interestingBlockMap,

				  __global int *startOffsetsInLocalIDMap,
				  __global int *blocksOfTets,
				  __global int *localIDsOfTets,

				  //volatile __global int *numOfParticlesByStageInBlocks,
				  volatile __global int *interestingBlockMarks,

				  __global int *activeBlocks,
				  __global int *activeBlockIndices,
				  volatile __global int *numOfActiveBlocks, // Initially 0

				  //__global int *activeBlockOfParticles,

				  int mark,
				  int numOfActiveParticles, //int numOfStages,
				  int numOfBlocksInX, int numOfBlocksInY, int numOfBlocksInZ,
				  double globalMinX, double globalMinY, double globalMinZ,
				  double blockSize,
				  double epsilon) {
	int globalID = get_global_id(0);

	if (globalID < numOfActiveParticles) {
		int particleID = activeParticles[globalID];
		double posX = placesOfInterest[particleID * 3];
		double posY = placesOfInterest[particleID * 3 + 1];
		double posZ = placesOfInterest[particleID * 3 + 2];

		int x = (int)((posX - globalMinX) / blockSize);
		int y = (int)((posY - globalMinY) / blockSize);
		int z = (int)((posZ - globalMinZ) / blockSize);

		// Intuitive block ID
		int blockID = GetBlockID(x, y, z, numOfBlocksInY, numOfBlocksInZ);
		int tetID = exitCells[particleID];

		int localTetID = GetLocalTetID(blockID, tetID, startOffsetsInLocalIDMap, blocksOfTets, localIDsOfTets);

		if (localTetID == -1) {
			int dx[3], dy[3], dz[3];
			int lx = 1, ly = 1, lz = 1;
			dx[0] = dy[0] = dz[0] = 0;

			double xLower = globalMinX + x * blockSize;
			double yLower = globalMinY + y * blockSize;
			double zLower = globalMinZ + z * blockSize;

			if (!Sign(xLower - posX, 2 * epsilon)) dx[lx++] = -1;
			if (!Sign(yLower - posY, 2 * epsilon)) dy[ly++] = -1;
			if (!Sign(zLower - posZ, 2 * epsilon)) dz[lz++] = -1;

			if (!Sign(xLower + blockSize - posX, 2 * epsilon)) dx[lx++] = 1;
			if (!Sign(yLower + blockSize - posY, 2 * epsilon)) dy[ly++] = 1;
			if (!Sign(zLower + blockSize - posZ, 2 * epsilon)) dz[lz++] = 1;

			// Check every necessary neightbor
			for (int i = 0; localTetID == -1 && i < lx; i++)
				for (int j = 0; localTetID == -1 && j < ly; j++)
					for (int k = 0; k < lz; k++) {
						if (i + j + k == 0) continue;
						int _x = x + dx[i];
						int _y = y + dy[j];
						int _z = z + dz[k];

						if (_x < 0 || _y < 0 || _z < 0 ||
						    _x >= numOfBlocksInX || _y >= numOfBlocksInY || _z >= numOfBlocksInZ)
							continue;

						blockID = GetBlockID(_x, _y, _z, numOfBlocksInY, numOfBlocksInZ);
						localTetID = GetLocalTetID(blockID, tetID, startOffsetsInLocalIDMap,
									   blocksOfTets, localIDsOfTets);

						if (localTetID != -1) break;
					}

			/// DEBUG ///
			if (localTetID == -1) while (1);
		}

		// localTetID must not be -1 at that point.

		localTetIDs[particleID] = localTetID;

		int interestingBlockID = interestingBlockMap[blockID];
		blockLocations[particleID] = interestingBlockID;

		int oldMark = atomic_add(interestingBlockMarks + interestingBlockID, 0);

		int index;

		if (oldMark < mark) {
			int delta = mark - oldMark;
			int newMark = atomic_add(interestingBlockMarks + interestingBlockID, delta);

			if (newMark >= mark)
				atomic_add(interestingBlockMarks + interestingBlockID, -delta);
			else {
				index = atomic_add(numOfActiveBlocks, 1);
				activeBlocks[index] = interestingBlockID;
				activeBlockIndices[interestingBlockID] = index;
			}
		}

		// This one cannot be calculated in that kernel
		//activeBlockOfParticles[particleID] = index;
	}
}

__kernel void GetNumOfParticlesByStageInBlocks(volatile __global int *numOfParticlesByStageInBlocks,
					       __global int *particleOrders,
					       __global int *stages,
					       __global int *activeParticles,
					       //__global int *activeBlockOfParticles,
					       __global int *blockLocations,
					       __global int *activeBlockIndices,
					       int numOfStages, int numOfActiveParticles) {
	int globalID = get_global_id(0);
	if (globalID < numOfActiveParticles) {
		int particleID = activeParticles[globalID];
		//int posi = activeBlockOfParticles[particleID] * numOfStages + stages[particleID];
		int posi = activeBlockIndices[blockLocations[particleID]] * numOfStages + stages[particleID];
		particleOrders[particleID] = atomic_add(numOfParticlesByStageInBlocks + posi, 1);
	}
}

__kernel void CollectParticlesToBlocks(__global int *numOfParticlesByStageInBlocks, // Now it is a prefix sum array.
				       __global int *particleOrders,
				       __global int *stages,
				       __global int *activeParticles,
				       __global int *blockLocations,
				       __global int *activeBlockIndices,

				       __global int *blockedParticleList,
				       int numOfStages, int numOfActiveParticles
				       ) {
	int globalID = get_global_id(0);

	if (globalID < numOfActiveParticles) {
		int particleID = activeParticles[globalID];

		int interestingBlockID = blockLocations[particleID];
		int activeBlockID = activeBlockIndices[interestingBlockID];
		int stage = stages[particleID];

		int position = numOfParticlesByStageInBlocks[activeBlockID * numOfStages + stage] + particleOrders[particleID];

		blockedParticleList[position] = particleID;
	}
}
