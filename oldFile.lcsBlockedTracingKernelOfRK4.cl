/******************************************************************
File			:		lcsBlockedTracingKernel.cl
Author			:		Mingcheng Chen
Last Update		:		July 7th, 2012
*******************************************************************/

inline double DeterminantThree(double *a) {
	// a[0] a[1] a[2]
	// a[3] a[4] a[5]
	// a[6] a[7] a[8]
	return a[0] * a[4] * a[8] + a[1] * a[5] * a[6] + a[2] * a[3] * a[7] -
		   a[0] * a[5] * a[7] - a[1] * a[3] * a[8] - a[2] * a[4] * a[6];
}

inline void CalculateNaturalCoordinates(double X, double Y, double Z,
										double *tetX, double *tetY, double *tetZ, double *coordinates) {
	X -= tetX[0];
	Y -= tetY[0];
	Z -= tetZ[0];

	double det[9] = {tetX[1] - tetX[0], tetY[1] - tetY[0], tetZ[1] - tetZ[0],
					 tetX[2] - tetX[0], tetY[2] - tetY[0], tetZ[2] - tetZ[0],
					 tetX[3] - tetX[0], tetY[3] - tetY[0], tetZ[3] - tetZ[0]};

	double V = 1 / DeterminantThree(det);

	double z41 = tetZ[3] - tetZ[0];
	double y34 = tetY[2] - tetY[3];
	double z34 = tetZ[2] - tetZ[3];
	double y41 = tetY[3] - tetY[0];
	double a11 = (z41 * y34 - z34 * y41) * V;

	double x41 = tetX[3] - tetX[0];
	double x34 = tetX[2] - tetX[3];
	double a12 = (x41 * z34 - x34 * z41) * V;

	double a13 = (y41 * x34 - y34 * x41) * V;

	coordinates[1] = a11 * X + a12 * Y + a13 * Z;

	double y12 = tetY[0] - tetY[1];
	double z12 = tetZ[0] - tetZ[1];
	double a21 = (z41 * y12 - z12 * y41) * V;

	double x12 = tetX[0] - tetX[1];
	double a22 = (x41 * z12 - x12 * z41) * V;

	double a23 = (y41 * x12 - y12 * x41) * V;

	coordinates[2] = a21 * X + a22 * Y + a23 * Z;

	double z23 = tetZ[1] - tetZ[2];
	double y23 = tetY[1] - tetY[2];
	double a31 = (z23 * y12 - z12 * y23) * V;

	double x23 = tetX[1] - tetX[2];
	double a32 = (x23 * z12 - x12 * z23) * V;

	double a33 = (y23 * x12 - y12 * x23) * V;

	coordinates[3] = a31 * X + a32 * Y + a33 * Z;

	coordinates[0] = 1 - coordinates[1] - coordinates[2] - coordinates[3];
}

inline int gFindCell(double *particle, __global int *connectivities, __global int *links,
					 __global double *vertexPositions,
					 double epsilon, int guess, double *coordinates) {
	double tetX[4], tetY[4], tetZ[4];

	while (true) {
		for (int i = 0; i < 4; i++) {
			int pointID = connectivities[(guess << 2) | i];

			tetX[i] = vertexPositions[pointID * 3];
			tetY[i] = vertexPositions[pointID * 3 + 1];
			tetZ[i] = vertexPositions[pointID * 3 + 2];
		}

		CalculateNaturalCoordinates(particle[0], particle[1], particle[2], tetX, tetY, tetZ, coordinates);
		
		int index = 0;
		for (int i = 1; i < 4; i++)
			if (coordinates[i] < coordinates[index]) index = i;
		if (coordinates[index] >= -epsilon) break;

		guess = links[(guess << 2) | index];
		
		if (guess == -1) break;
	}

	return guess;
}

inline int FindCell(double *particle, __local int *connectivities, __local int *links, __local double *vertexPositions,
					double epsilon, int guess, double *coordinates) {
	double tetX[4], tetY[4], tetZ[4];
	while (true) {
		for (int i = 0; i < 4; i++) {
			int pointID = connectivities[(guess << 2) | i];
			tetX[i] = vertexPositions[pointID * 3];
			tetY[i] = vertexPositions[pointID * 3 + 1];
			tetZ[i] = vertexPositions[pointID * 3 + 2];
		}
		CalculateNaturalCoordinates(particle[0], particle[1], particle[2], tetX, tetY, tetZ, coordinates);
		int index = 0;
		for (int i = 1; i < 4; i++)
			if (coordinates[i] < coordinates[index]) index = i;
		if (coordinates[index] >= -epsilon) break;
		guess = links[(guess << 2) | index];
		if (guess == -1) break;
	}
	return guess;
}

__kernel void BlockedTracing(__global double *globalVertexPositions,
							 __global double *globalStartVelocities,
							 __global double *globalEndVelocities,
							 __global int *globalTetrahedralConnectivities,
							 __global int *globalTetrahedralLinks,

							 __global int *startOffsetInCell,
							 __global int *startOffsetInPoint,

							 __global int *startOffsetInCellForBig,
							 __global int *startOffsetInPointForBig,
							 __global double *vertexPositionsForBig,
							 __global double *startVelocitiesForBig,
							 __global double *endVelocitiesForBig,

							 __global bool *canFitInSharedMemory,

							 __global int *blockedLocalConnectivities,
							 __global int *blockedLocalLinks,
							 __global int *blockedGlobalCellIDs,
							 __global int *blockedGlobalPointIDs,

							 __global int *activeBlockList, // Map active block ID to interesting block ID

							 __global int *stage,
							 __global double *lastPosition,
							 __global double *k1,
							 __global double *k2,
							 __global double *k3,
							 __global double *pastTimes,
							 __global int *startOffsetInParticle,
							 __global int *blockedActiveParticleIDList,
							 __global int *blockedCellLocationList,

							 __local void *sharedMemory,
							 
							 double startTime, double endTime, double timeStep,
							 double epsilon,
							 
							 __global int *squeezedStage,
							 __global double *squeezedLastPosition,
							 __global double *squeezedK1,
							 __global double *squeezedK2,
							 __global double *squeezedK3,
							 __global int *squeezedExitCells
							 ) {
	// Get work group ID, which is equal to active block ID
	int activeBlockID = get_group_id(0);
	
	// Get number of threads in a work group
	int numOfThreads = get_local_size(0);

	// Get local thread ID
	int localID = get_local_id(0);

	// Get interesting block ID of the current active block ID
	int interestingBlockID = activeBlockList[activeBlockID];

	// Declare some arrays
	__local double *vertexPositions;
	__local double *startVelocities;
	__local double *endVelocities;
	__local int *connectivities;
	__local int *links;

	__global double *gVertexPositions;
	__global double *gStartVelocities;
	__global double *gEndVelocities;
	__global int *gConnectivities;
	__global int *gLinks;

	bool canFit = canFitInSharedMemory[interestingBlockID];

	int startCell = startOffsetInCell[interestingBlockID];
	int startPoint = startOffsetInPoint[interestingBlockID];

	int numOfCells = startOffsetInCell[interestingBlockID + 1] - startCell;
	int numOfPoints = startOffsetInPoint[interestingBlockID + 1] - startPoint;

	int startCellForBig = startOffsetInCellForBig[interestingBlockID];
	int startPointForBig = startOffsetInPointForBig[interestingBlockID];

	if (canFit) { // This branch fills in the shared memory
		// Initialize vertexPositions, startVelocities and endVelocities
		vertexPositions = (__local double *)sharedMemory;
		startVelocities = vertexPositions + numOfPoints * 3;
		endVelocities = startVelocities + numOfPoints * 3;

		// Initialize connectivities and links
		connectivities = (__local int *)(endVelocities + numOfPoints * 3);
		links = connectivities + (numOfCells << 2);
	} else { // This branch fills in the global memory
		// Initialize vertexPositions, startVelocities and endVelocities
		gVertexPositions = vertexPositionsForBig + startPointForBig * 3;
		gStartVelocities = startVelocitiesForBig + startPointForBig * 3;
		gEndVelocities = endVelocitiesForBig + startPointForBig * 3;

		// Initialize connectivities and links
		gConnectivities = blockedLocalConnectivities + (startCell << 2);
		gLinks = blockedLocalLinks + (startCell << 2);
	}

	for (int i = localID; i < numOfPoints * 3; i += numOfThreads) {
		int localPointID = i / 3;
		int dimensionID = i % 3;
		int globalPointID = blockedGlobalPointIDs[startPoint + localPointID];

		if (canFit) {
			vertexPositions[i] = globalVertexPositions[globalPointID * 3 + dimensionID];
			startVelocities[i] = globalStartVelocities[globalPointID * 3 + dimensionID];
			endVelocities[i] = globalEndVelocities[globalPointID * 3 + dimensionID];
		} else {
			/*gVertexPositions[i] = gliobalVertexPositions[globalPointID * 3 + dimensionID];
			gStartVelocities[i] = globalStartVelocities[globalPointID * 3 + dimensionID];
			gEndVelocities[i] = globalEndVelocities[globalPointID * 3 + dimensionID];*/
		}
	}

	if (canFit)
		for (int i = localID; i < (numOfCells << 2); i += numOfThreads) {
			connectivities[i] = *(blockedLocalConnectivities + (startCell << 2) + i);
			links[i] = *(blockedLocalLinks + (startCell << 2) + i);
		}

	if (canFit)
		barrier(CLK_LOCAL_MEM_FENCE);
	else
		barrier(CLK_GLOBAL_MEM_FENCE);
	
	int numOfActiveParticles = startOffsetInParticle[activeBlockID + 1] - startOffsetInParticle[activeBlockID];

	for (int idx = localID; idx < numOfActiveParticles; idx += numOfThreads) {
		// activeParticleID here means the initial active particle ID
		int arrayIdx = startOffsetInParticle[activeBlockID] + idx;
		int activeParticleID = blockedActiveParticleIDList[arrayIdx];

		// Initialize the particle status
		int currStage = stage[activeParticleID];
		int currCell = blockedCellLocationList[startOffsetInParticle[activeBlockID] + idx];

		double currTime = pastTimes[activeParticleID];

		double currLastPosition[3];
		currLastPosition[0] = lastPosition[activeParticleID * 3];
		currLastPosition[1] = lastPosition[activeParticleID * 3 + 1];
		currLastPosition[2] = lastPosition[activeParticleID * 3 + 2];
		double currK1[3], currK2[3], currK3[3], currK4[3];
		if (currStage > 0) {
			currK1[0] = k1[activeParticleID * 3];
			currK1[1] = k1[activeParticleID * 3 + 1];
			currK1[2] = k1[activeParticleID * 3 + 2];
		}
		if (currStage > 1) {
			currK2[0] = k2[activeParticleID * 3];
			currK2[1] = k2[activeParticleID * 3 + 1];
			currK2[2] = k2[activeParticleID * 3 + 2];
		}
		if (currStage > 2) {
			currK3[0] = k3[activeParticleID * 3];
			currK3[1] = k3[activeParticleID * 3 + 1];
			currK3[2] = k3[activeParticleID * 3 + 2];
		}

		// At least one loop is executed.
		while (true) {
			double placeOfInterest[3];
			placeOfInterest[0] = currLastPosition[0];
			placeOfInterest[1] = currLastPosition[1];
			placeOfInterest[2] = currLastPosition[2];
			switch (currStage) {
			case 1: {
				placeOfInterest[0] += 0.5 * currK1[0];
				placeOfInterest[1] += 0.5 * currK1[1];
				placeOfInterest[2] += 0.5 * currK1[2];
					} break;
			case 2: {
				placeOfInterest[0] += 0.5 * currK2[0];
				placeOfInterest[1] += 0.5 * currK2[1];
				placeOfInterest[2] += 0.5 * currK2[2];
					} break;
			case 3: {
				placeOfInterest[0] += currK3[0];
				placeOfInterest[1] += currK3[1];
				placeOfInterest[2] += currK3[2];
					} break;
			}

			double coordinates[4];

			int nextCell;
			
			if (canFit)
				nextCell = FindCell(placeOfInterest, connectivities, links, vertexPositions, epsilon, currCell, coordinates);
			else
				nextCell = gFindCell(placeOfInterest, gConnectivities, gLinks, gVertexPositions, epsilon, currCell, coordinates);

			if (nextCell == -1 || currTime >= endTime) {
				// Find the next cell globally
				int globalCellID = blockedGlobalCellIDs[startCell + currCell];
				int nextGlobalCell;
				
				if (nextCell != -1)
					nextGlobalCell = blockedGlobalCellIDs[startCell + nextCell];
				else
					nextGlobalCell = gFindCell(placeOfInterest, globalTetrahedralConnectivities, globalTetrahedralLinks,
											   globalVertexPositions, epsilon, globalCellID, coordinates);

				if (currTime >= endTime && nextGlobalCell != -1) nextGlobalCell = -2 - nextGlobalCell;

				pastTimes[activeParticleID] = currTime;

				stage[activeParticleID] = currStage;

				lastPosition[activeParticleID * 3] = currLastPosition[0];
				lastPosition[activeParticleID * 3 + 1] = currLastPosition[1];
				lastPosition[activeParticleID * 3 + 2] = currLastPosition[2];
		
				if (currStage > 0) {
					k1[activeParticleID * 3] = currK1[0];
					k1[activeParticleID * 3 + 1] = currK1[1];
					k1[activeParticleID * 3 + 2] = currK1[2];
				}
				if (currStage > 1) {
					k2[activeParticleID * 3] = currK2[0];
					k2[activeParticleID * 3 + 1] = currK2[1];
					k2[activeParticleID * 3 + 2] = currK2[2];
				}
				if (currStage > 2) {
					k3[activeParticleID * 3] = currK3[0];
					k3[activeParticleID * 3 + 1] = currK3[1];
					k3[activeParticleID * 3 + 2] = currK3[2];
				}

				// Write squeezed arrays
				squeezedStage[arrayIdx] = currStage;
				squeezedExitCells[arrayIdx] = nextGlobalCell;

				squeezedLastPosition[arrayIdx * 3] = currLastPosition[0];
				squeezedLastPosition[arrayIdx * 3 + 1] = currLastPosition[1];
				squeezedLastPosition[arrayIdx * 3 + 2] = currLastPosition[2];
		
				if (currStage > 0) {
					squeezedK1[arrayIdx * 3] = currK1[0];
					squeezedK1[arrayIdx * 3 + 1] = currK1[1];
					squeezedK1[arrayIdx * 3 + 2] = currK1[2];
				}
				if (currStage > 1) {
					squeezedK2[arrayIdx * 3] = currK2[0];
					squeezedK2[arrayIdx * 3 + 1] = currK2[1];
					squeezedK2[arrayIdx * 3 + 2] = currK2[2];
				}
				if (currStage > 2) {
					squeezedK3[arrayIdx * 3] = currK3[0];
					squeezedK3[arrayIdx * 3 + 1] = currK3[1];
					squeezedK3[arrayIdx * 3 + 2] = currK3[2];
				}

				break;
			}

			currCell = nextCell;

			double alpha = (endTime - currTime) / (endTime - startTime);
			double beta = 1 - alpha;

			double vecX[4], vecY[4], vecZ[4];

			for (int i = 0; i < 4; i++)
				if (canFit) {
					int pointID = connectivities[(nextCell << 2) | i];
					vecX[i] = startVelocities[pointID * 3] * alpha + endVelocities[pointID * 3] * beta;
					vecY[i] = startVelocities[pointID * 3 + 1] * alpha + endVelocities[pointID * 3 + 1] * beta;
					vecZ[i] = startVelocities[pointID * 3 + 2] * alpha + endVelocities[pointID * 3 + 2] * beta;
				} else {
					int pointID = gConnectivities[(nextCell << 2) | i];
					vecX[i] = gStartVelocities[pointID * 3] * alpha + gEndVelocities[pointID * 3] * beta;
					vecY[i] = gStartVelocities[pointID * 3 + 1] * alpha + gEndVelocities[pointID * 3 + 1] * beta;
					vecZ[i] = gStartVelocities[pointID * 3 + 2] * alpha + gEndVelocities[pointID * 3 + 2] * beta;
				}

			double *currK;
			switch (currStage) {
			case 0: currK = currK1; break;
			case 1: currK = currK2; break;
			case 2: currK = currK3; break;
			case 3: currK = currK4; break;
			}

			currK[0] = currK[1] = currK[2] = 0;

			for (int i = 0; i < 4; i++) {
				currK[0] += vecX[i] * coordinates[i];
				currK[1] += vecY[i] * coordinates[i];
				currK[2] += vecZ[i] * coordinates[i];
			}

			currK[0] *= timeStep;
			currK[1] *= timeStep;
			currK[2] *= timeStep;

			if (currStage == 3) {
				currTime += timeStep;

				for (int i = 0; i < 3; i++)
					currLastPosition[i] += (currK1[i] + 2 * currK2[i] + 2 * currK3[i] + currK4[i]) / 6;

				currStage = 0;
			} else
				currStage++;
		}
	}
}	
