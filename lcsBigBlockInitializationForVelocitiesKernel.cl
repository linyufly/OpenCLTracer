/******************************************************************
File		:	lcsBigBlockInitializationForVelocities.cl
Author		:	Mingcheng Chen
Last Update	:	August 29th, 2012
*******************************************************************/

__kernel void BigBlockInitializationForVelocities(__global double *globalStartVelocities,
			     			  __global double *globalEndVelocities,
			
			     			  __global int *blockedGlobalPointIDs,

   			     			  __global int *startOffsetInPoint,

			     			  __global int *startOffsetInPointForBig,
			     			  __global double *startVelocitiesForBig,
			     			  __global double *endVelocitiesForBig,

			     			  __global int *bigBlocks
			     			  ) {
	// Get work group ID
	int workGroupID = get_group_id(0);
	
	// Get number of threads in a work group
	int numOfThreads = get_local_size(0);

	// Get local thread ID
	int localID = get_local_id(0);

	// Get interesting block ID of the current big block
	int interestingBlockID = bigBlocks[workGroupID];

	// Declare some work arrays
	__global double *gStartVelocities;
	__global double *gEndVelocities;
		
	int startPoint = startOffsetInPoint[interestingBlockID];

	int numOfPoints = startOffsetInPoint[interestingBlockID + 1] - startPoint;

	int startPointForBig = startOffsetInPointForBig[interestingBlockID];

	// Initialize startVelocities and endVelocities
	gStartVelocities = startVelocitiesForBig + startPointForBig * 3;
	gEndVelocities = endVelocitiesForBig + startPointForBig * 3;

	for (int i = localID; i < numOfPoints * 3; i += numOfThreads) {
		int localPointID = i / 3;
		int dimensionID = i % 3;
		int globalPointID = blockedGlobalPointIDs[startPoint + localPointID];

		gStartVelocities[i] = globalStartVelocities[globalPointID * 3 + dimensionID];
		gEndVelocities[i] = globalEndVelocities[globalPointID * 3 + dimensionID];
	}
}	
