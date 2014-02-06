/******************************************************************
File		:	lcsBigBlockInitializationForPositions.cl
Author		:	Mingcheng Chen
Last Update	:	August 28th, 2012
*******************************************************************/

__kernel void BigBlockInitializationForPositions(__global double *globalVertexPositions,
			
						 __global int *blockedGlobalPointIDs,

   						 __global int *startOffsetInPoint,

						 __global int *startOffsetInPointForBig,
						 __global double *vertexPositionsForBig,

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
	__global double *gVertexPositions;
		
	int startPoint = startOffsetInPoint[interestingBlockID];

	int numOfPoints = startOffsetInPoint[interestingBlockID + 1] - startPoint;

	int startPointForBig = startOffsetInPointForBig[interestingBlockID];

	// Initialize vertexPositions
	gVertexPositions = vertexPositionsForBig + startPointForBig * 3;

	for (int i = localID; i < numOfPoints * 3; i += numOfThreads) {
		int localPointID = i / 3;
		int dimensionID = i % 3;
		int globalPointID = blockedGlobalPointIDs[startPoint + localPointID];

		gVertexPositions[i] = globalVertexPositions[globalPointID * 3 + dimensionID];
	}
}	
