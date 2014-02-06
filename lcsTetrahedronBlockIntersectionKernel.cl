/******************************************************************
File			:		lcsTetrahedronBlockIntersectionKernel.cl
Author			:		Mingcheng Chen
Last Update		:		June 1st, 2012
*******************************************************************/

inline double VectorLength(double x, double y, double z) {
	return sqrt(x * x + y * y + z * z);
}

inline void CrossProductThree(double x1, double y1, double z1, double x2, double y2, double z2,
							  double *x, double *y, double *z) {
	*x = y1 * z2 - y2 * z1;
	*y = z1 * x2 - z2 * x1;
	*z = x1 * y2 - x2 * y1;
}

inline double DeterminantThree(double *a) {
	// a[0] a[1] a[2]
	// a[3] a[4] a[5]
	// a[6] a[7] a[8]
	return a[0] * a[4] * a[8] + a[1] * a[5] * a[6] + a[2] * a[3] * a[7] -
		   a[0] * a[5] * a[7] - a[1] * a[3] * a[8] - a[2] * a[4] * a[6];
}

inline double DirectedVolume(double x, double y, double z,
							 double x1, double y1, double z1,
							 double x2, double y2, double z2,
							 double x3, double y3, double z3) {
	double det[9] = {x1 - x, y1 - y, z1 - z,
					 x2 - x, y2 - y, z2 - z,
					 x3 - x, y3 - y, z3 - z};
	return DeterminantThree(det);
}

inline void GetBlockPoint(int num, int *x, int *y, int *z) {
	*x = !!(num & 4);
	*y = !!(num & 2);
	*z = num & 1;
}

inline void GetBlockEdge(int num, int *x1, int *y1, int *z1, int *x2, int *y2, int *z2) {
	switch (num / 3) {
	case 0: {
				*x1 = *y1 = *z1 = 0;
			} break;
	case 1: {
				*x1 = *y1 = 1;
				*z1 = 0;
			} break;
	case 2: {
				*x1 = *z1 = 1;
				*y1 = 0;
			} break;
	case 3: {
				*y1 = *z1 = 1;
				*x1 = 0;
			} break;
	}
	
	*x2 = *x1;
	*y2 = *y1;
	*z2 = *z1;

	switch (num % 3) {
	case 0: *x2 = 1 - *x2; break;
	case 1: *y2 = 1 - *y2; break;
	case 2: *z2 = 1 - *z2; break;
	}
}

inline void GetTetrahedralEdge(int num, int *id1, int *id2) {
	switch (num) {
		case 0: {
					*id1 = 0;
					*id2 = 1;
				} break;
		case 1: {
					*id1 = 0;
					*id2 = 2;
				} break;
		case 2: {
					*id1 = 0;
					*id2 = 3;
				} break;
		case 3: {
					*id1 = 1;
					*id2 = 2;
				} break;
		case 4: {
					*id1 = 1;
					*id2 = 3;
				} break;
		case 5: {
					*id1 = 2;
					*id2 = 3;
				} break;
	}
}

inline int Sign(double a, double epsilon) {
	return a < -epsilon ? -1 : a > epsilon;
}

inline bool CheckPlane(double x1, double y1, double z1,
					  double x2, double y2, double z2,
					  double x3, double y3, double z3,
					  double *tetX, double *tetY, double *tetZ,
					  double minX, double minY, double minZ,
					  double blockSize,
					  double epsilon) {
	double x, y, z;
	CrossProductThree(x2 - x1, y2 - y1, z2 - z1, x3 - x1, y3 - y1, z3 - z1, &x, &y, &z);
	if (!Sign(VectorLength(x, y, z), epsilon)) return 0;

	char tetPos = 0, tetNeg = 0;
	char blkPos = 0, blkNeg = 0;

	// Check tetrahedral points
	for (int i = 0; i < 4; i++) {
		double directedVolume = DirectedVolume(tetX[i], tetY[i], tetZ[i],
											   x1, y1, z1, x2, y2, z2, x3, y3, z3);
		int sign = Sign(directedVolume, epsilon);
		if (sign > 0) tetPos = 1;
		if (sign < 0) tetNeg = 1;
		if (tetPos * tetNeg) return 0;
	}

	// Check block points
	for (int dx = 0; dx <= 1; dx++)
		for (int dy = 0; dy <= 1; dy++)
			for (int dz = 0; dz <= 1; dz++) {
				x = minX + blockSize * dx;
				y = minY + blockSize * dy;
				z = minZ + blockSize * dz;
				double directedVolume = DirectedVolume(x, y, z,
													   x1, y1, z1, x2, y2, z2, x3, y3, z3);
				int sign = Sign(directedVolume, epsilon);
				if (sign > 0) blkPos = 1;
				if (sign < 0) blkNeg = 1;
				if (blkPos * blkNeg) return 0;
			}

	// Final Check
	if (tetPos && blkPos || tetNeg && blkNeg) return 0;
	return 1;
}

__kernel void TetrahedronBlockIntersection(__global double *vertexPositions,
										   __global int *tetrahedralConnectivities,
										   __global int *queryTetrahedron,
										   __global int *queryBlock,
										   __global char *queryResult,
										   int numOfBlocksInY, int numOfBlocksInZ,
										   double globalMinX, double globalMinY, double globalMinZ,
										   double blockSize,
										   double epsilon,
										   int numOfQueries
										   // __global double *tempValue
										   ) {
	// Get global ID
	int globalID = get_global_id(0);

	// Only use first "numOfQueries" threads
	if (globalID < numOfQueries) {
		int tetrahedronID = queryTetrahedron[globalID];
		int blockID = queryBlock[globalID];

		int tetPoint1 = tetrahedralConnectivities[tetrahedronID << 2];
		int tetPoint2 = tetrahedralConnectivities[(tetrahedronID << 2) + 1];
		int tetPoint3 = tetrahedralConnectivities[(tetrahedronID << 2) + 2];
		int tetPoint4 = tetrahedralConnectivities[(tetrahedronID << 2) + 3];

		double tetX[4], tetY[4], tetZ[4];

		tetX[0] = vertexPositions[tetPoint1 * 3];
		tetY[0] = vertexPositions[tetPoint1 * 3 + 1];
		tetZ[0] = vertexPositions[tetPoint1 * 3 + 2];

		tetX[1] = vertexPositions[tetPoint2 * 3];
		tetY[1] = vertexPositions[tetPoint2 * 3 + 1];
		tetZ[1] = vertexPositions[tetPoint2 * 3 + 2];

		tetX[2] = vertexPositions[tetPoint3 * 3];
		tetY[2] = vertexPositions[tetPoint3 * 3 + 1];
		tetZ[2] = vertexPositions[tetPoint3 * 3 + 2];

		tetX[3] = vertexPositions[tetPoint4 * 3];
		tetY[3] = vertexPositions[tetPoint4 * 3 + 1];
		tetZ[3] = vertexPositions[tetPoint4 * 3 + 2];

		int zIdx = blockID % numOfBlocksInZ;
		int temp = blockID / numOfBlocksInZ;
		int yIdx = temp % numOfBlocksInY;
		int xIdx = temp / numOfBlocksInY;

		double localMinX = globalMinX + xIdx * blockSize;
		double localMinY = globalMinY + yIdx * blockSize;
		double localMinZ = globalMinZ + zIdx * blockSize;

		char result = 0;

		// Test tetrahedral point and block edge
		for (int i = 0; !result && i < 4; i++) {
			double x1 = tetX[i];
			double y1 = tetY[i];
			double z1 = tetZ[i];
			for (int j = 0; j < 12; j++) {
				int dx1, dy1, dz1, dx2, dy2, dz2;
				GetBlockEdge(j, &dx1, &dy1, &dz1, &dx2, &dy2, &dz2);

				double x2 = localMinX + dx1 * blockSize;
				double y2 = localMinY + dy1 * blockSize;
				double z2 = localMinZ + dz1 * blockSize;

				double x3 = localMinX + dx2 * blockSize;
				double y3 = localMinY + dy2 * blockSize;
				double z3 = localMinZ + dz2 * blockSize;
				
				if (CheckPlane(x1, y1, z1, x2, y2, z2, x3, y3, z3,
							   tetX, tetY, tetZ, localMinX, localMinY, localMinZ,
							   blockSize, epsilon)) {
					result = 1;
					break;
				}
			}
		}

		// Test tetrahedral edge and block point
		for (int i = 0; !result && i < 6; i++) {
			int id1, id2;
			GetTetrahedralEdge(i, &id1, &id2);

			double x1 = tetX[id1];
			double y1 = tetY[id1];
			double z1 = tetZ[id1];

			double x2 = tetX[id2];
			double y2 = tetY[id2];
			double z2 = tetZ[id2];

			for (int j = 0; j < 8; j++) {
				int dx, dy, dz;
				GetBlockPoint(j, &dx, &dy, &dz);

				double x3 = localMinX + dx * blockSize;
				double y3 = localMinY + dy * blockSize;
				double z3 = localMinZ + dz * blockSize;

				if (CheckPlane(x1, y1, z1, x2, y2, z2, x3, y3, z3,
							   tetX, tetY, tetZ, localMinX, localMinY, localMinZ,
							   blockSize, epsilon)) {
					result = 1;
					break;
				}
			}
		}

		queryResult[globalID] = !result;
	}
}