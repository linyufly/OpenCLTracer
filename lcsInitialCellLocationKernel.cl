/******************************************************************
File			:		lcsInitialCellLocationKernel.cl
Author			:		Mingcheng Chen
Last Update		:		June 3rd, 2012
*******************************************************************/

inline int Sign(double a, double epsilon) {
	return a < -epsilon ? -1 : a > epsilon;
}

inline double DeterminantThree(double *a) {
	// a[0] a[1] a[2]
	// a[3] a[4] a[5]
	// a[6] a[7] a[8]
	return a[0] * a[4] * a[8] + a[1] * a[5] * a[6] + a[2] * a[3] * a[7] -
		   a[0] * a[5] * a[7] - a[1] * a[3] * a[8] - a[2] * a[4] * a[6];
}

inline bool Inside(double X, double Y, double Z, double *tetX, double *tetY, double *tetZ, double epsilon) {
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

	double coordinate1 = a11 * X + a12 * Y + a13 * Z;
	if (Sign(coordinate1, epsilon) < 0) return false;

	double y12 = tetY[0] - tetY[1];
	double z12 = tetZ[0] - tetZ[1];
	double a21 = (z41 * y12 - z12 * y41) * V;

	double x12 = tetX[0] - tetX[1];
	double a22 = (x41 * z12 - x12 * z41) * V;

	double a23 = (y41 * x12 - y12 * x41) * V;

	double coordinate2 = a21 * X + a22 * Y + a23 * Z;
	if (Sign(coordinate2, epsilon) < 0) return false;

	double z23 = tetZ[1] - tetZ[2];
	double y23 = tetY[1] - tetY[2];
	double a31 = (z23 * y12 - z12 * y23) * V;

	double x23 = tetX[1] - tetX[2];
	double a32 = (x23 * z12 - x12 * z23) * V;

	double a33 = (y23 * x12 - y12 * x23) * V;

	double coordinate3 = a31 * X + a32 * Y + a33 * Z;
	if (Sign(coordinate3, epsilon) < 0) return false;

	double coordinate0 = 1 - coordinate1 - coordinate2 - coordinate3;
	if (Sign(coordinate0, epsilon) < 0) return false;
	return true;
}

__kernel void InitialCellLocation(__global double *vertexPositions,
								  __global int *tetrahedralConnectivities,
								  volatile __global int *cellLocations,
								  int xRes, int yRes, int zRes,
								  double minX, double minY, double minZ,
								  double dx, double dy, double dz,
								  double epsilon,
								  int numOfCells
								  ) {
	// Get global ID and global size
	int globalID = get_global_id(0);
	int globalSize = get_global_size(0);

	// Only use first "numOfCells" threads
	if (globalID < numOfCells) {
		int point1 = tetrahedralConnectivities[globalID << 2];
		int point2 = tetrahedralConnectivities[(globalID << 2) + 1];
		int point3 = tetrahedralConnectivities[(globalID << 2) + 2];
		int point4 = tetrahedralConnectivities[(globalID << 2) + 3];

		double tetX[4], tetY[4], tetZ[4];

		tetX[0] = vertexPositions[point1 * 3];
		tetY[0] = vertexPositions[point1 * 3 + 1];
		tetZ[0] = vertexPositions[point1 * 3 + 2];

		tetX[1] = vertexPositions[point2 * 3];
		tetY[1] = vertexPositions[point2 * 3 + 1];
		tetZ[1] = vertexPositions[point2 * 3 + 2];

		tetX[2] = vertexPositions[point3 * 3];
		tetY[2] = vertexPositions[point3 * 3 + 1];
		tetZ[2] = vertexPositions[point3 * 3 + 2];

		tetX[3] = vertexPositions[point4 * 3];
		tetY[3] = vertexPositions[point4 * 3 + 1];
		tetZ[3] = vertexPositions[point4 * 3 + 2];

		double tetMinX = min(min(tetX[0], tetX[1]), min(tetX[2], tetX[3]));
		double tetMaxX = max(max(tetX[0], tetX[1]), max(tetX[2], tetX[3]));

		double tetMinY = min(min(tetY[0], tetY[1]), min(tetY[2], tetY[3]));
		double tetMaxY = max(max(tetY[0], tetY[1]), max(tetY[2], tetY[3]));

		double tetMinZ = min(min(tetZ[0], tetZ[1]), min(tetZ[2], tetZ[3]));
		double tetMaxZ = max(max(tetZ[0], tetZ[1]), max(tetZ[2], tetZ[3]));

		if (Sign(tetMaxX - minX, epsilon) < 0) return;
		if (Sign(tetMaxY - minY, epsilon) < 0) return;
		if (Sign(tetMaxZ - minZ, epsilon) < 0) return;

		double maxX = minX + dx * xRes;
		double maxY = minY + dy * yRes;
		double maxZ = minZ + dz * zRes;

		if (Sign(tetMinX - maxX, epsilon) > 0) return;
		if (Sign(tetMinY - maxY, epsilon) > 0) return;
		if (Sign(tetMinZ - maxZ, epsilon) > 0) return;

		int xStart = 0, yStart = 0, zStart = 0;
		if (tetMinX > minX) xStart = (int)((tetMinX - minX) / dx);
		if (tetMinY > minY) yStart = (int)((tetMinY - minY) / dy);
		if (tetMinZ > minZ) zStart = (int)((tetMinZ - minZ) / dz);

		int xFinish = xRes, yFinish = yRes, zFinish = zRes;
		if (tetMaxX < maxX) xFinish = min((int)((tetMaxX - minX) / dx) + 1, xRes);
		if (tetMaxY < maxY) yFinish = min((int)((tetMaxY - minY) / dy) + 1, yRes);
		if (tetMaxZ < maxZ) zFinish = min((int)((tetMaxZ - minZ) / dz) + 1, zRes);

		int numOfCandidates = (xFinish - xStart + 1) * (yFinish - yStart + 1) * (zFinish - zStart + 1);

		//for (int xIdx = xStart; xIdx <= xFinish; xIdx++)
		//	for (int yIdx = yStart; yIdx <= yFinish; yIdx++)
		//		for (int zIdx = zStart; zIdx <= zFinish; zIdx++)

		for (int i = 0; i < numOfCandidates; i++) {
			int zIdx = i % (zFinish - zStart + 1) + zStart;
			int temp = i / (zFinish - zStart + 1);
			int yIdx = temp % (yFinish - yStart + 1) + yStart;
			int xIdx = temp / (yFinish - yStart + 1) + xStart;
			double X = minX + dx * xIdx;
			double Y = minY + dy * yIdx;
			double Z = minZ + dz * zIdx;
			if (Inside(X, Y, Z, tetX, tetY, tetZ, epsilon)) {
				int index = (xIdx * (yRes + 1) + yIdx) * (zRes + 1) + zIdx;
				int oldValue = atomic_add(cellLocations + index, globalID + 1);
				if (oldValue != -1) atomic_add(cellLocations + index, -globalID - 1);
			}
		}
	}
}