/**********************************************
File			:		lcsUnitTest.cpp
Author			:		Mingcheng Chen
Last Update		:		June 3rd, 2012
***********************************************/

#include "lcsUnitTest.h"
#include "lcsUtility.h"

////////////////////////////////////////////////
bool CheckPlane(const lcs::Vector &p1, const lcs::Vector &p2, const lcs::Vector &p3,
				const lcs::Tetrahedron &tetrahedron,
				double localMinX, double localMinY, double localMinZ,
				double blockSize, double epsilon) {
	if (!lcs::Sign(Cross(p1 - p2, p1 - p3).Length(), epsilon)) return false;
	bool tetPos = 0, tetNeg = 0, blkPos = 0, blkNeg = 0;
	for (int i = 0; i < 4; i++) {
		lcs::Vector point = tetrahedron.GetVertex(i);
		double directedVolume = lcs::Mixed(p1 - point, p2 - point, p3 - point);
		if (lcs::Sign(directedVolume, epsilon) < 0) tetNeg = 1;
		if (lcs::Sign(directedVolume, epsilon) > 0) tetPos = 1;
		if (tetNeg * tetPos) return false;
	}
	for (int i = 0; i < 8; i++) {
		lcs::Vector point(localMinX, localMinY, localMinZ);
		if (i & 1) point.SetX(point.GetX() + blockSize);
		if (i & 2) point.SetY(point.GetY() + blockSize);
		if (i & 4) point.SetZ(point.GetZ() + blockSize);
		double directedVolume = lcs::Mixed(p1 - point, p2 - point, p3 - point);
		if (lcs::Sign(directedVolume, epsilon) < 0) blkNeg = 1;
		if (lcs::Sign(directedVolume, epsilon) > 0) blkPos = 1;
		if (blkNeg * blkPos) return false;
	}
	if (tetNeg && blkNeg || tetPos && blkPos) return false;
	return true;
}

bool TetrahedronContainsPoint(const lcs::Tetrahedron &tet, const lcs::Vector &pt, double eps) {
	double coordinates[4];
	tet.CalculateNaturalCoordinates(pt, coordinates);
	for (int i = 0; i < 4; i++)
		if (lcs::Sign(coordinates[i], eps) < 0) return false;
	return true;
}

////////////////////////////////////////////////
void lcs::UnitTestForTetBlkIntersection(lcs::TetrahedralGrid *grid, double blockSize,
								   double globalMinX, double globalMinY, double globalMinZ,
								   int numOfBlocksInY, int numOfBlocksInZ,
								   int *queryTetrahedron, int *queryBlock,
								   char *queryResults,
								   int numOfQueries,
								   double epsilon) {
	printf("Unit test for tetrahedron-block intersection ... ");

	for (int i = 0; i < numOfQueries; i++) {
		int tetID = queryTetrahedron[i];
		int blkID = queryBlock[i];
		lcs::Tetrahedron tet = grid->GetTetrahedron(tetID);
		int x, y, z;
		z = blkID % numOfBlocksInZ;
		int temp = blkID / numOfBlocksInZ;
		y = temp % numOfBlocksInY;
		x = temp / numOfBlocksInY;

		double localMinX = globalMinX + x * blockSize;
		double localMinY = globalMinY + y * blockSize;
		double localMinZ = globalMinZ + z * blockSize;

		// Test tetrahedral edge and block point
		bool flag = 0;

		for (int tetEdgeID = 0; !flag && tetEdgeID < 6; tetEdgeID++) {
			Vector p1, p2;
			switch (tetEdgeID) {
			case 0: {
						p1 = tet.GetVertex(0);
						p2 = tet.GetVertex(1);
					} break;
			case 1: {
						p1 = tet.GetVertex(0);
						p2 = tet.GetVertex(2);
					} break;
			case 2: {
						p1 = tet.GetVertex(0);
						p2 = tet.GetVertex(3);
					} break;
			case 3: {
						p1 = tet.GetVertex(1);
						p2 = tet.GetVertex(2);
					} break;
			case 4: {
						p1 = tet.GetVertex(1);
						p2 = tet.GetVertex(3);
					} break;
			case 5: {
						p1 = tet.GetVertex(2);
						p2 = tet.GetVertex(3);
					} break;
			}
			for (int dx = 0; !flag && dx <= 1; dx++)
				for (int dy = 0; !flag && dy <= 1; dy++)
					for (int dz = 0; dz <= 1; dz++) {
						Vector p3 = Vector(localMinX, localMinY, localMinZ) + Vector(dx, dy, dz) * blockSize;
						if (CheckPlane(p1, p2, p3, tet, localMinX, localMinY, localMinZ, blockSize, epsilon)) {
							flag = 1;

							//printf("tetrahedral edge and block point: %d, %d %d %d\n", tetEdgeID, dx, dy, dz);

							break;
						}
					}
		}

		// Test tetrahedral point and block edge
		for (int x1 = 0; !flag && x1 <= 1; x1++)
			for (int y1 = 0; !flag && y1 <= 1; y1++)
				for (int z1 = 0; !flag && z1 <= 1; z1++) {
					Vector p1(localMinX + x1 * blockSize, localMinY + y1 * blockSize, localMinZ + z1 *blockSize);
					for (int k = 0; !flag && k < 3; k++) {
						int x2 = x1, y2 = y1, z2 = z1;
						if (k == 0)
							if (x1 == 0) x2++;
							else continue;
						if (k == 1)
							if (y1 == 0) y2++;
							else continue;
						if (k == 2)
							if (z1 == 0) z2++;
							else continue;
						Vector p2(localMinX + x2 * blockSize, localMinY + y2 * blockSize, localMinZ + z2 * blockSize);
						for (int j = 0; j < 4; j++) {
							Vector p3 = tet.GetVertex(j);
							if (CheckPlane(p1, p2, p3, tet, localMinX, localMinY, localMinZ, blockSize, epsilon)) {
								flag = 1;
								break;
							}
						}
					}
				}

		bool result = !flag;

		if (result != queryResults[i]) {
			char error[100];
			sprintf(error, "Query %d has incorrect result.\ntet = %d, blk = %d\nkernel result: %d, CPU result: %d",
					i + 1, queryTetrahedron[i], queryBlock[i], queryResults[i], result);
			lcs::Error(error);
		}
	}

	printf("Passed\n");
}

void lcs::UnitTestForInitialCellLocations(lcs::TetrahedralGrid *grid,
										  int xRes, int yRes, int zRes,
										  double minX, double minY, double minZ,
										  double dx, double dy, double dz,
										  int *initialCellLocations,
										  double epsilon) {
	printf("Unit test for initial cell location ");

	int numOfGridPoints = (xRes + 1) * (yRes + 1) * (zRes + 1);
	int numOfMinusOnes = 0, numOfLocations = 0;
	for (int i = 0; i < numOfGridPoints; i++)
		if (initialCellLocations[i] == -1) numOfMinusOnes++;
		else numOfLocations++;
	printf("(outside: %d, inside: %d) ... ", numOfMinusOnes, numOfLocations);

	int idx = -1;
	for (int i = 0; i <= xRes; i++)
		for (int j = 0; j <= yRes; j++)
			for (int k = 0; k <= zRes; k++) {
				idx++;

				lcs::Vector point(minX + dx * i, minY + dy * j, minZ + dz * k);

				if (initialCellLocations[idx] != -1) {
					lcs::Tetrahedron tet = grid->GetTetrahedron(initialCellLocations[idx]);
					if (!TetrahedronContainsPoint(tet, point, epsilon)) {
						char str[100];
						sprintf(str, "Tetrahedron %d does not contain grid point %d (%d, %d, %d)",
								initialCellLocations[idx] + 1, idx, i, j, k);
						lcs::Error(str);
					}
					continue;
				}

				for (int tetID = 0; tetID < grid->GetNumOfCells(); tetID++) {
					lcs::Tetrahedron tet = grid->GetTetrahedron(tetID);
					if (TetrahedronContainsPoint(tet, point, epsilon)) {
						char str[100];
						sprintf(str, "Tetrahedron %d contains grid point %d (%d, %d, %d)",
								tetID, idx, i, j, k);
						lcs::Error(str);
					}
				}
			}

	printf("Passed\n");
}