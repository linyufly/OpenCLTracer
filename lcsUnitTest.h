/**********************************************
File			:		lcsUnitTest.h
Author			:		Mingcheng Chen
Last Update		:		June 3rd, 2012
***********************************************/

#ifndef __LCS_UNIT_TEST_H
#define __LCS_UNIT_TEST_H

#include "lcsGeometry.h"

namespace lcs {

void UnitTestForTetBlkIntersection(lcs::TetrahedralGrid *grid, double blockSize,
								   double globalMinX, double globalMinY, double globalMinZ,
								   int numOfBlocksInY, int numOfBlocksInZ,
								   int *queryTetrahedron, int *queryBlock,
								   char *queryResults,
								   int numOfQueries,
								   double epsilon);

void UnitTestForInitialCellLocations(lcs::TetrahedralGrid *grid,
									 int xRes, int yRes, int zRes,
									 double minX, double minY, double minZ,
									 double dx, double dy, double dz,
									 int *initialCellLocations,
									 double epsilon);

}

#endif