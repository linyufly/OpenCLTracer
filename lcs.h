/**********************************************
File			:		lcs.h
Author			:		Mingcheng Chen
Last Update		:		June 7th, 2012
***********************************************/

#ifndef __LCS_H
#define __LCS_H

#include "lcsGeometry.h"

namespace lcs {

class Frame {
public:
	Frame(double timePoint, const char *dataFile);

	~Frame();

	TetrahedralGrid *GetTetrahedralGrid() const {
		return tetrahedralGrid;
	}

	double GetTimePoint() const {
		return timePoint;
	}

private:
	TetrahedralGrid *tetrahedralGrid;
	double timePoint;
};

class ParticleRecord {
public:
	static const int FE = 0;
	static const int RK4 = 1;
	static const int RK45 = 2;

	ParticleRecord();
	ParticleRecord(int stage, int gridPointID, void *data);
	~ParticleRecord();

	static void SetDataType(int type);
	static int GetDataType();

	lcs::Vector GetPositionInInterest() const;
	int GetStage() const;
	int GetGridPointID() const;
	void *GetData() const;

	void SetStage(int stage);

private:
	static int dataType;
	int stage; // It may be used for sorting in order to reduce GPU thread divergency in a warp.
	int gridPointID;
	void *data;
};

class ParticleRecordDataForRK4 {
public:
	static const int COMPUTING_K1 = 0;
	static const int COMPUTING_K2 = 1;
	static const int COMPUTING_K3 = 2;
	static const int COMPUTING_K4 = 3;

	void SetLastPosition(const lcs::Vector &lastPosition);
	void SetK1(const lcs::Vector &k1);
	void SetK2(const lcs::Vector &k2);
	void SetK3(const lcs::Vector &k3);

	void SetLastPosition(const double *);
	void SetK1(const double *);
	void SetK2(const double *);
	void SetK3(const double *);

	void SetLastPosition(const float *);
	void SetK1(const float *);
	void SetK2(const float *);
	void SetK3(const float *);

	lcs::Vector GetLastPosition() const;
	lcs::Vector GetK1() const;
	lcs::Vector GetK2() const;
	lcs::Vector GetK3() const;

private:
	lcs::Vector lastPosition, k1, k2, k3;
};

class BlockRecord {
public:
	BlockRecord();
	~BlockRecord();

	int GetLocalNumOfCells() const;
	int GetLocalNumOfPoints() const;

	void SetLocalNumOfCells(int localNumOfCells);
	void SetLocalNumOfPoints(int localNumOfPoints);

	void CreateGlobalCellIDs(int *cellIDs);
	void CreateGlobalPointIDs(int *pointIDs);
	void CreateLocalConnectivities(int *connectivities);
	void CreateLocalLinks(int *links);

	int EvaluateNumOfBytes() const;

	int GetGlobalCellID(int localCellID) const;
	int GetGlobalPointID(int localPointID) const;

	int *GetGlobalCellIDs() const;
	int *GetGlobalPointIDs() const;
	int *GetLocalConnectivities() const;
	int *GetLocalLinks() const;

private:
	int *globalCellIDs, *globalPointIDs;
	int *localConnectivities, *localLinks;
	int localNumOfCells, localNumOfPoints;
};

class BlockTetrahedronPair {
public:
	BlockTetrahedronPair(int blockID, int tetrahedronID);

	int GetBlockID() const;
	int GetTetrahedronID() const;
	bool operator == (const BlockTetrahedronPair &anotherPair) const;

private:
	int blockID, tetrahedronID;
};

class HashForBlockTetrahedronPair {
public:
	int operator () (const lcs::BlockTetrahedronPair &btPair) const;
};

class ExecutionBlock {
public:
	ExecutionBlock();
	~ExecutionBlock();

	void SetNumOfParticles(int numOfParticles);
	void SetParticleID(int index, int particleID);
	void SetBlockRecord(const lcs::BlockRecord *blockRecord);

private:
	int *particleIDs;
	const lcs::BlockRecord *blockRecord;
	int numOfParticles;
};

}

#endif