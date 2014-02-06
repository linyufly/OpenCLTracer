/**********************************************
File			:		lcsGeometry.h
Author			:		Mingcheng Chen
Last Update		:		June 13th, 2012
***********************************************/

#ifndef __LCS_Geometry_H
#define __LCS_Geometry_H

#include <vtkUnstructuredGrid.h>

namespace lcs {

////////////////////////////////////////////////
int Sign(double a, double epsilon);

double Sqr(double a);

class Vector;

lcs::Vector operator + (const lcs::Vector &, const lcs::Vector &);
lcs::Vector operator - (const lcs::Vector &, const lcs::Vector &);
lcs::Vector operator * (const lcs::Vector &, const double &);
lcs::Vector operator / (const lcs::Vector &, const double &);
lcs::Vector Cross(const lcs::Vector &, const lcs::Vector &);
double Dot(const lcs::Vector &, const lcs::Vector &);
double Mixed(const lcs::Vector &, const lcs::Vector &, const lcs::Vector &);

////////////////////////////////////////////////
class Vector {
public:
	Vector() {
		this->x = this->y = this->z = 0;
	}

	Vector(double x, double y, double z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}

	Vector(double *arr) {
		this->x = arr[0];
		this->y = arr[1];
		this->z = arr[2];
	}

	Vector(const Vector &vec) {
		this->x = vec.x;
		this->y = vec.y;
		this->z = vec.z;
	}

	void SetX(double x) {
		this->x = x;
	}

	void SetY(double y) {
		this->y = y;
	}

	void SetZ(double z) {
		this->z = z;
	}

	double GetX() const {
		return this->x;
	}

	double GetY() const {
		return this->y;
	}

	double GetZ() const {
		return this->z;
	}

	double Length() const;

	friend lcs::Vector operator + (const lcs::Vector &, const lcs::Vector &);
	friend lcs::Vector operator - (const lcs::Vector &, const lcs::Vector &);
	friend lcs::Vector operator * (const lcs::Vector &, const double &);
	friend lcs::Vector operator / (const lcs::Vector &, const double &);
	friend lcs::Vector lcs::Cross(const lcs::Vector &, const lcs::Vector &);
	friend double lcs::Dot(const lcs::Vector &, const lcs::Vector &);
	friend double lcs::Mixed(const lcs::Vector &, const lcs::Vector &, const lcs::Vector &);

private:
	double x, y, z;
};

////////////////////////////////////////////////
class Tetrahedron {
public:
	Tetrahedron() {
	}

	Tetrahedron(const Vector &a, const Vector &b, const Vector &c, const Vector &d) {
		this->vertices[0] = a;
		this->vertices[1] = b;
		this->vertices[2] = c;
		this->vertices[3] = d;
	}

	Tetrahedron(const Vector *);

	Vector GetVertex(int index) const {
		return this->vertices[index];
	}

	void SetVertex(int index, const Vector &a) {
		this->vertices[index] = a;
	}

	void CalculateNaturalCoordinates(const Vector &, double *) const;

private:
	Vector vertices[4];
};

////////////////////////////////////////////////
class TetrahedralGrid {
public:
	TetrahedralGrid() {
		numOfVertices = 0;
		numOfCells = 0;
		vertices = NULL;
		tetrahedralConnectivities = NULL;
		tetrahedralLinks = NULL;
	}

	TetrahedralGrid(vtkUnstructuredGrid *);

	~TetrahedralGrid() {
		if (vertices) delete [] vertices;
		if (tetrahedralConnectivities) delete [] tetrahedralConnectivities;
		if (tetrahedralLinks) delete [] tetrahedralLinks;
	}

	Tetrahedron GetTetrahedron(int) const;

	Vector GetVertex(int index) const {
		return vertices[index];
	}

	Vector GetVelocity(int index) const {
		return velocities[index];
	}

	int FindCell(const Vector &, const double &) const;

	int FindCell(const Vector &, const double &, int) const;

	int ProfiledFindCell(const Vector &, const double &, int);

	void GetInterpolatedVelocity(const Vector &, int, double *) const;

	int GetNumOfVertices() const {
		return this->numOfVertices;
	}

	int GetNumOfCells() const {
		return this->numOfCells;
	}

	void GetCellLink(int index, int *arr) const {
		memcpy(arr, this->tetrahedralLinks + (index << 2), sizeof(int) * 4);
	}

	void GetCellConnectivity(int index, int *arr) const {
		memcpy(arr, this->tetrahedralConnectivities + (index << 2), sizeof(int) * 4);
	}

	int GetLastFindCellCost() const {
		return this->lastFindCellCost;
	}

	void ReadConnectivities(int *destination) const {
		memcpy(destination, this->tetrahedralConnectivities, sizeof(int) * 4 * this->numOfCells);
	}

	void ReadLinks(int *destination) const {
		memcpy(destination, this->tetrahedralLinks, sizeof(int) * 4 * this->numOfCells);
	}

	void ReadPositions(double *destination) const {
		for (int i = 0; i < this->numOfVertices; i++) {
			destination[i * 3] = this->vertices[i].GetX();
			destination[i * 3 + 1] = this->vertices[i].GetY();
			destination[i * 3 + 2] = this->vertices[i].GetZ();
		}
	}

	void ReadPositions(float *destination) const {
		for (int i = 0; i < this->numOfVertices; i++) {
			destination[i * 3] = this->vertices[i].GetX();
			destination[i * 3 + 1] = this->vertices[i].GetY();
			destination[i * 3 + 2] = this->vertices[i].GetZ();
		}
	}

	void ReadVelocities(double *destination) const {
		for (int i = 0; i < this->numOfVertices; i++) {
			destination[i * 3] = this->velocities[i].GetX();
			destination[i * 3 + 1] = this->velocities[i].GetY();
			destination[i * 3 + 2] = this->velocities[i].GetZ();
		}
	}

	void ReadVelocities(float *destination) const {
		for (int i = 0; i < this->numOfVertices; i++) {
			destination[i * 3] = this->velocities[i].GetX();
			destination[i * 3 + 1] = this->velocities[i].GetY();
			destination[i * 3 + 2] = this->velocities[i].GetZ();
		}
	}

private:
	int numOfVertices;
	int numOfCells;
	Vector *vertices, *velocities;
	int *tetrahedralConnectivities;
	int *tetrahedralLinks;

// Profiling Result
	int lastFindCellCost;
};

}

#endif