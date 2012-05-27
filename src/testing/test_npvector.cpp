#include "NPVector.h"

using namespace std;

int main() {
	NPVector3f x,y,z,w;
	x[0] = 1.0; x[1] = 2.0; x[2] = 3.0;
	y[0] = 0.0;; y[1] = 1.0; y[2] = 2.0;

	z = x+y;
	cout << endl; z.print(); cout << endl;

	z = z * 2.0f;
	cout << endl; z.print(); cout << endl;
	z = z/3.0f;
	cout << endl; z.print(); cout << endl;
 }
