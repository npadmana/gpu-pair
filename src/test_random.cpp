#include <iostream>
#include "utils.h"
#include <algorithm>

using namespace std;

int main() {
  vector<float> tmp(10);
  GSLRandom rnd(101);

  rnd(2.0, 7.0, tmp);

  for_each(tmp.begin(), tmp.end(), [](float x) {
      cout << x << endl;
      });
}
