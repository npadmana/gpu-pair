#ifndef HIST_H_
#define HIST_H_ 

#include <vector>

class RHist {
  public :
    float rmin, dr;
    int Nbins;
    std::vector<unsigned long long> hist;

    RHist(int n, float r0, float _dr);
    void print();

};


#endif /* HIST_H_ */
