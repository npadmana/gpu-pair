include ../Makefile.inc

OBJS = cpu_utils.o 
HEADERS = cpu_utils.h Particles.h Hist.h CPU_PairCounts.h

all : libpairs_global.a

libpairs_global.a : $(OBJS)
	ar -rs libpairs_global.a $(OBJS) 

Hist.o :
	true

CPU_PairCounts.o :
	true

.cpp.o : $(HEADERS)
	$(CXX) $(CXX_FLAGS) -c $< -o $@ $(CXX_INCLUDE_PATHS)


clean :
	-rm *.o *.a

