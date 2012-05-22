include ../Makefile.inc

HEADERS = utils.h Particles.h Hist.h CPU_PairCounts.h
OBJS = utils.o Particles.o Hist.o CPU_PairCounts.o

all : libpairs_global.a

libpairs_global.a : $(OBJS)
	ar -rs libpairs_global.a $(OBJS) 

.cpp.o : $(HEADERS)
	$(CXX) $(CXX_FLAGS) -c $< -o $@ $(CXX_INCLUDE_PATHS)


clean :
	-rm *.o

