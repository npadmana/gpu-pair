include ../Makefile.inc

TESTS = test_random test_particles1
DEPENDS = ../global/libglobal_pairs.a ../global/Particles.h ../global/cpu_utils.h 

LIBS = -lgsl -L/opt/local/lib -L../global -lpairs_global

all : $(DEPENDS)
	for t in $(TESTS); do ( $(CXX) $(CXX_FLAGS) -I../global $$t.cpp -o $$t.x $(CXX_INCLUDE_PATHS) $(LIBS) ); done

clean :
	-rm *.o *.x
