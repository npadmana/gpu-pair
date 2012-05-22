include ../Makefile.inc

TESTS = test_random.x test_particles1.x

LIBS = -lgsl -L/opt/local/lib -L../global -lpairs_global

all : $(TESTS)

test_%.x : test_%.cpp
	$(CXX) $(CXX_FLAGS) -I../global $< -o $@ $(CXX_INCLUDE_PATHS) $(OBJS) $(LIBS)

clean :
	-rm *.o 
