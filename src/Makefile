include Makefile.inc

DIRS=global testing

all : 
	for d in $(DIRS); do (cd $$d; make all); done
	
clean :
	-for d in $(DIRS); do (cd $$d; make clean); done
