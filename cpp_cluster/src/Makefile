all: build

CCFLAGS := -Ofast -DNDEBUG -flto

build: u1_3d_cluster

main.o: main.cpp cluster.h lattice.h measurements.h args.hxx
	g++ $(CCFLAGS) -o $@ -c $<
cluster.o: cluster.cpp cluster.h lattice.h measurements.h
	g++ $(CCFLAGS) -o $@ -c $<
u1_3d_cluster: main.o cluster.o
	g++ $(CCFLAGS) -o $@ $+

clean:
	rm -f u1_3d_cluster main.o cluster.o
