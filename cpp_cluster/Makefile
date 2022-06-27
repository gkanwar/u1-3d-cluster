all: build

CCFLAGS := -O3 -DNDEBUG

build: u1_3d_cluster

main.o: main.cpp cluster.h
	g++ $(CCFLAGS) -o $@ -c $<
cluster.o: cluster.cpp cluster.h
	g++ $(CCFLAGS) -o $@ -c $<
u1_3d_cluster: main.o cluster.o
	g++ $(CCFLAGS) -o $@ $+

clean:
	rm -f u1_3d_cluster main.o cluster.o
