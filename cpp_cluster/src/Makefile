all: build

CXX := g++
AR := ar
CCFLAGS := -Ofast -flto -DNDEBUG -std=c++17 -Wall -Wextra
ARFLAGS := -rcs

build: u1_3d_cluster

MY_HEADERS = cluster.h lattice.h measurements.h
main.o: main.cpp $(MY_HEADERS) args.hxx
	$(CXX) $(CCFLAGS) -o $@ -c $<
cluster.o: cluster.cpp $(MY_HEADERS)
	$(CXX) $(CCFLAGS) -o $@ -c $<
u1_3d_cluster: main.o cluster.o
	$(CXX) $(CCFLAGS) -o $@ $^

GTEST_DIR := gtest/googletest
GTEST_CCFLAGS := -std=c++17 -g -Wall -Wextra -pthread -isystem $(GTEST_DIR)/include
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
		$(GTEST_DIR)/include/gtest/internal/*.h
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)
$(GTEST_DIR)/gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_CCFLAGS) -I$(GTEST_DIR) -c \
            $(GTEST_DIR)/src/gtest-all.cc -o $(GTEST_DIR)/gtest-all.o

$(GTEST_DIR)/gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_CCFLAGS) -I$(GTEST_DIR) -c \
            $(GTEST_DIR)/src/gtest_main.cc -o $(GTEST_DIR)/gtest_main.o

$(GTEST_DIR)/gtest.a : $(GTEST_DIR)/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

$(GTEST_DIR)/gtest_main.a : $(GTEST_DIR)/gtest-all.o $(GTEST_DIR)/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

tests.o: tests.cpp $(MY_HEADERS) $(GTEST_HEADERS)
	$(CXX) $(GTEST_CCFLAGS) -o $@ -c $<

tests: tests.o cluster.o $(GTEST_DIR)/gtest_main.a
	$(CXX) $(GTEST_CCFLAGS) -lpthread $^ -o $@

clean:
	rm -f u1_3d_cluster tests main.o cluster.o tests.o
