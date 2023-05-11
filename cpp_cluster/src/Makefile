all: build

CXX := g++
AR := ar
# CCFLAGS := -DNDEBUG -std=c++17 -Wall -Wextra -march=native -Ofast -flto
CCFLAGS := -DNDEBUG -std=c++17 -Wall -Wextra -march=native -Ofast -flto -g
CCFLAGS += -DLATT_SIZE=$(LATT_SIZE)
ifdef CPER
	CCFLAGS += -DCPER
endif
ifdef STAG
	CCFLAGS += -DSTAG
endif
ARFLAGS := -rcs

build: u1_3d_cluster u1_3d_wloop_L$(LATT_SIZE)

MY_HEADERS = \
	cluster.h lattice.h latt_ops.h measurements.h \
	metropolis.h my_rand.h util.h
main.o: main.cpp $(MY_HEADERS) args.hxx
	$(CXX) $(CCFLAGS) -o $@ -c $<
main_wloop_L$(LATT_SIZE).o: main_wloop.cpp $(MY_HEADERS) args.hxx
	$(CXX) $(CCFLAGS) -o $@ -c $<
main_prof.o: main_prof.cpp $(MY_HEADERS) args.hxx
	$(CXX) $(CCFLAGS) -o $@ -c $<
cluster.o: cluster.cpp $(MY_HEADERS)
	$(CXX) $(CCFLAGS) -o $@ -c $<
metropolis.o: metropolis.cpp $(MY_HEADERS)
	$(CXX) $(CCFLAGS) -o $@ -c $<
u1_3d_cluster: main.o cluster.o metropolis.o
	$(CXX) $(CCFLAGS) -o $@ $^
u1_3d_wloop_L$(LATT_SIZE): main_wloop_L$(LATT_SIZE).o
	$(CXX) $(CCFLAGS) -o $@ $^
u1_3d_prof: main_prof.o
	$(CXX) $(CCFLAGS) -o $@ $^
# u1_3d_prof: main_prof.cpp $(MY_HEADERS) args.hxx
# 	$(CXX) $(CCFLAGS) -o $@ $^

GTEST_DIR := gtest/googletest
GTEST_BUILD_DIR := gtest_build
GTEST_CCFLAGS := -std=c++17 -g -Wall -Wextra -pthread -isystem $(GTEST_DIR)/include
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
		$(GTEST_DIR)/include/gtest/internal/*.h
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)
$(GTEST_BUILD_DIR)/.dirstamp :
	mkdir -p $(GTEST_BUILD_DIR) && touch $@
$(GTEST_BUILD_DIR)/gtest-all.o : $(GTEST_SRCS_) $(GTEST_BUILD_DIR)/.dirstamp
	$(CXX) $(GTEST_CCFLAGS) -I$(GTEST_DIR) -c \
            $(GTEST_DIR)/src/gtest-all.cc -o $@

$(GTEST_BUILD_DIR)/gtest_main.o : $(GTEST_SRCS_) $(GTEST_BUILD_DIR)/.dirstamp
	$(CXX) $(GTEST_CCFLAGS) -I$(GTEST_DIR) -c \
            $(GTEST_DIR)/src/gtest_main.cc -o $@

$(GTEST_BUILD_DIR)/gtest.a : $(GTEST_BUILD_DIR)/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

$(GTEST_BUILD_DIR)/gtest_main.a : $(GTEST_BUILD_DIR)/gtest-all.o $(GTEST_BUILD_DIR)/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

tests.o: tests.cpp $(MY_HEADERS) $(GTEST_HEADERS)
	$(CXX) $(GTEST_CCFLAGS) -o $@ -c $<

tests: tests.o cluster.o $(GTEST_BUILD_DIR)/gtest_main.a
	$(CXX) $(GTEST_CCFLAGS) -lpthread $^ -o $@

clean:
	rm -f u1_3d_cluster u1_3d_wloop_* u1_3d_prof tests \
	main.o main_wloop_*.o main_prof.o metropolis.o cluster.o tests.o
	rm -rf $(GTEST_BUILD_DIR)
