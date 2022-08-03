#ifndef CLUSTER_H
#define CLUSTER_H

#include <cassert>
#include "lattice.h"
#include "my_rand.h"


std::vector<int> flip_clusters(
    int* cfg, int cfg_star, double e2,
    my_rand &rng, const latt_shape* shape);

void sample_bonds(
    const int* cfg, int* bonds, double e2, double h_star,
    my_rand& rng, const latt_shape* shape);

#endif
