#ifndef CLUSTER_H
#define CLUSTER_H

#include <cassert>
#include "lattice.h"
#include "my_rand.h"
#include "util.h"


std::vector<int> flip_clusters(
    cfg_t* cfg, cfg_t cfg_star, float_t e2,
    my_rand &rng, const latt_shape* shape);

// void sample_bonds(
//     const cfg_t* cfg, int* bonds, float_t e2, double h_star,
//     my_rand& rng, const latt_shape* shape);

#endif
