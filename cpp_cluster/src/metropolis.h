#ifndef METROPOLIS_H
#define METROPOLIS_H

#include "lattice.h"
#include "my_rand.h"
#include "util.h"

double metropolis_update(
    int* cfg, double e2, my_rand &rng, const latt_shape* shape);
double metropolis_update_with_wloop(
    int* cfg, double e2, my_rand &rng, const latt_shape* shape, const WloopShape& wloop);

#endif
