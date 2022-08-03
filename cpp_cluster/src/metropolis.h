#ifndef METROPOLIS_H
#define METROPOLIS_H

#include "lattice.h"
#include "my_rand.h"

double metropolis_update(
    int* cfg, double e2, my_rand &rng, const latt_shape* shape);

#endif
