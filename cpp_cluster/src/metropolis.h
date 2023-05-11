#ifndef METROPOLIS_H
#define METROPOLIS_H

#include "lattice.h"
#include "my_rand.h"
#include "util.h"

float_t metropolis_update(
    cfg_t* cfg, float_t e2, my_rand &rng, const latt_shape* shape);
float_t metropolis_update_with_wloop(
    cfg_t* cfg, float_t e2, my_rand &rng, const latt_shape* shape, const WloopShape& wloop);

#endif
