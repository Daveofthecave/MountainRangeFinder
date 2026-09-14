// viz/vizshim.c -- tiny C ABI so the GIF scripts sample the real climate fields.
// Build from the project root:
//   gcc -O2 -fPIC -shared -I cubiomes viz/vizshim.c \
//       $(ls cubiomes/*.c | grep -v tests.c) -o viz/libmrfshim.so -lm
#include <stdint.h>
#include "biomenoise.h"
#include "generator.h"

static BiomeNoise g_bn;
static Generator  g_gen;

// Fill `out` (row-major, nz x nx) with one climate parameter over a quart grid.
// nptype: 0 temp, 1 humidity, 2 cont, 3 erosion, 5 weirdness.
// nmax: 2 = "0B", 4 = "1B", -1 = full octaves -- same convention as CV.
void mrf_climate_grid(uint64_t seed, int nptype, int nmax,
                      int32_t x0q, int32_t z0q, int32_t nx, int32_t nz,
                      int32_t stepq, double *out) {
    setClimateParaSeed(&g_bn, seed, 0, nptype, nmax);
    for (int32_t j = 0; j < nz; j++)
        for (int32_t i = 0; i < nx; i++)
            out[(size_t)j * nx + i] =
                sampleClimatePara(&g_bn, NULL, x0q + i * stepq, z0q + j * stepq);
}

void mrf_gen_seed(int mc, uint64_t seed) {
    setupGenerator(&g_gen, mc, 0);
    applySeed(&g_gen, DIM_OVERWORLD, seed);
}

void mrf_height_grid(int32_t x0, int32_t z0, int32_t nx, int32_t nz,
                     int32_t step, float *out) {
    for (int32_t j = 0; j < nz; j++)
        for (int32_t i = 0; i < nx; i++) {
            float y = -1.0f;
            mapApproxHeight(&y, NULL, &g_gen, NULL,
                            (x0 + i * step) >> 2, (z0 + j * step) >> 2, 1, 1);
            out[(size_t)j * nx + i] = y;
        }
}

int mrf_biome(int32_t x, int32_t z) { return getBiomeAt(&g_gen, 1, x, 256, z); }
