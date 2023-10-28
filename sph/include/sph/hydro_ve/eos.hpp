/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Density i-loop OpenMP driver
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "sph/sph_gpu.hpp"
#include "sph/eos.hpp"

namespace sph
{

/*! @brief ideal gas EOS interface w/o temperature for SPH where rho is computed on-the-fly
 *
 * @tparam Dataset
 * @param startIndex  index of first locally owned particle
 * @param endIndex    index of last locally owned particle
 * @param d           the dataset with the particle buffers
 *
 * In this simple version of equation of state, we calculate all dependent quantities
 * also for halos, not just assigned particles in [startIndex:endIndex], so that
 * we could potentially avoid halo exchange of p and c in return for exchanging halos of u.
 */
template<typename Dataset, class T>
void computeEOS_Impl(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    const auto* temp  = d.temp.data();
    const auto* m     = d.m.data();
    const auto* kx    = d.kx.data();
    const auto* xm    = d.xm.data();
    const auto* gradh = d.gradh.data();

    auto* prho = d.prho.data();
    auto* c    = d.c.data();

    bool storeRho = (d.rho.size() == d.m.size());
    bool storeP   = (d.p.size() == d.m.size());

    bool fbcX = (box.boundaryX() == cstone::BoundaryType::fixed);
    bool fbcY = (box.boundaryY() == cstone::BoundaryType::fixed);
    bool fbcZ = (box.boundaryZ() == cstone::BoundaryType::fixed);

    bool anyFBC = fbcX || fbcY || fbcZ;

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        if (anyFBC && d.vx[i] == T(0) && d.vy[i] == T(0) && d.vz[i] == T(0))
        {
            if (fbcCheck(d.x[i], d.h[i], box.xmax(), box.xmin(), fbcX) ||
                fbcCheck(d.y[i], d.h[i], box.ymax(), box.ymin(), fbcY) ||
                fbcCheck(d.z[i], d.h[i], box.zmax(), box.zmin(), fbcZ))
            {
                continue;
            }
        }

        auto rho      = kx[i] * m[i] / xm[i];
        auto [pi, ci] = idealGasEOS(temp[i], rho, d.muiConst, d.gamma);
        prho[i]       = pi / (kx[i] * m[i] * m[i] * gradh[i]);
        c[i]          = ci;
        if (storeRho) { d.rho[i] = rho; }
        if (storeP) { d.p[i] = pi; }
    }
}

template<class Dataset, class T>
void computeEOS(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeEOS(startIndex, endIndex, d.muiConst, d.gamma, rawPtr(d.devData.h), rawPtr(d.devData.x),
                         rawPtr(d.devData.y), rawPtr(d.devData.z), rawPtr(d.devData.vx), rawPtr(d.devData.vy),
                         rawPtr(d.devData.vz), rawPtr(d.devData.temp), rawPtr(d.devData.m), rawPtr(d.devData.kx),
                         rawPtr(d.devData.xm), rawPtr(d.devData.gradh), rawPtr(d.devData.prho), rawPtr(d.devData.c),
                         rawPtr(d.devData.rho), rawPtr(d.devData.p), box);
    }
    else { computeEOS_Impl(startIndex, endIndex, d, box); }
}

} // namespace sph
