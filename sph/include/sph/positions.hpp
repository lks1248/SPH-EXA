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
 * @brief 2nd order time-step integrator
 *
 * @author Aurelien Cavelan
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/util/array.hpp"
#include "cstone/util/tuple.hpp"
#include "cstone/tree/accel_switch.hpp"

#include "sph/sph_gpu.hpp"
#include "sph/eos.hpp"
#include "sph/table_lookup.hpp"

namespace sph
{

//! @brief checks whether a particle is close to a fixed boundary and reflects the velocity if so
template<class Tc, class Th>
HOST_DEVICE_FUN void fbcAdjust(const cstone::Vec3<Tc> X, cstone::Vec3<Tc>& V, const cstone::Vec3<Tc>& A,
                               const cstone::Box<Tc>& box, const Th& hi, const double dt, const Th* wh)
{
    constexpr float    threshold       = 1.;
    constexpr float    invTHold        = 1 / threshold;
    cstone::Vec3<bool> isBoundaryFixed = {
        box.boundaryX() == cstone::BoundaryType::fixed,
        box.boundaryY() == cstone::BoundaryType::fixed,
        box.boundaryZ() == cstone::BoundaryType::fixed,
    };
    cstone::Vec3<Tc> boxMax = {box.xmax(), box.ymax(), box.zmax()};
    cstone::Vec3<Tc> boxMin = {box.xmin(), box.ymin(), box.zmin()};

    for (int j = 0; j < 3; ++j)
    {
        if (isBoundaryFixed[j])
        {
            // Flip the velocity if integration would put the particle in the "critical" zone
            Tc dXj            = X[j] + V[j] * dt + 0.5 * A[j] * dt * dt;
            Th relDistanceMax = std::abs(boxMax[j] - dXj) / hi;
            Th relDistanceMin = std::abs(boxMin[j] - dXj) / hi;
            Th minDistance    = relDistanceMin < relDistanceMax ? relDistanceMin : relDistanceMax;

            // if (minDistance < 2 * threshold) { V[j] *= -1 + invTHold * minDistance; }
            if (minDistance < 2 * threshold) { V[j] *= 1 - lt::lookup(wh, minDistance * invTHold * Th(0.5)); }
        }
    }
}

//! @brief update the energy according to Adams-Bashforth (2nd order)
template<class T1, class T2>
HOST_DEVICE_FUN double energyUpdate(double u_old, double dt, double dt_m1, T1 du, T2 du_m1)
{
    double deltaA = 0.5 * dt * dt / dt_m1;
    double deltaB = dt + deltaA;
    double u_new  = u_old + du * deltaB - du_m1 * deltaA;
    // To prevent u < 0 (when cooling with GRACKLE is active)
    if (u_new < 0.) { u_new = u_old * std::exp(u_new * dt / u_old); }
    return u_new;
}

//! @brief Update positions according to Press (2nd order)
template<class T, class Th>
HOST_DEVICE_FUN auto positionUpdate(double dt, double dt_m1, cstone::Vec3<T> X, cstone::Vec3<T> A, cstone::Vec3<T> X_m1,
                                    const cstone::Box<T>& box, bool anyFbc, const Th& hi, const Th* wh)
{
    double deltaA = dt + T(0.5) * dt_m1;
    double deltaB = T(0.5) * (dt + dt_m1);

    auto Val = X_m1 * (T(1) / dt_m1);
    auto V   = Val + A * deltaA;
    if (anyFbc) { fbcAdjust(X, V, A, box, hi, dt, wh); }
    double deltaC = deltaB / deltaA;
    auto   dX     = dt * Val + (V - Val) * dt * deltaC;
    X             = cstone::putInBox(X + dX, box);

    return util::tuple<cstone::Vec3<T>, cstone::Vec3<T>, cstone::Vec3<T>>{X, V, dX};
}

template<class T, class Dataset>
void updatePositionsHost(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    bool fbcX = (box.boundaryX() == cstone::BoundaryType::fixed);
    bool fbcY = (box.boundaryY() == cstone::BoundaryType::fixed);
    bool fbcZ = (box.boundaryZ() == cstone::BoundaryType::fixed);

    bool anyFBC = fbcX || fbcY || fbcZ;

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {

        cstone::Vec3<T> A{d.ax[i], d.ay[i], d.az[i]};
        cstone::Vec3<T> X{d.x[i], d.y[i], d.z[i]};
        cstone::Vec3<T> X_m1{d.x_m1[i], d.y_m1[i], d.z_m1[i]};
        cstone::Vec3<T> V;
        util::tie(X, V, X_m1) = positionUpdate(d.minDt, d.minDt_m1, X, A, X_m1, box, anyFBC, d.h[i], d.wh.data());

        util::tie(d.x[i], d.y[i], d.z[i])          = util::tie(X[0], X[1], X[2]);
        util::tie(d.x_m1[i], d.y_m1[i], d.z_m1[i]) = util::tie(X_m1[0], X_m1[1], X_m1[2]);
        util::tie(d.vx[i], d.vy[i], d.vz[i])       = util::tie(V[0], V[1], V[2]);
    }
}

template<class Dataset, class T>
void updateTempHost(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    bool haveMui = !d.mui.empty();
    auto constCv = idealGasCv(d.muiConst, d.gamma);

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        auto cv    = haveMui ? idealGasCv(d.mui[i], d.gamma) : constCv;
        auto u_old = cv * d.temp[i];
        d.temp[i]  = energyUpdate(u_old, d.minDt, d.minDt_m1, d.du[i], d.du_m1[i]) / cv;
        d.du_m1[i] = d.du[i];
    }
}

template<class Dataset, class T>
void updateIntEnergyHost(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        d.u[i]     = energyUpdate(d.u[i], d.minDt, d.minDt_m1, d.du[i], d.du_m1[i]);
        d.du_m1[i] = d.du[i];
    }
}

template<class T, class Dataset>
void computePositions(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        T     constCv = d.mui.empty() ? idealGasCv(d.muiConst, d.gamma) : -1.0;
        auto* d_mui   = d.mui.empty() ? nullptr : rawPtr(d.devData.mui);

        computePositionsGpu(startIndex, endIndex, d.minDt, d.minDt_m1, rawPtr(d.devData.x), rawPtr(d.devData.y),
                            rawPtr(d.devData.z), rawPtr(d.devData.vx), rawPtr(d.devData.vy), rawPtr(d.devData.vz),
                            rawPtr(d.devData.x_m1), rawPtr(d.devData.y_m1), rawPtr(d.devData.z_m1),
                            rawPtr(d.devData.ax), rawPtr(d.devData.ay), rawPtr(d.devData.az), rawPtr(d.devData.temp),
                            rawPtr(d.devData.u), rawPtr(d.devData.du), rawPtr(d.devData.du_m1), rawPtr(d.devData.h),
                            rawPtr(d.devData.wh), d_mui, d.gamma, constCv, box);
    }
    else
    {
        updatePositionsHost(startIndex, endIndex, d, box);

        if (!d.temp.empty()) { updateTempHost(startIndex, endIndex, d, box); }
        else if (!d.u.empty()) { updateIntEnergyHost(startIndex, endIndex, d, box); }
    }
}

} // namespace sph
