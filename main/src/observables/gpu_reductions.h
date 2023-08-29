/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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
 * @brief reductions for different observables on the GPU
 *
 * @author Lukas Schmidt
 */

#pragma once

#include <tuple>
#include "cstone/sfc/box.hpp"
#include "cstone/tree/definitions.h"
#include "sph/math.hpp"

namespace sphexa
{
/*!@brief GPU reductions for the Kelvin-Helmholtz Growth rate
 *
 * @param x             x-coordinates
 * @param y             y-coordinates
 * @param vy            y-velocities
 * @param xm            volume element definition
 * @param kx            volume element normalization
 * @param box           global domain
 * @param startIndex    first particle to be included
 * @param endIndex      last particle to be included
 * @return              a tuple containing the local si, di, ci
 */
template<class Tc, class Tv, class Tm>
extern std::tuple<double, double, double> gpuGrowthRate(const Tc* x, const Tc* y, const Tv* vy, const Tm* xm,
                                                        const Tm* kx, const cstone::Box<Tc>& box, size_t startIndex,
                                                        size_t endIndex);

/*! @brief compute square of local Mach number sum
 *
 * @tparam     Tc     float or double
 * @tparam     Tv     float or double
 * @param[in]  vx     velocities x-component
 * @param[in]  vy     velocities y-component
 * @param[in]  vz     velocities z-component
 * @param[in]  c      local speed of sound
 * @param[in]  first  first local particle in vx,vy,vz,c arrays
 * @param[in]  last   last local particle
 * @return
 */
template<class Tc, class Tv>
extern double machSquareSumGpu(const Tv* vx, const Tv* vy, const Tv* vz, const Tc* c, size_t first, size_t last);

/*!@brief sum up the particles that still belong to the cloud
 *
 * @tparam Tc
 * @tparam Tt       float or double
 * @tparam Tm
 * @param temp      particle temperatures
 * @param kx        VE normalization
 * @param xmass     VE definition
 * @param m         particle mass
 * @param rhoBubble initial cloud density
 * @param tempWind  initial cloud temperature
 * @param first     index of first local particle
 * @param last      index of last local particle
 * @return
 */
template<class Tc, class Tt, class Tm>
extern size_t survivorsGpu(const Tt* temp, const Tt* kx, const Tc* xmass, const Tm* m, double rhoBubble,
                           double tempWind, size_t first, size_t last);

/*!
 * @brief struct and sorting functors for the Rayleigh-Taylor Observable
 * @tparam T floating point type of coordinates and velocities
 */
template<class T>
struct AuxT
{
    T pos;
    T vel;
};

struct greaterRT
{
    template<class AuxT>
    HOST_DEVICE_FUN
    bool operator()(AuxT const& a, AuxT const& b) const
    {
        return a.pos > b.pos;
    }
};

struct lowerRT
{
    template<class AuxT>
    HOST_DEVICE_FUN
    bool operator()(AuxT const& a, AuxT const& b) const
    {
        return a.pos < b.pos;
    }
};
/*!
 * @brief collect velocity data of the bubble and peaks in the RT
 *
 * @tparam T
 * @tparam Tc
 * @tparam Tm
 * @param startIndex
 * @param endIndex
 * @param ymin
 * @param ymax
 * @param h
 * @param y
 * @param vy
 * @param markRamp
 * @return
 */
template<class T, class Tc, class Tm>
extern std::tuple<std::vector<AuxT<T>>, std::vector<AuxT<T>>> localGrowthRateRTGpu(size_t first, size_t last, Tc ymin,
                                                                                    Tc ymax, const T* h, const T* y,
                                                                                    const T* vy, const Tm* markRamp);

} // namespace sphexa