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
 * @brief  Implementation of halo discovery and halo exchange
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>

#include "cstone/domain/layout.hpp"
#include "cstone/halos/exchange_halos_gpu.cuh"
#include "cstone/halos/halos.hpp"
#include "cstone/halos/radii.hpp"
#include "cstone/traversal/collisions.hpp"
#include "cstone/util/gsl-lite.hpp"
#include "cstone/util/index_ranges.hpp"

namespace cstone
{

template<class KeyType>
template<class... DeviceVectors, class DeviceVector>
void Halos<KeyType>::exchangeHalosGpu(std::tuple<DeviceVectors&...> arrays,
                                      DeviceVector& sendBuffer,
                                      DeviceVector& receiveBuffer) const
{
    std::apply(
        [this, &sendBuffer, &receiveBuffer](auto&... arrays)
        {
            haloExchangeGpu(haloEpoch_++, incomingHaloIndices_, outgoingHaloIndices_, sendBuffer, receiveBuffer,
                            thrust::raw_pointer_cast(arrays.data())...);
        },
        arrays);
}

} // namespace cstone
