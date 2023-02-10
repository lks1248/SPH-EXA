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

/*!@file
 * @brief utilities for initial condition generation
 *
 * @author Lukas Schmidt
 */
#pragma once

#include <vector>

namespace sphexa
{

template<class T, class KeyType>
void sortBySFCKey(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, size_t blockSize)
{
    cstone::Box<T> templateBox(0, 1, 0, 1, 0, 1);

    std::vector<KeyType> codes(blockSize);
    computeSfcKeys(x.data(), y.data(), z.data(), cstone::sfcKindPointer(codes.data()), blockSize, templateBox);

    std::vector<cstone::LocalIndex> sfcOrder(blockSize);
    std::iota(begin(sfcOrder), end(sfcOrder), cstone::LocalIndex(0));
    cstone::sort_by_key(begin(codes), end(codes), begin(sfcOrder));

    std::vector<T> buffer(blockSize);
    cstone::gather<cstone::LocalIndex>(sfcOrder, x.data(), buffer.data());
    std::swap(x, buffer);
    cstone::gather<cstone::LocalIndex>(sfcOrder, y.data(), buffer.data());
    std::swap(y, buffer);
    cstone::gather<cstone::LocalIndex>(sfcOrder, z.data(), buffer.data());
    std::swap(z, buffer);
}

/*!@brief remove every n-th particle of a template block
 * @param   n           factor to reduce density
 * @params  x,y,z       template blocks
 * @param   blockSize   original template size
 * @return  template block with density 1/n of the original
 */
template<class T, class Dataset>
auto makeLessDenseTemplate(size_t n, std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, size_t blockSize)
{
    using KeyType = typename Dataset::KeyType;

    std::vector<T> xSmall, ySmall, zSmall;
    xSmall.reserve(blockSize);
    ySmall.reserve(blockSize);
    zSmall.reserve(blockSize);

    sortBySFCKey<T, KeyType>(x, y, z, blockSize);

    for (size_t i = 0; i < blockSize; i += n)
    {
        xSmall.push_back(x[i]);
        ySmall.push_back(y[i]);
        zSmall.push_back(z[i]);
    }

    xSmall.shrink_to_fit();
    ySmall.shrink_to_fit();
    zSmall.shrink_to_fit();

    return std::make_tuple(xSmall, ySmall, zSmall);
}

/*!@brief apply fixed boundary conditions to one axis
 *
 * @tparam T    field type
 * @param pos   position data of axis to be applied
 * @param axisMax max coordinate of the box in the axis of pos
 * @param axisMin min coordinate of the box in the axis of pos
 * @param size  number of particles
 */
template<class T>
void applyFixedBoundaries(T* pos, T* vx, T* vy, T* vz, T* h, T axisMax, T axisMin, size_t size)
{

#pragma omp parallel for
    for (size_t i = 0; i < size; i++)
    {
        T distMax = std::abs(axisMax - pos[i]);
        T distMin = std::abs(axisMin - pos[i]);

        if (distMax < 2.0 * h[i] || distMin < 2.0 * h[i])
        {
            vx[i] = 0.0;
            vy[i] = 0.0;
            vz[i] = 0.0;
        }
    }
}
}; // namespace sphexa