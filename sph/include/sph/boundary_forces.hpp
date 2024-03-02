/*
 * MIT License
 *
 * Copyright (c) 2023 CSCS, ETH Zurich
 *               2023 University of Basel
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

#include "sph/table_lookup.hpp"

/*! @file
 * @brief Force correction approach for fixed boundaries
 *
 * @author Lukas Schmidt
 */
namespace sph
{

template<class Dataset>
void fixedBoundaryForceCorrection(size_t startIndex, size_t endIndex, Dataset& d,
                                  const cstone::Box<typename Dataset::RealType>& box)
{
    using T  = typename Dataset::RealType;
    using Th = typename Dataset::HydroType;

    cstone::Vec3<bool> isBoundaryFixed = {
        box.boundaryX() == cstone::BoundaryType::fixed,
        box.boundaryY() == cstone::BoundaryType::fixed,
        box.boundaryZ() == cstone::BoundaryType::fixed,
    };
    cstone::Vec3<T>   boxMax = {box.xmax(), box.ymax(), box.zmax()};
    cstone::Vec3<T>   boxMin = {box.xmin(), box.ymin(), box.zmin()};
    cstone::Vec3<Th* const> A      = {d.ax.data(), d.ay.data(), d.az.data()};

    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        throw std::runtime_error("no cuda force correction!\n");
    }
    else
    {
#pragma omp parallel for schedule(static)
        for (int j = 0; j < 3; j++)
        {
            if (isBoundaryFixed[j])
            {
                for (size_t i = startIndex; i < endIndex; i++)
                {
                    cstone::Vec3<T> X              = {d.x[i], d.y[i], d.z[i]};
                    Th              hi             = d.h[i];
                    Th              relDistanceMax = std::abs(boxMax[j] - X[j]) / hi;
                    Th              relDistanceMin = std::abs(boxMin[j] - X[j]) / hi;
                    if (relDistanceMax < 2)
                    {
                        T whi = lt::lookup(d.wh.data(), relDistanceMax); // need to normalize?
                        A[j][i] -= 2 * A[j][i] * whi;
                    }
                    if (relDistanceMin < 2)
                    {
                        T whi = lt::lookup(d.wh.data(), relDistanceMin);
                        A[j][i] -= 2 * A[j][i] * whi;
                    }
                }
            }
        }
    }
}

}; // namespace sph