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
 * @brief output and calculate energies and growth rate for Rayleigh-Taylor tests
 *
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 *
 */

#include <array>
#include <mpi.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstddef>

#include "conserved_quantities.hpp"
#include "iobservables.hpp"
#include "io/file_utils.hpp"
#include "cstone/tree/definitions.h"
#include "sph/particles_data.hpp"
#include "sph/positions.hpp"
#include "gpu_reductions.h"

namespace sphexa
{

/*! @brief local calculation of the maximum density (usually central) and radius
 *
 * @tparam        T            double or float
 * @param[in]     startIndex   first locally assigned particle index of buffers in @p d
 * @param[in]     endIndex     last locally assigned particle index of buffers in @p d
 * @param[in]     y            Y coordinate array
 *
 * Returns the 50 local particles with higher density and the 50 local particles
 * with higher radius. Sort function uses greater to sort in reverse order so
 * that we can benefit from resize to cut the vectors down to 50.
 */
template<class T, class Th, class Tc>
std::tuple<std::vector<AuxT<T>>, std::vector<AuxT<T>>>
localVelocitiesRTGrowthRate(size_t startIndex, size_t endIndex, Tc ymin, Tc ymax, const Th* h, const T* y, const Th* vy,
                            const Th* markRamp)
{
    std::vector<AuxT<T>> localUp(endIndex - startIndex);
    std::vector<AuxT<T>> localDown(endIndex - startIndex);

#pragma omp parallel for
    for (size_t i = startIndex; i < endIndex; i++)
    {

        if (markRamp[i] > 0.05 && !sph::fbcCheck(y[i], 2.0 * h[i], ymax, ymin, true))
        {
            localUp[i - startIndex]   = {y[i], vy[i]};
            localDown[i - startIndex] = {y[i], vy[i]};
        }
    }

    auto endUp   = std::remove_if(localUp.begin(), localUp.end(), invalidAuxTEntry<T>());
    auto endDown = std::remove_if(localDown.begin(), localDown.end(), invalidAuxTEntry<T>());

    std::sort(localUp.begin(), endUp, greaterRT());
    std::sort(localDown.begin(), endDown, lowerRT());

    localUp.resize(50);
    localDown.resize(50);

    return {localUp, localDown};
}

/*! @brief global calculation of the growth rate
 *
 * @tparam        T            double or float
 * @tparam        Dataset
 * @tparam        Box
 * @param[in]     startIndex   first locally assigned particle index of buffers in @p d
 * @param[in]     endIndex     last locally assigned particle index of buffers in @p d
 * @param[in]     d            particle data set
 * @param[in]     box          bounding box
 */
template<typename T, class Dataset>
util::tuple<T, T, T, T> computeVelocitiesRTGrowthRate(size_t startIndex, size_t endIndex, Dataset& d, MPI_Comm comm,
                                                      const cstone::Box<T>& box)
{
    std::tuple<std::vector<AuxT<T>>, std::vector<AuxT<T>>> localRet;
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        std::tie(std::get<0>(localRet), std::get<1>(localRet)) =
            localGrowthRateRTGpu(startIndex, endIndex, box.ymin(), box.ymax(), rawPtr(d.devData.h), rawPtr(d.devData.y),
                                 rawPtr(d.devData.vy), rawPtr(d.devData.markRamp));
    }
    else
    {
        localRet = localVelocitiesRTGrowthRate(startIndex, endIndex, box.ymin(), box.ymax(), d.h.data(), d.y.data(),
                                               d.vy.data(), d.markRamp.data());
    }

    std::vector<AuxT<T>> localUp   = util::get<0>(localRet);
    std::vector<AuxT<T>> localDown = util::get<1>(localRet);

    int rootRank = 0;
    int mpiranks;

    MPI_Comm_size(comm, &mpiranks);
    size_t rootsize = 50 * mpiranks;

    std::vector<AuxT<T>> globalUp(rootsize);
    std::vector<AuxT<T>> globalDown(rootsize);

    /* Create a MPI type for struct AuxT */
    const int    nitems          = 2;
    int          blocklengths[2] = {1, 1};
    MPI_Datatype types[2]        = {MpiType<T>{}, MpiType<T>{}};
    MPI_Datatype mpi_AuxT_type;
    MPI_Aint     offsets[2];
    offsets[0] = offsetof(AuxT<T>, pos);
    offsets[1] = offsetof(AuxT<T>, vel);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_AuxT_type);
    MPI_Type_commit(&mpi_AuxT_type);

    MPI_Gather(localUp.data(), 50, mpi_AuxT_type, globalUp.data(), 50, mpi_AuxT_type, rootRank, comm);
    MPI_Gather(localDown.data(), 50, mpi_AuxT_type, globalDown.data(), 50, mpi_AuxT_type, rootRank, comm);

    int rank;
    MPI_Comm_rank(comm, &rank);

    T vy_max = 0.;
    T vy_min = 0.;

    if (rank == 0)
    {
        std::sort(globalUp.begin(), globalUp.end(), greaterRT());
        std::sort(globalDown.begin(), globalDown.end(), lowerRT());

        globalUp.resize(50);
        globalDown.resize(50);

        for (size_t i = 0; i < 50; i++)
        {
            vy_max += globalUp[i].vel;
            vy_min += globalDown[i].vel;
        }
    }

    return {vy_max / T(50.), vy_min / T(50.), globalUp[0].pos, globalDown[0].pos};
}

//! @brief Observables that includes times and velocities Rayleigh-Taylor growth rate
template<class Dataset>
class TimeVelocitiesGrowthRT : public IObservables<Dataset>
{
    std::ostream& constantsFile;

public:
    TimeVelocitiesGrowthRT(std::ostream& constPath)
        : constantsFile(constPath)
    {
    }

    using T = typename Dataset::RealType;

    void computeAndWrite(Dataset& simData, size_t firstIndex, size_t lastIndex, cstone::Box<T>& box)
    {
        int rank;
        MPI_Comm_rank(simData.comm, &rank);
        auto& d = simData.hydro;

        computeConservedQuantities(firstIndex, lastIndex, d, simData.comm);

        auto [vy_max, vy_min, pos_max, pos_min] =
            computeVelocitiesRTGrowthRate<T>(firstIndex, lastIndex, d, simData.comm, box);

        if (rank == 0)
        {
            fileutils::writeColumns(constantsFile, ' ', d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav,
                                    d.linmom, d.angmom, vy_min, vy_max, pos_min, pos_max);
        }
    }
};

} // namespace sphexa
