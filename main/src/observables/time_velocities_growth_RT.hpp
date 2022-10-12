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
 * @brief output and calculate energies and growth rate for Kelvin-Helmholtz tests
 *        This calculation for the growth rate was taken from McNally et al. ApJSS, 201 (2012)
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
#include "io/ifile_writer.hpp"
#include "sph/math.hpp"
#include "cstone/tree/definitions.h"

namespace sphexa
{


template<class T>
struct AuxT
{
    T pos;
    T vel;
};

struct greaterRT
{
template<class AuxT> bool operator()(AuxT const &a, AuxT const &b) const { return a.pos > b.pos; }
};

struct lowerRT
{
template<class AuxT> bool operator()(AuxT const &a, AuxT const &b) const { return a.pos < b.pos; }
};


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
template<class T> util::tuple<std::vector<AuxT<T>>, std::vector<AuxT<T>>>
localVelocitiesRTGrowthRate(size_t startIndex, size_t endIndex, size_t ngmax, size_t n, const T Atmin, const T Atmax, const T ramp,
        const T* y, const T* vy, const T* rho, const cstone::LocalIndex* neighbors, const unsigned* neighborsCount)
{
    std::vector<AuxT<T>> localUp(n);
    std::vector<AuxT<T>> localDown(n);

#pragma omp parallel for
    for (size_t i = startIndex; i < endIndex; i++)
    {
        T mark_ramp = 0.;

        const cstone::LocalIndex* localNeighbors      = neighbors + ngmax * (i - startIndex);
        unsigned                  localNeighborsCount = stl::min(neighborsCount[i], unsigned(ngmax));

        for (unsigned pj = 0; pj < localNeighborsCount; ++pj)
        {
            cstone::LocalIndex j = localNeighbors[pj];

            T Atwood   = (std::abs(rho[i] - rho[j])) / (rho[i] + rho[j]);
            if (Atwood > Atmax)
            {
                mark_ramp += T(1);
            }
            else if (Atwood >= Atmin)
            {
                T sigma_ij = ramp * (Atwood - Atmin);
                mark_ramp += sigma_ij;
            }
        }

        if (mark_ramp > 0.05) {
            localUp[i-startIndex] = {y[i], vy[i]};
            localDown[i-startIndex] = {y[i], vy[i]};
        }
    }

    std::sort(localUp.begin(), localUp.end(), greaterRT());
    std::sort(localDown.begin(), localDown.end(), lowerRT());

    localUp.resize(50);
    localDown.resize(50);

    return{localUp, localDown};
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
template<typename T, class Dataset> util::tuple<T, T>
computeVelocitiesRTGrowthRate(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box, size_t ngmax)
{
    auto [localUp, localDown] = localVelocitiesRTGrowthRate(startIndex, endIndex, ngmax, d.x.size(), d.Atmin, d.Atmax, d.ramp,
            d.y.data(), d.vy.data(), d.prho.data(), d.neighbors.data(), d.nc.data());

    int rootRank = 0;
    int mpiranks;

    MPI_Comm_size(d.comm, &mpiranks);
    size_t rootsize = 50 * mpiranks;

    std::vector<AuxT<T>> globalUp(rootsize);
    std::vector<AuxT<T>> globalDown(rootsize);

    /* Create a MPI type for struct AuxT */
    const int    nitems=2;
    int          blocklengths[2] = {1,1};
    MPI_Datatype types[2] = {MpiType<T>{}, MpiType<T>{}};
    MPI_Datatype mpi_AuxT_type;
    MPI_Aint     offsets[2];
    offsets[0] = offsetof(AuxT<T>, pos);
    offsets[1] = offsetof(AuxT<T>, vel);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_AuxT_type);
    MPI_Type_commit(&mpi_AuxT_type);

    MPI_Gather(localUp.data(), 50, mpi_AuxT_type, globalUp.data(), 50, mpi_AuxT_type, rootRank, d.comm);
    MPI_Gather(localDown.data(), 50, mpi_AuxT_type, globalDown.data(), 50, mpi_AuxT_type, rootRank, d.comm);

    int rank;
    MPI_Comm_rank(d.comm, &rank);

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

    return {vy_max/T(50.), vy_min/T(50.)};
}

//! @brief Observables that includes times and velocities Rayleigh-Taylor growth rate
template<class Dataset>
class TimeVelocitiesGrowthRT : public IObservables<Dataset>
{
    std::ofstream& constantsFile;
    size_t         ngmax;

public:
    TimeVelocitiesGrowthRT(std::ofstream& constPath, size_t ngmax)
        : constantsFile(constPath), ngmax(ngmax)
    {
    }

    using T = typename Dataset::RealType;

    void computeAndWrite(Dataset& d, size_t firstIndex, size_t lastIndex, cstone::Box<T>& box)
    {
        auto [vy_max, vy_min] = computeVelocitiesRTGrowthRate<T>(firstIndex, lastIndex, d, box, ngmax);

        int rank;
        MPI_Comm_rank(d.comm, &rank);

        if (rank == 0)
        {
            fileutils::writeColumns(constantsFile, ' ', d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav,
                                    d.linmom, d.angmom, vy_min, vy_max);
        }
    }
};

} // namespace sphexa
