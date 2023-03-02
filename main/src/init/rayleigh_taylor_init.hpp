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
 * @brief Rayleigh Taylor simulation data initialization
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 * @author Lukas Schmidt
 *
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/primitives/gather.hpp"
#include "io/mpi_file_utils.hpp"
#include "isim_init.hpp"

#include "utils.hpp"
#include "grid.hpp"

namespace sphexa
{

template<class T, class Dataset>
void initRayleighTaylorFields(Dataset& d, const std::map<std::string, double>& constants, T massPart)
{
    T rhoUp         = constants.at("rhoUp");
    T rhoDown       = constants.at("rhoDown");
    T firstTimeStep = constants.at("firstTimeStep");
    T omega0        = constants.at("omega0");
    T gamma         = constants.at("gamma");
    T p0            = constants.at("p0");
    T y0            = constants.at("y0");

    size_t ng0   = 100;
    T      hUp   = 0.5 * std::cbrt(3. * ng0 * massPart / 4. / M_PI / rhoUp);
    T      hDown = 0.5 * std::cbrt(3. * ng0 * massPart / 4. / M_PI / rhoDown);

    std::fill(d.m.begin(), d.m.end(), massPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mue.begin(), d.mue.end(), 2.0);
    std::fill(d.mui.begin(), d.mui.end(), 10.0);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamax);
    std::fill(d.vx.begin(), d.vx.end(), 0.0);
    std::fill(d.vz.begin(), d.vz.end(), 0.0);

    d.minDt    = firstTimeStep;
    d.minDt_m1 = firstTimeStep;

    auto cv = sph::idealGasCv(d.muiConst, d.gamma);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {

        d.vy[i] = omega0 * (1. - std::cos(4 * M_PI * d.x[i])) * (1. - std::cos(4 * M_PI * d.y[i] / 3.));

        if (d.y[i] < y0)
        {
            T p = p0 + rhoDown * (y0 - d.y[i]);
            T u = p / (rhoDown * (gamma - 1.));

            d.h[i]    = hDown;
            d.temp[i] = u / cv;
        }
        else
        {
            T p = p0 + rhoUp * (y0 - d.y[i]);
            T u = p / (rhoUp * (gamma - 1.));

            d.h[i]    = hUp;
            d.temp[i] = u / cv;
        }

        d.x_m1[i] = d.vx[i] * firstTimeStep;
        d.y_m1[i] = d.vy[i] * firstTimeStep;
        d.z_m1[i] = d.vz[i] * firstTimeStep;
    }
}

std::map<std::string, double> RayleighTaylorConstants()
{
    return {{"rhoUp", 2.},         {"rhoDown", 1.}, {"gamma", 1.4},     {"firstTimeStep", 1e-4},
            {"p0", 2.5},           {"y0", 0.75},    {"omega0", 0.0025}, {"ay0", -0.5},
            {"blockSize", 0.0625}, {"xSize", 0.5},  {"ySize", 1.5},     {"zSize", 0.0625}};
}

template<class Dataset>
class RayleighTaylorGlass : public ISimInitializer<Dataset>
{
    std::string                   glassBlock;
    std::map<std::string, double> constants_;

public:
    RayleighTaylorGlass(std::string initBlock)
        : glassBlock(initBlock)
    {
        constants_ = RayleighTaylorConstants();
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart,
                                                 Dataset& simData) const override
    {

        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;
        auto& d       = simData.hydro;

        if (d.propagator != "RT-ve")
        {
            throw std::runtime_error("ERROR: For the Rayleigh Taylor test (--init RT) the SPH propagator has to be "
                                     "RT-ve. Please restart with '--prop RT-ve'\n");
        }

        T rhoUp = constants_.at("rhoUp");

        T blockSize = constants_.at("blockSize");
        T xSize     = constants_.at("xSize");
        T ySize     = constants_.at("ySize");
        T zSize     = constants_.at("zSize");

        int xBlocks = xSize / blockSize;
        int yBlocks = ySize / blockSize;
        int zBlocks = zSize / blockSize;

        size_t nBlocks    = xBlocks * yBlocks * zBlocks;
        size_t halfBlocks = nBlocks / 2;

        std::vector<T> xBlock, yBlock, zBlock;
        fileutils::readTemplateBlock(glassBlock, xBlock, yBlock, zBlock);

        cstone::Box<T> globalBox(0, xSize, 0, ySize, 0, zSize, cstone::BoundaryType::periodic,
                                 cstone::BoundaryType::fixed, cstone::BoundaryType::periodic);

        unsigned level             = cstone::log8ceil<KeyType>(100 * numRanks);
        auto     initialBoundaries = cstone::initialDomainSplits<KeyType>(numRanks, level);
        KeyType  keyStart          = initialBoundaries[rank];
        KeyType  keyEnd            = initialBoundaries[rank + 1];

        auto [xHalf, yHalf, zHalf] = makeLessDenseTemplate<T>(2, xBlock, yBlock, zBlock);

        int                   multi1D      = std::rint(cbrtNumPart / std::cbrt(xBlock.size()));
        cstone::Vec3<int> multiplicity = {xBlocks * multi1D, yBlocks / 2 * multi1D, multi1D};

        cstone::Box<T> layer1(0, xSize, 0, ySize / 2., 0, zSize, cstone::BoundaryType::periodic,
                              cstone::BoundaryType::periodic, cstone::BoundaryType::periodic);
        cstone::Box<T> layer2(0, xSize, ySize / 2., ySize, 0, zSize, cstone::BoundaryType::periodic,
                              cstone::BoundaryType::periodic, cstone::BoundaryType::periodic);

        assembleCuboid<T>(keyStart, keyEnd, layer1, multiplicity, xHalf, yHalf, zHalf, d.x, d.y, d.z);
        assembleCuboid<T>(keyStart, keyEnd, layer2, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        size_t npartUp  = halfBlocks * xBlock.size();
        T      volumeHD = xSize * constants_.at("y0") * zSize; // (x_size * y_size * z_size) in the high-density zone
        T      particleMass = volumeHD * rhoUp / npartUp;

        d.resize(d.x.size());

        initRayleighTaylorFields(d, constants_, particleMass);
        applyFixedBoundaries(d.y.data(), d.vx.data(), d.vy.data(), d.vz.data(), d.h.data(), globalBox.ymax(),
                             globalBox.ymin(), d.x.size());

        d.numParticlesGlobal = d.x.size();
        MPI_Allreduce(MPI_IN_PLACE, &d.numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, simData.comm);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
