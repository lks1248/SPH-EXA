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
        d.x[i] /= 16.;
        d.y[i] /= 16.;
        d.z[i] /= 16.;

        d.vy[i] = omega0 * (1. -std::cos(4 * M_PI * d.x[i])) * (1. -std::cos(4 * M_PI * d.y[i] / 3.));

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

template<class T, class Dataset>
auto makeHalfDenseTemplateRT(std::vector<T> x, std::vector<T> y, std::vector<T> z, size_t blockSize)
{
    using KeyType = typename Dataset::KeyType;

    std::vector<T> xHalf, yHalf, zHalf;
    xHalf.reserve(blockSize);
    yHalf.reserve(blockSize);
    zHalf.reserve(blockSize);
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

    for (size_t i = 0; i < blockSize; i += 2)
    {
        xHalf.push_back(x[i]);
        yHalf.push_back(y[i]);
        zHalf.push_back(z[i]);
    }

    xHalf.shrink_to_fit();
    yHalf.shrink_to_fit();
    zHalf.shrink_to_fit();

    return std::make_tuple(xHalf, yHalf, zHalf);
}

/*!
 * @brief assembles the global Rayleigh-Taylor initial conditions
 *
 * @params x_HD, y_HD, z_HD:     x, y and z coordinate vector of the high density template
 * @params x_LD, y_LD, z_HD:     x, y and z coordinate vector of the low density template
 */
template<class T, class Dataset>
void assembleRayleighTaylor(std::vector<T>& x_HD, std::vector<T>& y_HD, std::vector<T>& z_HD, std::vector<T>& x_LD,
                            std::vector<T>& y_LD, std::vector<T>& z_LD, Dataset& d, size_t start, size_t end,
                            size_t xBlocks, size_t yBlocks, size_t zBlocks)
{
    for (size_t x = 0; x < xBlocks; x++)
    {
        for (size_t y = 0; y < yBlocks; y++)
        {
            for (size_t z = 0; z < zBlocks; z++)
            {
                T xFloat = static_cast<T>(x);
                T yFloat = static_cast<T>(y);
                T zFloat = static_cast<T>(z);

                cstone::Box<T> temp(xFloat, xFloat + 1.0, yFloat, yFloat + 1.0, zFloat, zFloat + 1.0,
                                    cstone::BoundaryType::open, cstone::BoundaryType::open, cstone::BoundaryType::open);

                if (y < yBlocks / 2)
                {
                    assembleCube<T>(start, end, temp, 1, x_LD, y_LD, z_LD, d.x, d.y, d.z);
                }
                else
                {
                    assembleCube<T>(start, end, temp, 1, x_HD, y_HD, z_HD, d.x, d.y, d.z);
                }
            }
        }
    }
}

std::map<std::string, double> RayleighTaylorConstants()
{
    return {{"rhoUp", 2.},
            {"rhoDown", 1.},
            {"gamma", 1.4},
            {"firstTimeStep", 1e-9},
            {"p0", 2.5},
            {"y0", 0.75},
            {"omega0", 0.0025},
            {"ay0", -0.5},
            {"blockSize", 0.0625},
            {"xSize", 0.5},
            {"ySize", 1.5},
            {"zSize", 0.0625}
    };
}

template<class Dataset>
class RayleighTaylorGlass : public ISimInitializer<Dataset>
{
    std::string                   glassBlock;
    std::map<std::string, double> constants_;

public:
    RayleighTaylorGlass(std::string initBlock, std::string propChoice)
        : glassBlock(initBlock)
    {
        if (propChoice != "ve-accel")
        {
            std::cout << "\n ERROR: In RayleighTaylor test (--init RT) the SPH propagator have to be 've-accel', but now it is '" << propChoice << "'. Please, add the option '--prop ve-accel' in your execution.\n" << std::endl;
            exit(-1);
        }
        constants_ = RayleighTaylorConstants();
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData) const override
    {
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;
        auto& d       = simData.hydro;

        T rhoUp = constants_.at("rhoUp");

        T blockSize = constants_.at("blockSize");
        T xSize     = constants_.at("xSize");
        T ySize     = constants_.at("ySize");
        T zSize     = constants_.at("zSize");

        size_t xBlocks = xSize / blockSize;
        size_t yBlocks = ySize / blockSize;
        size_t zBlocks = zSize / blockSize;

        size_t nBlocks    = xBlocks * yBlocks * zBlocks;
        size_t halfBlocks = nBlocks / 2;

        d.ay0 = constants_.at("ay0");

        std::vector<T> xBlock, yBlock, zBlock;
        fileutils::readTemplateBlock(glassBlock, xBlock, yBlock, zBlock);

        cstone::Box<T> globalBox(0,  xSize, 0, ySize, 0, zSize, cstone::BoundaryType::periodic, cstone::BoundaryType::fixed, cstone::BoundaryType::periodic);

        unsigned level             = cstone::log8ceil<KeyType>(100 * numRanks);
        auto     initialBoundaries = cstone::initialDomainSplits<KeyType>(numRanks, level);
        KeyType  keyStart          = initialBoundaries[rank];
        KeyType  keyEnd            = initialBoundaries[rank + 1];

        auto [xHalf, yHalf, zHalf] = makeHalfDenseTemplateRT<T, Dataset>(xBlock, yBlock, zBlock, xBlock.size());
        assembleRayleighTaylor(xBlock, yBlock, zBlock, xHalf, yHalf, zHalf, d, keyStart, keyEnd, xBlocks, yBlocks, zBlocks);

        size_t npartUp      = halfBlocks * xBlock.size();
        T      volumeHD     = xSize * constants_.at("y0") * zSize; // (x_size * y_size * z_size) in the high-density zone
        T      particleMass = volumeHD * rhoUp / npartUp;

        size_t totalNPart = halfBlocks * (xBlock.size() + xHalf.size());
        d.resize(totalNPart);
        initRayleighTaylorFields(d, constants_, particleMass);

        d.numParticlesGlobal = d.x.size();
        MPI_Allreduce(MPI_IN_PLACE, &d.numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, simData.comm);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa
