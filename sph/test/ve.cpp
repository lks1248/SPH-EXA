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
 * @brief SPH density kernel tests
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include "gtest/gtest.h"

#include "cstone/util/tuple_util.hpp"

#include "sph/hydro_ve/av_switches_kern.hpp"
#include "sph/hydro_ve/divv_curlv_kern.hpp"
#include "sph/hydro_ve/iad_kern.hpp"
#include "sph/hydro_ve/momentum_energy_kern.hpp"
#include "sph/hydro_ve/ve_def_gradh_kern.hpp"
#include "sph/hydro_ve/xmass_kern.hpp"
#include "sph/sph_kernel_tables.hpp"
#include "sph/table_lookup.hpp"
#include "../../main/src/io/file_utils.hpp"

#include <iomanip>

using namespace sph;

//! @brief test fixture, defining and initializing all data needed to call SPH kernels
class SphKernelTests : public testing::Test
{
protected:
    using T = double;
    constexpr static auto precision{std::numeric_limits<T>::digits10 + 1};

    void SetUp() override
    {
        neighbors.resize(neighborsCount);
        std::iota(neighbors.begin(), neighbors.end(), 1);

        wh  = tabulateFunction<T, lt::kTableSize>(getSphKernel(kernelType, sincIndex), 0.0, 2.0);
        whd = tabulateFunction<T, lt::kTableSize>(getSphKernelDerivative(kernelType, sincIndex), 0.0, 2.0);

        auto fieldVectors =
            std::tie(x, y, z, h, m, gradh, rho0, sumwhrho0, vx, vy, vz, c, p, u, divv, alpha, c11, c12, c13, c22, c23,
                     c33, dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz, sumwh, xm, kx, prho, rho);

        // resize all vectors to npart
        util::for_each_tuple([this](auto& vec) { vec.resize(npart); }, fieldVectors);

        // read example data into the specified fields
        std::apply(
            [this](auto&&... vecs)
            { sphexa::fileutils::readAscii(path, npart, std::vector<T*>{vecs.data()...}); }, std::tie(
                                                                                                 x, y, z, vx, vy, vz, h,
                                                                                                 c, c11, c12, c13, c22,
                                                                                                 c23, c33, p, gradh,
                                                                                                 rho0, sumwhrho0, sumwh,
                                                                                                 rho,
                                                                                                 /*
       dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz,*/ alpha, u, divv));

        std::fill(m.begin(), m.end(), mpart);

        for (unsigned i = 0; i < npart; i++)
        {
            xm[i]   = mpart / rho0[i];
            kx[i]   = K * xm[i] / std::pow(h[i], 3);
            prho[i] = p[i] / (kx[i] * m[i] * m[i] * gradh[i]);
        }
    }

    static auto box() { return cstone::Box<T>(-1.e9, 1.e9, cstone::BoundaryType::open); }

    T                             sincIndex  = 6.0;
    SphKernelType                 kernelType = SphKernelType::sinc_n1_sinc_n2;
    std::array<T, lt::kTableSize> wh{0}, whd{0};

    // T K              = sphynx_3D_k(sincIndex);
    T K              = kernel_3D_k(getSphKernel(kernelType, sincIndex), 2.0);
    T alphamin       = 0.05;
    T alphamax       = 1.0;
    T decay_constant = 0.2;
    // T mpart          = 3.781038064465603e26;
    T mpart = 1e-6;
    T dt    = 0.3;
    T Atmin = 0.1;
    T Atmax = 0.2;
    T ramp  = 1.0 / (Atmax - Atmin);

    uint64_t                        npart          = 145;
    unsigned                        neighborsCount = npart - 1;
    std::vector<cstone::LocalIndex> neighbors;

    std::vector<T> x, y, z, h, m, gradh, rho0, sumwhrho0, vx, vy, vz, c, p, u, divv, alpha, c11, c12, c13, c22, c23,
        c33, dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz, sumwh, xm, kx, prho, rho;

    std::string             path    = "testparticles/part_910568.d";
    std::string             outPath = path.substr(0, path.find('.')) + "_results.txt";
    std::ios_base::openmode append  = std::ofstream::app;
};

TEST_F(SphKernelTests, AVSwitches)
{
    T newAlpha = AVswitchesJLoop(0, K, box(), neighbors.data(), neighborsCount, x.data(), y.data(), z.data(), vx.data(),
                                 vy.data(), vz.data(), h.data(), c.data(), c11.data(), c12.data(), c13.data(),
                                 c22.data(), c23.data(), c33.data(), wh.data(), whd.data(), kx.data(), xm.data(),
                                 divv.data(), dt, alphamin, alphamax, decay_constant, alpha[0]);

    std::ofstream fout(outPath, std::ofstream::out);
    if (fout.is_open()) { fout << "alpha \t" << std::setprecision(precision) << newAlpha << std::endl; }
    else { throw std::runtime_error("Can't open file at path: " + path); }
    fout.close();

    EXPECT_NEAR(newAlpha, 0.93941905320351171, 2e-9);
}
/*
TEST_F(SphKernelTests, Divv_Curlv)
{
    auto [divv, curlv, dV11, dV12, dV13, dV22, dV23, dV33] = std::array<T, 8>{-1, -1, -1, -1, -1, -1, -1, -1};

    divV_curlVJLoop(0, K, box(), neighbors.data(), neighborsCount, x.data(), y.data(), z.data(), vx.data(), vy.data(),
                    vz.data(), h.data(), c11.data(), c12.data(), c13.data(), c22.data(), c23.data(), c33.data(),
                    wh.data(), whd.data(), kx.data(), xm.data(), &divv, &curlv, &dV11, &dV12, &dV13, &dV22, &dV23,
                    &dV33, true);

    std::ofstream fout(outPath, append);
    if(fout.is_open())
    {
        fout << "divv \t" << divv << std::endl;
        fout << "curlv \t" << curlv << std::endl;
        fout << "dV11 \t" << dV11 << std::endl;
        fout << "dV12 \t" << dV12 << std::endl;
        fout << "dV13 \t" << dV13 << std::endl;
        fout << "dV22 \t" << dV22 << std::endl;
        fout << "dV23 \t" << dV23 << std::endl;
        fout << "dV33 \t" << dV33 << std::endl;
    } else { throw std::runtime_error("Can't open file at path: " + path); }
    fout.close();

    EXPECT_NEAR(divv, 3.3760353440920682e-2, 2e-9);
    EXPECT_NEAR(curlv, 3.7836647734377962e-2, 2e-9);
    EXPECT_NEAR(dV11, 0.0013578323369918166, 2e-9);
    EXPECT_NEAR(dV12, 0.02465266861727711, 2e-9);
    EXPECT_NEAR(dV13, -0.0046604174274769167, 2e-9);
    EXPECT_NEAR(dV22, 0.022556438947324862, 2e-9);
    EXPECT_NEAR(dV23, 0.0097704904179710741, 2e-9);
    EXPECT_NEAR(dV33, 0.0098460821566040066, 2e-9);
}
*/
TEST_F(SphKernelTests, IAD)
{
    // fill with invalid initial value to make sure that the kernel overwrites it instead of add to it
    std::vector<T> iad(6, -1);

    // compute the 6 tensor components for particle 0
    IADJLoop(0, K, box(), neighbors.data(), neighborsCount, x.data(), y.data(), z.data(), h.data(), wh.data(),
             whd.data(), xm.data(), kx.data(), &iad[0], &iad[1], &iad[2], &iad[3], &iad[4], &iad[5]);

    std::ofstream fout(outPath, append);
    if (fout.is_open())
    {
        fout << "c11 \t" << std::setprecision(precision) << iad[0] << std::endl;
        fout << "c12 \t" << std::setprecision(precision) << iad[1] << std::endl;
        fout << "c13 \t" << std::setprecision(precision) << iad[2] << std::endl;
        fout << "c22 \t" << std::setprecision(precision) << iad[3] << std::endl;
        fout << "c23 \t" << std::setprecision(precision) << iad[4] << std::endl;
        fout << "c33 \t" << std::setprecision(precision) << iad[5] << std::endl;
    }
    else { throw std::runtime_error("Can't open file at path: " + path); }
    fout.close();

    EXPECT_NEAR(iad[0], 1.9296619855715329e-18, 1e-10);
    EXPECT_NEAR(iad[1], -1.7838691836843698e-20, 1e-10);
    EXPECT_NEAR(iad[2], -1.2892885646884301e-20, 1e-10);
    EXPECT_NEAR(iad[3], 1.9482845913025683e-18, 1e-10);
    EXPECT_NEAR(iad[4], 1.635410357476855e-20, 1e-10);
    EXPECT_NEAR(iad[5], 1.9246939006338132e-18, 1e-10);
}

template<class T>
void symmetrizeGradV(util::array<const T*, 9> dV, util::array<T*, 6> sdV, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        sdV[0][i] = dV[0][i];
        sdV[1][i] = dV[1][i] + dV[3][i];
        sdV[2][i] = dV[2][i] + dV[6][i];
        sdV[3][i] = dV[4][i];
        sdV[4][i] = dV[5][i] + dV[7][i];
        sdV[5][i] = dV[8][i];
    }
}

TEST_F(SphKernelTests, MomentumEnergy)
{

    std::vector<T> dV11(npart), dV12(npart), dV13(npart), dV22(npart), dV23(npart), dV33(npart);
    symmetrizeGradV<T>({dvxdx.data(), dvxdy.data(), dvxdz.data(), dvydx.data(), dvydy.data(), dvydz.data(),
                        dvzdx.data(), dvzdy.data(), dvzdz.data()},
                       {dV11.data(), dV12.data(), dV13.data(), dV22.data(), dV23.data(), dV33.data()}, npart);
    /*
        { // test with AV cleaning
            auto [du, grad_Px, grad_Py, grad_Pz, maxvsignal] = std::array<T, 5>{-1, -1, -1, -1, -1};

            momentumAndEnergyJLoop<true>(0, K, box(), neighbors.data(), neighborsCount, x.data(), y.data(), z.data(),
                                         vx.data(), vy.data(), vz.data(), h.data(), m.data(), prho.data(),
                                         (const T*)nullptr, c.data(), c11.data(), c12.data(), c13.data(), c22.data(),
                                         c23.data(), c33.data(), Atmin, Atmax, ramp, wh.data(), kx.data(), xm.data(),
                                         alpha.data(), dV11.data(), dV12.data(), dV13.data(), dV22.data(), dV23.data(),
                                         dV33.data(), &grad_Px, &grad_Py, &grad_Pz, &du, &maxvsignal);


            std::ofstream fout(outPath, append);
            if(fout.is_open())
            {
                fout << "With AV-cleaning: "<< std::endl;
                fout << "ax \t" << grad_Px << std::endl;
                fout << "ay \t" << grad_Py << std::endl;
                fout << "az \t" << grad_Pz << std::endl;
                fout << "du \t" << du << std::endl;
                fout << "maxvsignal \t" << maxvsignal << "\n" << std::endl;

            } else { throw std::runtime_error("Can't open file at path: " + path); }
            fout.close();

            EXPECT_NEAR(grad_Px, -505548.68073726865, 0.023);
            EXPECT_NEAR(grad_Py, 303384.91384746187, 0.053);
            EXPECT_NEAR(grad_Pz, -1767463.9739728321, 0.043);
            EXPECT_NEAR(du, 8.5525242525359648e12, 7.1e5);
            EXPECT_NEAR(maxvsignal, 26490876.319252387, 1e-6);
        } */
    { // test without AV cleaning
        auto [du, grad_Px, grad_Py, grad_Pz, maxvsignal] = std::array<T, 5>{-1, -1, -1, -1, -1};

        momentumAndEnergyJLoop<false>(0, K, box(), neighbors.data(), neighborsCount, x.data(), y.data(), z.data(),
                                      vx.data(), vy.data(), vz.data(), h.data(), m.data(), prho.data(),
                                      (const T*)nullptr, c.data(), c11.data(), c12.data(), c13.data(), c22.data(),
                                      c23.data(), c33.data(), Atmin, Atmax, ramp, wh.data(), kx.data(), xm.data(),
                                      alpha.data(), dV11.data(), dV12.data(), dV13.data(), dV22.data(), dV23.data(),
                                      dV33.data(), &grad_Px, &grad_Py, &grad_Pz, &du, &maxvsignal);

        std::ofstream fout(outPath, append);
        if (fout.is_open())
        {
            fout << "Without AV-cleaning: " << std::endl;
            fout << "ax \t" << std::setprecision(precision) << grad_Px << std::endl;
            fout << "ay \t" << std::setprecision(precision) << grad_Py << std::endl;
            fout << "az \t" << std::setprecision(precision) << grad_Pz << std::endl;
            fout << "du \t" << std::setprecision(precision) << du << std::endl;
            fout << "maxvsignal \t" << std::setprecision(precision) << maxvsignal << "\n" << std::endl;
        }
        else { throw std::runtime_error("Can't open file at path: " + path); }
        fout.close();

        EXPECT_NEAR(grad_Px, -521261.07791667967, 0.022);
        EXPECT_NEAR(grad_Py, -74471.016515749841, 0.064);
        EXPECT_NEAR(grad_Pz, -1730426.827721074, 0.042);
        EXPECT_NEAR(du, 7.1838438980436924e12, 3.1e5);
        EXPECT_NEAR(maxvsignal, 26490876.319252387, 1e-6);
    }
}

TEST_F(SphKernelTests, VeDefGradh)
{
    auto [kx, gradh] = veDefGradhJLoop(0, K, box(), neighbors.data(), neighborsCount, x.data(), y.data(), z.data(),
                                       h.data(), m.data(), wh.data(), whd.data(), xm.data());

    T density = kx * m[0] / xm[0];

    std::ofstream fout(outPath, append);
    if (fout.is_open())
    {
        fout << "rho \t" << std::setprecision(precision) << density << std::endl;
        fout << "gradh \t" << std::setprecision(precision) << gradh << std::endl;
        fout << "kx \t" << std::setprecision(precision) << kx << std::endl;
    }
    else { throw std::runtime_error("Can't open file at path: " + path); }
    fout.close();

    // EXPECT_NEAR(density, 3.4662283566584293e1, 8e-7);
    // EXPECT_NEAR(gradh, 0.98699067585409861, 5e-7);
    // EXPECT_NEAR(kx, 1.0042661134076782, 3e-7);
    EXPECT_NEAR(density, rho[0], 3e-7);
}

TEST_F(SphKernelTests, XMass)
{
    T xmass = xmassJLoop(0, K, box(), neighbors.data(), neighborsCount, x.data(), y.data(), z.data(), h.data(),
                         m.data(), wh.data(), whd.data());
    T rho0i = m[0] / xmass;

    std::ofstream fout(outPath, append);
    if (fout.is_open())
    {
        fout << "rho0 \t" << std::setprecision(precision) << rho0i << std::endl;
        fout << "xmass \t" << std::setprecision(precision) << xmass << std::endl;
    }
    else { throw std::runtime_error("Can't open file at path: " + path); }
    fout.close();

    EXPECT_NEAR(rho0i, 34.515038498081417, 7.33e-7);
    EXPECT_NEAR(xmass, m[0] / rho0i, 1e-10);
    EXPECT_NEAR(xmass, m[0] / rho0[0], m[0] / rho0[0] * 1.e-7);
}
