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
 * @brief calculates correction terms for fixed boundary conditions
 *
 * These calculations are taken from Herant, Dirty Tricks for SPH (1994)
 *
 * @author Lukas Schmidt
 */

#pragma once

#include "math.hpp"
namespace sphexa
{
namespace sph
{

//! @brief calculates the correction term for the density
template<class T>
T magCorrection(T hi, T dist)
{
    T delta = dist / hi;
    T corr;
    if (delta <= 1.0)
    {
        corr = (30.0 - 42.0 * delta + 20.0 * std::pow(delta, 3) - 9.0 * std::pow(delta, 5) + 3.0 * std::pow(delta, 6)) / 60.0;
    }
    else
    {
        corr = (32.0 - 48.0 * delta + 40.0 * std::pow(delta, 3) - 30.0 * std::pow(delta, 4) + 9.0 * std::pow(delta, 5) - std::pow(delta, 6)) / 60.0;
    }
    return 1 / (1.0 - corr);
}

//! @brief calculates the correction term for the pressure gradients
template<class T>
T gradCorrection(T hi, T pi, T rhoi, T dist, bool fbc)
{
    if(!fbc) return 0.0;
    else
    {
        T delta = dist / hi;
        T corr;
        if (delta <= 1.0)
        {
            corr = pi / rhoi *
                   (-(14.0 - 20.0 * delta * delta + 15.0 * std::pow(delta, 4) - 6 * std::pow(delta, 5)) / 20.0 * hi);
        }
        else
        {
            corr = pi / rhoi *
                   (-(16.0 - 40.0 * delta * delta + 40.0 * std::pow(delta, 3) - 15.0 * std::pow(delta, 4) +
                      2.0 * std::pow(delta, 5)) /
                    20 * hi);
        }
        return corr;
    }

}

//! @brief
template<class T>
T distToFbc(T xi, T yi, T zi, const cstone::Box<T> box)
{
    if(box.fbcX())
    {
        T distXmax = std::abs(box.xmax() - xi);
        T distXmin = std::abs(box.xmin() - xi);

        if(distXmax < distXmin) return distXmax;
        else return distXmin;
    }

    if(box.fbcY())
    {
        T distYmax = std::abs(box.ymax() - yi);
        T distYmin = std::abs(box.ymin() - yi);

        if(distYmax < distYmin) return distYmax;
        else return distYmin;
    }

    if(box.fbcZ())
    {
        T distZmax = std::abs(box.zmax() - zi);
        T distZmin = std::abs(box.zmin() - zi);

        if(distZmax < distZmin) return distZmax;
        else return distZmin;
    }

    throw std::runtime_error("this should not happen");
}
}
}


