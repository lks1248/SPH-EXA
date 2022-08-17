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
 * @brief Utility functions to resolve names of particle fields to pointers
 */

#pragma once

#include <vector>
#include <variant>

#include "cstone/util/traits.hpp"
#include "particles_data_stubs.hpp"

namespace sphexa
{

//! @brief extract a vector of pointers to particle fields for file output
template<class Dataset>
auto getOutputArrays(Dataset& dataset)
{
    auto fieldPointers = dataset.data();
    using FieldType    = std::variant<float*, double*, int*, unsigned*, uint64_t*>;

    std::vector<FieldType> outputFields;
    outputFields.reserve(dataset.outputFieldIndices.size());

    for (int i : dataset.outputFieldIndices)
    {
        if (!dataset.isAllocated(i))
        {
            throw std::runtime_error("Cannot output field " + std::string(dataset.fieldNames[i]) +
                                     ", because it is not active.");
        }
        std::visit([&outputFields](auto& arg) { outputFields.push_back(arg->data()); }, fieldPointers[i]);
    }
    return outputFields;
}

//! @brief resizes the neighbors list, only used in the CPU version
template<class Dataset>
void resizeNeighbors(Dataset& d, size_t size)
{
    double growthRate = 1.05;
    //! If we have a GPU, neighbors are calculated on-the-fly, so we don't need space to store them
    reallocate(d.neighbors, cstone::HaveGpu<typename Dataset::AcceleratorType>{} ? 0 : size, growthRate);
}

//! @brief compile-time index look-up of a string literal in a list of strings
template<class Array>
constexpr size_t getFieldIndex(std::string_view field, const Array& fieldNames)
{
    for (size_t i = 0; i < fieldNames.size(); ++i)
    {
        if (field == fieldNames[i]) { return i; }
    }
    return fieldNames.size();
}

//! @brief translate string fields to an array with the field indices
template<class Array1, class Array2>
constexpr auto fieldNamesToIndices(const Array1& queries, const Array2& fieldNames)
{
    typename util::SwapArg<Array1, size_t>::type ret;
    for (size_t i = 0; i < queries.size(); ++i)
    {
        ret[i] = getFieldIndex(queries[i], fieldNames);
    }
    return ret;
}

/*! @brief Look up indices of a (runtime-variable) number of field names
 *
 * @tparam     Array
 * @param[in]  subsetNames  array of strings of field names to look up in @p allNames
 * @param[in]  allNames     array of strings with names of all fields
 * @return                  the indices of @p subsetNames in @p allNames
 */
template<class Array>
std::vector<int> fieldStringsToInt(const std::vector<std::string>& subsetNames, const Array& allNames)
{
    std::vector<int> subsetIndices(subsetNames.size());
    for (size_t i = 0; i < subsetNames.size(); ++i)
    {
        subsetIndices[i] = getFieldIndex(subsetNames[i], allNames);
        if (subsetIndices[i] == allNames.size()) { throw std::runtime_error("Field not found: " + subsetNames[i]); }
    }
    return subsetIndices;
}

template<auto Indices, class Tuple, size_t... Is>
auto accessFields_helper(Tuple&& tuple, std::index_sequence<Is...>)
{
    return std::tie(std::get<Indices[Is]>(std::forward<Tuple>(tuple))...);
}

//! @brief return a tuple of references to the elements of @p tuple specified by compile-time Indices
template<auto Indices, class Tuple>
auto accessFields(Tuple&& tuple)
{
    constexpr size_t numIndices = Indices.size();
    return accessFields_helper<Indices>(std::forward<Tuple>(tuple), std::make_index_sequence<numIndices>{});
}

} // namespace sphexa
