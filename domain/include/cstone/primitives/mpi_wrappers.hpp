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
 * @brief  A few C++ wrappers for MPI C functions
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <mpi.h>

template<class T>
struct MpiType
{
};

template<>
struct MpiType<double>
{
    operator MPI_Datatype() const noexcept { return MPI_DOUBLE; }
};

template<>
struct MpiType<float>
{
    operator MPI_Datatype() const noexcept { return MPI_FLOAT; }
};

template<>
struct MpiType<char>
{
    operator MPI_Datatype() const noexcept { return MPI_CHAR; }
};

template<>
struct MpiType<unsigned char>
{
    operator MPI_Datatype() const noexcept { return MPI_UNSIGNED_CHAR; }
};

template<>
struct MpiType<short>
{
    operator MPI_Datatype() const noexcept { return MPI_SHORT; }
};

template<>
struct MpiType<unsigned short>
{
    operator MPI_Datatype() const noexcept { return MPI_UNSIGNED_SHORT; }
};

template<>
struct MpiType<int>
{
    operator MPI_Datatype() const noexcept { return MPI_INT; }
};

template<>
struct MpiType<unsigned>
{
    operator MPI_Datatype() const noexcept { return MPI_UNSIGNED; }
};

template<>
struct MpiType<long>
{
    operator MPI_Datatype() const noexcept { return MPI_LONG; }
};

template<>
struct MpiType<unsigned long>
{
    operator MPI_Datatype() const noexcept { return MPI_UNSIGNED_LONG; }
};

template<>
struct MpiType<unsigned long long>
{
    operator MPI_Datatype() const noexcept { return MPI_UNSIGNED_LONG_LONG; }
};

template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
auto mpiSendAsync(T* data, int count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    requests.push_back(MPI_Request{});
    return MPI_Isend(data, count, MpiType<std::decay_t<T>>{}, rank, tag, MPI_COMM_WORLD, &requests.back());
}

//! @brief adaptor to wrap compile-time size arrays into flattened arrays of the underlying type
template<class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
auto mpiSendAsync(T* data, int count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    using ValueType    = typename T::value_type;
    constexpr size_t N = T{}.size();
    ValueType* ptr     = reinterpret_cast<ValueType*>(data);

    return mpiSendAsync(ptr, count * N, rank, tag, requests);
}

template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
auto mpiRecvSync(T* data, int count, int rank, int tag, MPI_Status* status)
{
    return MPI_Recv(data, count, MpiType<std::decay_t<T>>{}, rank, tag, MPI_COMM_WORLD, status);
}

//! @brief adaptor to wrap compile-time size arrays into flattened arrays of the underlying type
template<class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
auto mpiRecvSync(T* data, int count, int rank, int tag, MPI_Status* status)
{
    using ValueType    = typename T::value_type;
    constexpr size_t N = T{}.size();
    ValueType* ptr     = reinterpret_cast<ValueType*>(data);

    return mpiRecvSync(ptr, count * N, rank, tag, status);
}

template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
auto mpiRecvAsync(T* data, int count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    requests.push_back(MPI_Request{});
    return MPI_Irecv(data, count, MpiType<std::decay_t<T>>{}, rank, tag, MPI_COMM_WORLD, &requests.back());
}

//! @brief adaptor to wrap compile-time size arrays into flattened arrays of the underlying type
template<class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
auto mpiRecvAsync(T* data, int count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    using ValueType    = typename T::value_type;
    constexpr size_t N = T{}.size();
    ValueType* ptr     = reinterpret_cast<ValueType*>(data);

    return mpiRecvAsync(ptr, count * N, rank, tag, requests);
}

template<class Ts, class Td, std::enable_if_t<std::is_arithmetic_v<Td>, int> = 0>
auto mpiAllreduce(const Ts* src, Td* dest, int count, MPI_Op op)
{
    return MPI_Allreduce(src, dest, count, MpiType<Td>{}, op, MPI_COMM_WORLD);
}

//! @brief adaptor to wrap compile-time size arrays into flattened arrays of the underlying type
template<class Ts, class Td, std::enable_if_t<!std::is_arithmetic_v<Td>, int> = 0>
auto mpiAllreduce(const Ts* src, Td* dest, int count, MPI_Op op)
{
    using ValueType    = typename Td::value_type;
    constexpr size_t N = Td{}.size();

    using SrcType = std::conditional_t<std::is_same_v<void, Ts>, void, ValueType>;
    auto src_ptr  = reinterpret_cast<const SrcType*>(src);
    auto dest_ptr = reinterpret_cast<ValueType*>(dest);

    return mpiAllreduce(src_ptr, dest_ptr, count * N, op);
}
