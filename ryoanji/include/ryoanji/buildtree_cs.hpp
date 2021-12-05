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
 * @brief  Build a tree for Ryoanji with the cornerstone framework
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>

#include "cstone/tree/octree.cuh"
#include "cstone/tree/octree_internal.cuh"

#include "ryoanji/types.h"

template<class KeyType>
__global__ void convertTree(cstone::OctreeGpuDataView<KeyType> cstoneTree, const cstone::LocalParticleIndex* layout,
                            CellData* ryoanjiTree)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < cstoneTree.numInternalNodes + cstoneTree.numLeafNodes)
    {
        cstone::LocalParticleIndex firstParticle = 0;
        cstone::LocalParticleIndex lastParticle  = 0;

        cstone::TreeNodeIndex child = 0;
        int numChildren             = 1;

        bool isLeaf = (cstoneTree.childOffsets[tid] == 0);
        if (!isLeaf)
        {
            child       = cstoneTree.childOffsets[tid];
            numChildren = 8;
        }
        else
        {
            cstone::TreeNodeIndex leafIndex = cstoneTree.nodeOrder[tid] - cstoneTree.numInternalNodes;
            assert(leafIndex >= 0);
            firstParticle = layout[leafIndex];
            lastParticle  = layout[leafIndex + 1];
        }

        unsigned level = cstone::decodePrefixLength(cstoneTree.prefixes[tid]) / 3;
        unsigned parentIdx = (tid == 0) ? 0 : cstoneTree.parents[(tid - 1) / 8];
        ryoanjiTree[tid] =
            CellData(level, parentIdx, firstParticle, lastParticle - firstParticle, child, numChildren);
    }
}

template<class KeyType, class T>
__global__ void
computeSfcKeysRealKernel(KeyType* keys, const fvec4* bodies, size_t numKeys, const cstone::Box<T> box)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numKeys)
    {
        keys[tid] = cstone::sfc3D<cstone::HilbertKey<KeyType>>(bodies[tid][0], bodies[tid][1], bodies[tid][2], box);
    }
}

template<class KeyType>
class TreeBuilder
{
public:
    TreeBuilder() : bucketSize_(64) { }

    TreeBuilder(unsigned ncrit) : bucketSize_(ncrit) { }

    cstone::TreeNodeIndex update(fvec4* bodies, size_t numBodies, const Box& box, cudaVec<CellData>& ryoanjiTree,
                                 cudaVec<int2>& levelRange)
    {
        unsigned bucketSize = 64;

        using T = fvec4::value_type;

        thrust::device_vector<KeyType> d_keys(numBodies);

        cstone::Box<T> csBox(box.X[0] - box.R,
                             box.X[0] + box.R,
                             box.X[1] - box.R,
                             box.X[1] + box.R,
                             box.X[2] - box.R,
                             box.X[2] + box.R,
                             false,
                             false,
                             false);

        {
            constexpr unsigned numThreads = 256;
            unsigned numBlocks            = (numBodies - 1) / numThreads + 1;
            computeSfcKeysRealKernel<<<numBlocks, numThreads>>>(
                thrust::raw_pointer_cast(d_keys.data()), bodies, numBodies, csBox);
        }

        thrust::sort_by_key(thrust::device, d_keys.begin(), d_keys.end(), bodies);

        if (d_tree_.size() == 0)
        {
            // initial guess on first call. use previous tree as guess on subsequent calls
            d_tree_   = std::vector<KeyType>{0, cstone::nodeRange<KeyType>(0)};
            d_counts_ = std::vector<unsigned>{unsigned(numBodies)};
        }

        while (!cstone::updateOctreeGpu(thrust::raw_pointer_cast(d_keys.data()),
                                        thrust::raw_pointer_cast(d_keys.data()) + d_keys.size(),
                                        bucketSize,
                                        d_tree_,
                                        d_counts_,
                                        tmpTree_,
                                        workArray_));

        cstone::OctreeGpuDataAnchor<KeyType> octreeGpuData;
        octreeGpuData.resize(cstone::nNodes(d_tree_));
        cstone::buildInternalOctreeGpu(thrust::raw_pointer_cast(d_tree_.data()), octreeGpuData.getData());

        cstone::TreeNodeIndex numNodes = octreeGpuData.numInternalNodes + octreeGpuData.numLeafNodes;

        ryoanjiTree.alloc(numNodes, true);

        thrust::device_vector<cstone::LocalParticleIndex> d_layout(d_counts_.size() + 1);
        thrust::copy(d_counts_.begin(), d_counts_.end(), d_layout.begin());
        thrust::exclusive_scan(thrust::device,
                               thrust::raw_pointer_cast(d_layout.data()),
                               thrust::raw_pointer_cast(d_layout.data()) + d_layout.size(),
                               thrust::raw_pointer_cast(d_layout.data()));

        {
            constexpr unsigned numThreads = 256;
            unsigned numBlocks            = (numNodes - 1) / numThreads + 1;
            convertTree<<<numBlocks, numThreads>>>(
                octreeGpuData.getData(), thrust::raw_pointer_cast(d_layout.data()), ryoanjiTree.d());
        }

        thrust::host_vector<int> h_levelRange = octreeGpuData.levelRange;

        int numLevels = 0;
        for (int level = 1; level <= cstone::maxTreeLevel<KeyType>{}; ++level)
        {
            if (h_levelRange[level + 1] == 0)
            {
                numLevels = level - 1;
                break;
            }

            levelRange[level].x = h_levelRange[level];
            levelRange[level].y = h_levelRange[level + 1];
        }

        return numLevels;
    }

private:
    unsigned bucketSize_;

    thrust::device_vector<KeyType> d_tree_;
    thrust::device_vector<unsigned> d_counts_;

    thrust::device_vector<KeyType> tmpTree_;
    thrust::device_vector<cstone::TreeNodeIndex> workArray_;
};
