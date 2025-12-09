#ifndef ZERO_OUT_IMPL_H
#define ZERO_OUT_IMPL_H

#include "kernel_operator.h"

using namespace AscendC;

/**
 * @brief ZeroOut Kernel Implementation
 * @tparam T Data type
 * @tparam TILE_ELEMS Number of elements in the temp buffer (Adjustable constexpr)
 */
template<typename T, int32_t TILE_ELEMS = 512>
class KernelZeroOut {
public:
    __aicore__ inline KernelZeroOut() {}

    __aicore__ inline void Init(AscendC::TPipe* pipe, GM_ADDR addr, uint32_t total_num) {
        this->pipe = pipe;
        this->addrGm.SetGlobalBuffer((__gm__ T*)addr);
        this->total_num = total_num;

        // 1. 计算对齐参数 (512 Bytes)
        constexpr uint32_t ALIGN_BYTES = 512;
        constexpr uint32_t ELEM_SIZE = sizeof(T);
        constexpr uint32_t ALIGN_ELEMS = ALIGN_BYTES / ELEM_SIZE;

        // 5. 初始化 Buffer
        pipe->InitBuffer(zeroBuf, TILE_ELEMS * sizeof(T));
    }

    __aicore__ inline void Process() {
        uint32_t core_idx = GetBlockIdx();
        uint32_t core_num = GetBlockNum();

        LocalTensor<T> zero_local = zeroBuf.Get<T>();
        Duplicate(zero_local, (T)0, TILE_ELEMS);

        for (uint32_t offset = core_idx * TILE_ELEMS; offset < total_num; offset += core_num * TILE_ELEMS) {
            uint32_t cur_len = (total_num - offset < TILE_ELEMS) ? total_num - offset : TILE_ELEMS;
            DataCopy(addrGm[offset], zero_local, cur_len);
        }
        DataSyncBarrier<MemDsbT::ALL>();
    }

private:
    AscendC::TPipe* pipe;
    AscendC::GlobalTensor<T> addrGm;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroBuf;
    uint32_t total_num;
};

#endif // ZERO_OUT_IMPL_H