#!/bin/bash
set -e

BUILD_DEEPEP_MODULE="ON"
BUILD_DEEPEP_OPS="ON"
BUILD_KERNELS_MODULE="ON"
BUILD_MEMORY_SAVER_MODULE="ON"

ONLY_BUILD_DEEPEP_ADAPTER_MODULE="OFF"
ONLY_BUILD_DEEPEP_KERNELs_MODULE="OFF"
ONLY_BUILD_MEMORY_SAVER_MODULE="OFF"

DEBUG_MODE="OFF"
CLEAN_BUILD="OFF"  # 新增：默认不清理构建目录，开启增量编译

# 修改 getopts 增加 'c' 选项
while getopts ":a:hdc" opt; do
    case ${opt} in
        a )
            BUILD_DEEPEP_MODULE="OFF"
            BUILD_KERNELS_MODULE="OFF"
            BUILD_MEMORY_SAVER_MODULE="OFF"
            case "$OPTARG" in
                deepep )
                    BUILD_DEEPEP_MODULE="ON"
                    BUILD_DEEPEP_OPS="ON"
                    ;;
                deepep2 )
                    BUILD_DEEPEP_MODULE="ON"
                    BUILD_DEEPEP_OPS="OFF"
                    ;;
                kernels )
                    BUILD_KERNELS_MODULE="ON"
                    ;;
                deepep-adapter )
                    BUILD_DEEPEP_MODULE="ON"
                    ONLY_BUILD_DEEPEP_ADAPTER_MODULE="ON"
                    ;;
                deepep-kernels )
                    BUILD_DEEPEP_MODULE="ON"
                    ONLY_BUILD_DEEPEP_KERNELs_MODULE="ON"
                    ;;
                memory-saver )
                    BUILD_MEMORY_SAVER_MODULE="ON"
                    ONLY_BUILD_MEMORY_SAVER_MODULE="ON"
                    ;;
                * )
                    echo "Error: Invalid Value"
                    echo "Allowed value: deepep|kernels|deepep-adapter|deepep-kernels|memory-saver"
                    exit 1
                    ;;
            esac
            ;;
        d )
            DEBUG_MODE="ON"
            ;;
        c ) # 新增：清理构建目录的选项
            CLEAN_BUILD="ON"
            ;;
        h )
            echo "Use './build.sh' build all modules (Incremental by default)."
            echo "Use './build.sh -c' to FORCE CLEAN build (delete build directories)."
            echo "Use './build.sh -a <target>' to build specific parts of the project."
            echo "    <target> can be:"
            echo "    deepep            Only build deep_ep."
            echo "    kernels           Only build sgl_kernel_npu."
            echo "    deepep-adapter    Only build deepep adapter layer and use old build of deepep kernels."
            echo "    deepep-kernels    Only build deepep kernels and use old build of deepep adapter layer."
            echo "    memory-saver      Only build torch_memory_saver (under contrib)."
            exit 1
            ;;
        \? )
            echo "Error: unknown flag: -$OPTARG" 1>&2
            echo "Run './build.sh -h' for more information."
            exit 1
            ;;
        : )
            echo "Error: -$OPTARG requires a value" 1>&2
            echo "Run './build.sh -h' for more information."
            exit 1
            ;;
    esac
done

shift $((OPTIND -1))


export DEBUG_MODE=$DEBUG_MODE

SOC_VERSION="${1:-Ascend910B3}"

if [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
fi

if [ -n "$ASCEND_INCLUDE_DIR" ]; then
    ASCEND_INCLUDE_DIR=$ASCEND_INCLUDE_DIR
else
    ASCEND_INCLUDE_DIR=${_ASCEND_INSTALL_PATH}/aarch64-linux/include
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
echo "ascend path: ${ASCEND_HOME_PATH}"
source $(dirname ${ASCEND_HOME_PATH})/set_env.sh

CURRENT_DIR=$(pwd)
PROJECT_ROOT=$(dirname "$CURRENT_DIR")
VERSION="1.0.0"
OUTPUT_DIR=$CURRENT_DIR/output
mkdir -p $OUTPUT_DIR
echo "outpath: ${OUTPUT_DIR}"

COMPILE_OPTIONS=""

function build_kernels()
{
    if [[ "$ONLY_BUILD_DEEPEP_KERNELs_MODULE" == "ON" ]]; then return 0; fi
    if [[ "$ONLY_BUILD_MEMORY_SAVER_MODULE" == "ON" ]]; then return 0; fi

    # 1. 动态获取最大核心数，解决“慢”的问题
    MAX_JOBS=$(nproc 2>/dev/null || echo 16)
    echo "[INFO] Using ${MAX_JOBS} parallel jobs for Make."

    CMAKE_DIR=""
    BUILD_DIR="build"

    cd "$CMAKE_DIR" || exit

    if [[ "$CLEAN_BUILD" == "ON" ]]; then
        echo "[INFO] Cleaning build directory for kernels..."
        rm -rf $BUILD_DIR
    fi
    mkdir -p $BUILD_DIR

    cmake $COMPILE_OPTIONS \
        -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
        -DASCEND_HOME_PATH=$ASCEND_HOME_PATH \
        -DSOC_VERSION=$SOC_VERSION \
        -DBUILD_DEEPEP_MODULE=$BUILD_DEEPEP_MODULE \
        -DBUILD_KERNELS_MODULE=$BUILD_KERNELS_MODULE \
        -B "$BUILD_DIR" \
        -S .
    
    # 3. 使用所有核心进行编译
    cmake --build "$BUILD_DIR" -j"${MAX_JOBS}" --target install
    cd -
}

function build_deepep_kernels()
{
    if [[ "$ONLY_BUILD_DEEPEP_ADAPTER_MODULE" == "ON" ]]; then return 0; fi
    if [[ "$BUILD_DEEPEP_MODULE" != "ON" ]]; then return 0; fi

    if [[ "$BUILD_DEEPEP_OPS" == "ON" ]]; then
        KERNEL_DIR="csrc/deepep/ops"
    else
        KERNEL_DIR="csrc/deepep/ops2"
    fi
    CUSTOM_OPP_DIR="${CURRENT_DIR}/python/deep_ep/deep_ep"

    cd "$KERNEL_DIR" || exit

    # 注意：deepep 的内部 build.sh 可能自己含有清理逻辑。
    # 如果你也想让 deepep 增量编译，可能需要检查并修改 $KERNEL_DIR/build.sh
    chmod +x build.sh
    chmod +x cmake/util/gen_ops_filter.sh
    ./build.sh

    custom_opp_file=$(find ./build_out -maxdepth 1 -type f -name "custom_opp*.run")
    if [ -z "$custom_opp_file" ]; then
        echo "can not find run package"
        exit 1
    else
        echo "find run package: $custom_opp_file"
        chmod +x "$custom_opp_file"
    fi
    rm -rf "$CUSTOM_OPP_DIR"/vendors
    ./build_out/custom_opp_*.run --install-path=$CUSTOM_OPP_DIR
    cd -
}

function build_memory_saver()
{
    if [[ "$BUILD_MEMORY_SAVER_MODULE" != "ON" ]]; then return 0; fi
    echo "[memory_saver] Building torch_memory_saver via setup.py"
    cd contrib/torch_memory_saver/python || exit
    # Python setup.py clean 比较快，通常建议保留以防止 Wheel 包污染
    # 如果想极致速度，可以注释掉下面两行，但一般不推荐
    rm -rf "$CURRENT_DIR"/contrib/torch_memory_saver/python/build
    rm -rf "$CURRENT_DIR"/contrib/torch_memory_saver/python/dist
    python3 setup.py clean --all
    python3 setup.py bdist_wheel
    mv -v "$CURRENT_DIR"/contrib/torch_memory_saver/python/dist/torch_memory_saver*.whl "${OUTPUT_DIR}/"
    rm -rf "$CURRENT_DIR"/contrib/torch_memory_saver/python/dist
    cd -
}

function make_deepep_package()
{
    cd python/deep_ep || exit

    cp -v ${OUTPUT_DIR}/lib/* "$CURRENT_DIR"/python/deep_ep/deep_ep/
    rm -rf "$CURRENT_DIR"/python/deep_ep/build
    python3 setup.py clean --all
    python3 setup.py bdist_wheel
    mv -v "$CURRENT_DIR"/python/deep_ep/dist/deep_ep*.whl ${OUTPUT_DIR}/
    rm -rf "$CURRENT_DIR"/python/deep_ep/dist
    cd -
}

function make_sgl_kernel_npu_package()
{
    cd python/sgl_kernel_npu || exit

    rm -rf "$CURRENT_DIR"/python/sgl_kernel_npu/dist
    python3 setup.py clean --all
    python3 setup.py bdist_wheel
    mv -v "$CURRENT_DIR"/python/sgl_kernel_npu/dist/sgl_kernel_npu*.whl ${OUTPUT_DIR}/
    rm -rf "$CURRENT_DIR"/python/sgl_kernel_npu/dist
    cd -
}

function main()
{

    build_kernels
    build_deepep_kernels
    if pip3 show wheel;then
        echo "wheel has been installed"
    else
        pip3 install wheel==0.45.1
    fi
    build_memory_saver
    if [[ "$BUILD_DEEPEP_MODULE" == "ON" ]]; then
        make_deepep_package
    fi
    if [[ "$BUILD_KERNELS_MODULE" == "ON" ]]; then
        make_sgl_kernel_npu_package
    fi

}

main