 #!/bin/bash


FILE_PATH="$1"
FILE_NAME=$(basename ${FILE_PATH})
TEST_NAME=${FILE_NAME%.*}

echo "Run test ${TEST_NAME}"

DUMP_DIR="${TEST_NAME}_dump"
rm -fr $DUMP_DIR
mkdir $DUMP_DIR

# export TRITON_ENABLE_LLVM_DEBUG=1 
# export TRITON_KERNEL_DUMP=1
# export TRITON_DUMP_DIR="$DUMP_DIR/triton"
# export TRITON_ALWAYS_COMPILE=1

# export MLIR_ENABLE_DUMP=1 
# export MLIR_DUMP_PATH="$DUMP_DIR/mlir_dump" 

LOG="$DUMP_DIR/run.log" 
touch $LOG
python3 $1 2>&1 | tee -a $LOG
