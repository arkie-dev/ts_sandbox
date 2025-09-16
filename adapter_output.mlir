#map = affine_map<(d0) -> (d0)>
module {
  func.func @origin_index_select(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c1971940 = arith.constant 1971940 : index
    %c49299_i32 = arith.constant 49299 : i32
    %c16 = arith.constant {Undefined} 16 : index
    %c1024 = arith.constant {Undefined} 1024 : index
    %c1 = arith.constant {Undefined} 1 : index
    %c0 = arith.constant {Undefined} 0 : index
    %c1024_i32 = arith.constant {Undefined} 1024 : i32
    %c0_i32 = arith.constant {Undefined} 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %0 = tensor.empty() : tensor<1024x1xi32>
    %1 = linalg.fill ins(%c16_i32 : i32) outs(%0 : tensor<1024x1xi32>) -> tensor<1024x1xi32>
    %2 = tensor.empty() : tensor<16xi32>
    %3 = arith.muli %arg9, %c49299_i32 : i32
    %4 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : tensor<16xi32>) {
    ^bb0(%out: i32):
      %6 = linalg.index 0 : index
      %7 = arith.index_cast %6 : index to i32
      linalg.yield %7 : i32
    } -> tensor<16xi32>
    %5 = tensor.empty() : tensor<1024x16xi32>
    %broadcasted = linalg.broadcast ins(%4 : tensor<16xi32>) outs(%5 : tensor<1024x16xi32>) dimensions = [0] 
    scf.for %arg12 = %c0_i32 to %c49299_i32 step %c1024_i32  : i32 {
      %6 = arith.index_cast %3 : i32 to index
      %7 = arith.index_cast %arg12 : i32 to index
      %8 = arith.addi %6, %7 : index
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%8], sizes: [1024], strides: [1] : memref<?xi32> to memref<1024xi32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<1024xi32>
      %9 = arith.addi %6, %c1024 : index
      %10 = arith.addi %9, %7 : index
      %11 = arith.maxsi %8, %c1971940 : index
      %12 = arith.minsi %10, %11 : index
      %13 = arith.subi %12, %8 : index
      %14 = arith.cmpi slt, %13, %c1024 : index
      scf.if %14 {
        linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<1024xi32>)
      }
      %subview = memref.subview %reinterpret_cast[0] [%13] [1] : memref<1024xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
      %subview_0 = memref.subview %alloc[0] [%13] [1] : memref<1024xi32> to memref<?xi32, strided<[1]>>
      memref.copy %subview, %subview_0 : memref<?xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1]>>
      %15 = bufferization.to_tensor %alloc restrict writable : memref<1024xi32>
      %expanded = tensor.expand_shape %15 [[0, 1]] output_shape [1024, 1] : tensor<1024xi32> into tensor<1024x1xi32>
      %16 = arith.muli %expanded, %1 {MixUse} : tensor<1024x1xi32>
      %17 = arith.extsi %16 {MixUse} : tensor<1024x1xi32> to tensor<1024x1xi64>
      %18 = tensor.empty() : tensor<1024x16xi64>
      %collapsed = tensor.collapse_shape %17 [[0, 1]] : tensor<1024x1xi64> into tensor<1024xi64>
      %broadcasted_1 = linalg.broadcast ins(%collapsed : tensor<1024xi64>) outs(%18 : tensor<1024x16xi64>) dimensions = [1] 
      %19 = arith.extsi %broadcasted {MixUse} : tensor<1024x16xi32> to tensor<1024x16xi64>
      %20 = arith.addi %broadcasted_1, %19 {MixUse} : tensor<1024x16xi64>
      %21 = tensor.empty() {DataUse} : tensor<1024x16xf32>
      %22 = scf.for %arg13 = %c0 to %c1024 step %c1 iter_args(%arg14 = %21) -> (tensor<1024x16xf32>) {
        %25 = scf.for %arg15 = %c0 to %c16 step %c1 iter_args(%arg16 = %arg14) -> (tensor<1024x16xf32>) {
          %extracted = tensor.extract %20[%arg13, %arg15] : tensor<1024x16xi64>
          %26 = arith.index_cast %extracted : i64 to index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
          %27 = memref.load %reinterpret_cast_4[%c0] : memref<1xf32, strided<[1], offset: ?>>
          %28 = tensor.empty() : tensor<1x1xf32>
          %29 = linalg.fill ins(%27 : f32) outs(%28 : tensor<1x1xf32>) -> tensor<1x1xf32>
          %inserted_slice = tensor.insert_slice %29 into %arg16[%arg13, %arg15] [1, 1] [16, 1] {DataUse} : tensor<1x1xf32> into tensor<1024x16xf32>
          scf.yield {Undefined} %inserted_slice : tensor<1024x16xf32>
        } {DataUse}
        scf.yield {Undefined} %25 : tensor<1024x16xf32>
      } {DataUse, ExtractedLoadOrStore}
      %23 = arith.muli %8, %c16 : index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%23], sizes: [1024, 16], strides: [16, 1] : memref<?xf32> to memref<1024x16xf32, strided<[16, 1], offset: ?>>
      %24 = arith.minsi %13, %c1024 : index
      %extracted_slice = tensor.extract_slice %22[0, 0] [%24, 16] [1, 1] : tensor<1024x16xf32> to tensor<?x16xf32>
      %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%24, 16] [1, 1] : memref<1024x16xf32, strided<[16, 1], offset: ?>> to memref<?x16xf32, strided<[16, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_3 : (tensor<?x16xf32>, memref<?x16xf32, strided<[16, 1], offset: ?>>) -> ()
    } {Undefined}
    return
  }
}

