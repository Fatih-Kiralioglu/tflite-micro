/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/all_ops_resolver.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include <iostream>
namespace tflite {

AllOpsResolver::AllOpsResolver() {
  // Please keep this list of Builtin Operators in alphabetical order.
  std::cout << "all ops loading 0..." << std::endl;
  AddAbs();
  AddUnidirectionalSequenceLSTM();
  AddAdd();
  AddAddN();
  AddArgMax();
  AddArgMin();
  AddAssignVariable();
  AddAveragePool2D();
  AddBatchToSpaceNd();
  std::cout << "all ops loading 1..." << std::endl;
  AddBroadcastArgs();
  AddBroadcastTo();
  AddCallOnce();
  AddCast();
  AddCeil();
  AddCircularBuffer();
  AddConcatenation();
  AddConv2D();
  AddCos();
  std::cout << "all ops loading 2..." << std::endl;
  AddCumSum();
  AddDepthToSpace();
  AddDepthwiseConv2D();
  AddDequantize();
  AddDetectionPostprocess();
  AddDiv();
  AddElu();
  AddEqual();
  AddEthosU();
  AddExp();
  AddExpandDims();
  AddFill();
  AddFloor();
  std::cout << "all ops loading 3..." << std::endl;
  AddFloorDiv();
  AddFloorMod();
  AddFullyConnected();
  AddGather();
  AddGatherNd();
  AddGreater();
  AddGreaterEqual();
  AddHardSwish();
  AddIf();
  AddL2Normalization();
  AddL2Pool2D();
  std::cout << "all ops loading 4..." << std::endl;
  AddLeakyRelu();
  AddLess();
  AddLessEqual();
  AddLog();
  AddLogicalAnd();
  AddLogicalNot();
  AddLogicalOr();
  AddLogistic();
  AddMaxPool2D();
  AddMaximum();
  AddMean();
  std::cout << "all ops loading 5..." << std::endl;
  AddMinimum();
  AddMirrorPad();
  AddMul();
  AddNeg();
  AddNotEqual();
  AddPack();
  AddPad();
  AddPadV2();
  AddPrelu();
  AddQuantize();
  std::cout << "all ops loading 6..." << std::endl;
  AddReadVariable();
  AddReduceMax();
  AddRelu();
  AddRelu6();
  AddReshape();
  AddResizeBilinear();
  AddResizeNearestNeighbor();
  AddRound();
  AddRsqrt();
  AddShape();
  AddSin();
  AddSlice();
  std::cout << "all ops loading 7..." << std::endl;
  AddSoftmax();
  AddSpaceToBatchNd();
  AddSpaceToDepth();
  AddSplit();
  AddSplitV();
  AddSqrt();
  AddSquare();
  AddSqueeze();
  AddStridedSlice();
  AddSub();
  AddSum();
  AddSvdf();
  std::cout << "all ops loading 8..." << std::endl;
  AddTanh();
  AddTranspose();
  AddTransposeConv();
  AddUnpack();
  AddVarHandle();
  AddWhile();
  AddZerosLike();

  std::cout << "all ops loaded." << std::endl;

}

}  // namespace tflite
