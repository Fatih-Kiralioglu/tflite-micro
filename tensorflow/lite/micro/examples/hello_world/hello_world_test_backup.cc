/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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



#include <iostream>
#include <math.h>
#include <chrono>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/examples/hello_world/ah_187_int8_model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <fstream>
#include <unistd.h>
void process_mem_usage(double& vm_usage, double& resident_set)
{
   using std::ios_base;
   using std::ifstream;
   using std::string;

   vm_usage     = 0.0;
   resident_set = 0.0;

   // 'file' stat seems to give the most reliable results
   //
   ifstream stat_stream("/proc/self/stat",ios_base::in);

   // dummy vars for leading entries in stat that we don't care about
   //
   string pid, comm, state, ppid, pgrp, session, tty_nr;
   string tpgid, flags, minflt, cminflt, majflt, cmajflt;
   string utime, stime, cutime, cstime, priority, nice;
   string O, itrealvalue, starttime;

   // the two fields we want
   //
   unsigned long vsize;
   long rss;

   stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
               >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
               >> utime >> stime >> cutime >> cstime >> priority >> nice
               >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

   stat_stream.close();

   long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
   vm_usage     = (double)vsize / 1024.0;
   resident_set = rss * page_size_kb;
}



template <typename container>
void RandomInitArray(container& array, const unsigned int LEN, const int R_MIN, const int R_MAX)
{
    for (unsigned int array_ind = 0; array_ind < LEN; array_ind++)
    {
        array[array_ind] =
            static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (R_MAX - R_MIN));
        array[array_ind] += R_MIN;
    }
}

/**
 *  @brief Initialize a vector with random arrays
 *  @details Template function to initialize a "vector of containers".
 *  It initializes all containers with random value between a certain range.
 *  Main container is std::array, but it may support any container.
 *  Support any sizes.
 *  @param random_buffer_array (a vector of containers)
 *  @param R_MIN (int) (It corresponds to the minimum of the random values).
 *  @param R_MAX (int) (It corresponds to the maximum of the random values).
 */
template <typename container>
void RandomInitBufferArray(std::vector<container>& random_buffer_array,
                           const int R_MIN,
                           const int R_MAX)
{
    for (auto& vector_element : random_buffer_array)
        RandomInitArray(vector_element, vector_element.size(), R_MIN, R_MAX);
}


TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
  // Define the input and the expected output
  //float x = 0.0f;
  //float y_true = sin(x);

  // Set up logging
  tflite::MicroErrorReporter micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(g_ah_187_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }

  // This pulls in all the operation implementations we need
  tflite::AllOpsResolver resolver;
  

  constexpr int kTensorArenaSize = 400000;
  uint8_t tensor_arena[kTensorArenaSize];

  // Build an interpreter to run the model with
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize, &micro_error_reporter);
  // Allocate memory from the tensor_arena for the model's tensors
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  unsigned int B_SIZE = 1;
  constexpr int INPUT_ARRAY_SIZE = 6 * 48;
  std::vector<std::array<float, INPUT_ARRAY_SIZE> > in_data(B_SIZE);
  RandomInitBufferArray(in_data, -2, +2);
  
  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect
  TF_LITE_MICRO_EXPECT(input != nullptr);
  // The property "dims" tells us the tensor's shape. It has one element for
  // each dimension. Our input is a 2D tensor containing 1 element, so "dims"
  // should have size 2.
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  // The value of each element gives the length of the corresponding tensor.
  // We should expect two single element tensors (one is contained within the
  // other).
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(6, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(48, input->dims->data[3]);
  // The input is an 8 bit integer value
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);


  std::chrono::time_point<std::chrono::system_clock> now = 
    std::chrono::system_clock::now();

  unsigned int test_count = 1000;
  for(unsigned int k=0; k< test_count; k++)
  {
    RandomInitBufferArray(in_data, -2, +2);
    /* for(int i=0; i<INPUT_ARRAY_SIZE; i++)
    {
      input->data.f[i] = in_data[0][i];
    }*/
    std::copy(in_data[0].begin(),in_data[0].end(), &input->data.f[0]);

    // Run the model and check that it succeeds
    TfLiteStatus invoke_status = interpreter.Invoke();
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);
    TfLiteTensor* output1_old = interpreter.output(1);
    std::cout << "\n old: ";
    for(int i=0; i < 48; i++)
    {
        std::cout << output1_old->data.f[i] << " " << std::endl;
    }
    //interpreter.ResetVariableTensors();
    input = interpreter.input(0);
    std::copy(in_data[0].begin(),in_data[0].end(), &input->data.f[0]);

    invoke_status = interpreter.Invoke();
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);
    TfLiteTensor* output1_new = interpreter.output(1);

    std::cout << "\n new: ";
    for(int i=0; i < 48; i++)
    {
        std::cout << output1_new->data.f[i] << " " << std::endl;
    }
  }
  
  std::chrono::time_point<std::chrono::system_clock> now2 = std::chrono::system_clock::now();
   auto millis = std::chrono::duration_cast<std::chrono::microseconds>(now2 - now).count();

   std::cout << "duration: " << millis << std::endl;
   
  std::cout << "test completed: " << test_count << std::endl;

  // Obtain a pointer to the output tensor and make sure it has the
  // properties we expect. It should be the same as the input tensor.
  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(3, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);

  TfLiteTensor* output1 = interpreter.output(1);
  TF_LITE_MICRO_EXPECT_EQ(3, output1->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output1->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, output1->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(48, output1->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output1->type);

   double vm, rss;
   process_mem_usage(vm, rss);
   std::cout << "VM: " << vm << "; RSS: " << rss << std::endl;

  /*
  // Get the output quantization parameters
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  // Obtain the quantized output from model's output tensor
  int8_t y_pred_quantized = output->data.int8[0];
  // Dequantize the output from integer to floating-point
  float y_pred = (y_pred_quantized - output_zero_point) * output_scale;

  // Check if the output is within a small range of the expected output
  float epsilon = 0.05f;
  TF_LITE_MICRO_EXPECT_NEAR(y_true, y_pred, epsilon);

  // Run inference on several more values and confirm the expected outputs
  x = 1.f;
  y_true = sin(x);
  input->data.int8[0] = x / input_scale + input_zero_point;
  interpreter.Invoke();
  y_pred = (output->data.int8[0] - output_zero_point) * output_scale;
  TF_LITE_MICRO_EXPECT_NEAR(y_true, y_pred, epsilon);

  x = 3.f;
  y_true = sin(x);
  input->data.int8[0] = x / input_scale + input_zero_point;
  interpreter.Invoke();
  y_pred = (output->data.int8[0] - output_zero_point) * output_scale;
  TF_LITE_MICRO_EXPECT_NEAR(y_true, y_pred, epsilon);

  x = 5.f;
  y_true = sin(x);
  input->data.int8[0] = x / input_scale + input_zero_point;
  interpreter.Invoke();
  y_pred = (output->data.int8[0] - output_zero_point) * output_scale;
  TF_LITE_MICRO_EXPECT_NEAR(y_true, y_pred, epsilon);
  */
}

TF_LITE_MICRO_TESTS_END
