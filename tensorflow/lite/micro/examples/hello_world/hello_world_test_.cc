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

#include "tflm_wrapper.hpp"

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



int main() {

  vcaudioml::TFLMConfig config;
  config.model_data = (unsigned char * )g_ah_187_int8_model_data;
  config.model_size = g_ah_187_int8_model_data_size;
  config.tensor_arena_size_ = 400000;
  config.n_threads = -1;

  vcaudioml::TFLMWrapper wrapper(config);
  std::cout << "set interpreter" << std::endl;
  wrapper.SetInterpreter();

  
  
  std::cout << "running interpreter" << std::endl;
  unsigned int B_SIZE = 1;
  constexpr int INPUT_ARRAY_SIZE = 6 * 48;
  std::vector<std::array<float, INPUT_ARRAY_SIZE> > in_data(B_SIZE);
  RandomInitBufferArray(in_data, -5, +5);
  
  std::cout << "runing interpreter" << std::endl;
  wrapper.RunInterpreter(in_data[0].data());

  std::vector<TfLiteTensor *> outputs = wrapper.GetOutputs();
  
  TfLiteTensor* output1 = outputs[1];

  for(int i=0; i < 48; i++)
  {
      std::cout << output1->data.f[i] << std::endl;
  }

  std::cout << "runing interpreter" << std::endl;
  wrapper.RunInterpreter(in_data[0].data());

  std::vector<TfLiteTensor *> outputs1 = wrapper.GetOutputs();
  
  TfLiteTensor* output11 = outputs1[1];

  for(int i=0; i < 48; i++)
  {
      std::cout << output11->data.f[i] << std::endl;
  }
   
  

   

  return 0;

}

