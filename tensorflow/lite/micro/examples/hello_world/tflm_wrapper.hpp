/**
 *  @file tflite_wrapper.hpp
 *  @brief Global TFLite Wrapper file to integrate any TFLite model(s) in any application(s).
 *  @author Moussa
 *
 *  Goal is to encapsulate TFLite C++ API and provide a simplified Wrapper API. \n
 *  This new API contains 2 key functions: SetInterpreter() and RunInterpreter(). \n
 *  It also contains 2 keys structures to embed TFLite metadata information. \n
 *  This wrapper requires TFLite (2.7.0) dependencies (headers + libs). \n
 *  In addition, this wrapper has one dependency: the tflite model \n
 *
 *  @todo Support ML models with multiple inputs.
 *  @todo Use C++ Templates to support different data types.
 *
 *  @example HOW-TO vcaudioml::TFLiteWrapper
 *  ### Pre-requisites
 *  Set your environment variable `TFLITE_LOGI_PATH` to the tflite models directory.
 *  @code
 *      export TFLITE_LOGI_PATH=<your_absolute_path_to tflite_models_directory>/
 *  @endcode
 *  Prepare your model metadata structure, following one of the 3 strategies: \n
 *      - with C++ const/constexpr \n
 *      - with C defines/macros \n
 *      - manually \n
 *  User should not allocate extra memory during the initialization of the structures. \n
 *  User should leverage from the compiler. \n
 *  @code
 *      vcaudioml::TFLiteConfig nn_config = AI_NN_CONFIG;
 *  @endcode
 *  ### Initialization block
 *  Create the class instance:
 *  @code
 *      vcaudioml::TFLiteWrapper tflite_graph(
 *          vcaudioml::TFLiteConfig nn_config
 *      );
 *  @endcode
 *  Initialize the interpreter:
 *  @code
 *      tflite_graph.SetInterpreter();
 *  @endcode
 *  ### Main runtime loop
 *  run the tflite inference  (in_data is your input float-32 data array):
 *  @code
 *      tflite_graph.RunInterpreter(in_data);
 *      auto nn_outputs = tflite_graph.GetOutputs();
 *  @endcode
 *  ### Additional configs
 *  The follow configurations can be enabled before SetInterpreter() call:
 *  @code
 *      tflite_graph.EnableFp16();
 *      tflite_graph.EnableProbeMode();
 *  @endcode
 *  ### For debugging purpose
 *  Get the config file to get some metadata information:
 *  @code
 *      vcaudioml::TFLiteConfig tflite_model_config = tflite_graph.GetConfig();
 *  @endcode
 *  Get the inputs/outputs buffers to get some metadata information and track buffer status.
 *  @code
 *      auto tflite_outputs = tflite_graph.GetOutputs();
 *      auto tflite_inputs = tflite_graph.GetInputs();
 *  @endcode
 *
 *
 */

#ifndef TFLM_WRAPPER_HPP
#define TFLM_WRAPPER_HPP

/**  TFLite dependencies  */
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/examples/hello_world/ah_187_int8_model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <iostream>
/** TFLite Profiler */
#if TFLITE_PROFILER
#include "tensorflow/lite/profiling/profiler.h"
#endif  // TFLITE_PROFILER

/** Data structures */

namespace vcaudioml
{
/** @class vcaudioml::TFLiteWrapper
 *  @brief This single class is wrapping the TFLite C++ API
 *
 *  It provides a simplified wrapper API with 2 high-level functions:
 *  @code
 *  vcaudioml::TFLiteWrapper::SetInterpreter()
 *  vcaudioml::TFLiteWrapper::RunInterpreter()
 *  @endcode
 *
 */

struct TFLMConfig
{
        unsigned char * model_data;
        unsigned int model_size;
        unsigned int tensor_arena_size_;
        int n_threads;
};

class TFLMWrapper
{
    private:
        TFLMConfig config_;

        std::unique_ptr<tflite::MicroInterpreter> tflm_interpreter_;

        std::vector<uint8_t> tensor_arena_;
        tflite::AllOpsResolver resolver;
        tflite::MicroErrorReporter micro_error_reporter;
#if TFLITE_PROFILER
        std::unique_ptr<tflite::profiling::BufferedProfiler> tflite_profiler_;
        std::vector<TFLiteNodeProfiling> profiling_buffer_;
#endif  // TFLITE_PROFILER
        //TFLiteConfig config_;

        std::vector<TfLiteTensor *> outputs_;
        /// Initialise the ml inputs, outputs with the tflite signature API 2.7

    public:
        TFLMWrapper(TFLMConfig config) : config_(config) {}
        /// Main constructor
        //TFLMWrapper(TFLiteConfig tflite_config);
        /// Function to build a TFLite interpreter using a TFLite model.
        /// This function is called once at initialisation time in your application.
        bool SetInterpreter()
        {
                /// Use local model and interpreter
                /// Create the variable when you really start using it
                std::cout << "start SetInterpreter" << std::endl;

                /// Load a TFLite model and initialize
                /// Cmake can help using a environment variable
                /// If the model's size is 0, it means that there is no compiled object (to improve)
                /// Then, it loads the ML model as a flatbuffer binary file (*.tflite)
           
                std::cout << "tensor_arena_ resizer" << std::endl;

                tensor_arena_.resize(config_.tensor_arena_size_);

                std::cout << "loading model" << std::endl;
                const tflite::Model * model = tflite::GetModel(config_.model_data);
                if (model->version() != TFLITE_SCHEMA_VERSION) 
                {
                        TF_LITE_REPORT_ERROR(&micro_error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
                }
                //tflm_model_ = std::move(model);
                
                std::cout << "setting interpreter" << std::endl;
                tflm_interpreter_.reset(new tflite::MicroInterpreter(model, resolver, tensor_arena_.data(), tensor_arena_.size(), &micro_error_reporter));
                std::cout<<" set interpretersss" << std::endl;
                // Allocate memory from the tensor_arena for the model's tensors

                
                // Allocate inputs/outputs tensors here, as they will not be modified later on
                std::cout<<" AllocateTensors" << std::endl;
                if (tflm_interpreter_->AllocateTensors() != kTfLiteOk)
                {
                        /**
                        *  @todo Add log error message
                        */
                        return false;
                }

                /// Support multiple threads (useful when multiples subgraphs can be executed in parallel)
                if (config_.n_threads != -1)
                {
                        //tflm_interpreter_->SetNumThreads(config_.n_threads);
                } 


                return true;
        }
        /// Function to set inputs, run tflite inference and get outputs.
        /// This function is called in the main loop (runtime).
        /// It supports and requires an input data with float_32 bits format.
        void RunInterpreter(const float* input_data)
        {
                TfLiteTensor* input = tflm_interpreter_->input(0);
                int total_float_count = input->dims->data[2] * input->dims->data[3];
                std::copy(input_data, input_data + total_float_count, &input->data.f[0]);
                // Run the model and check that it succeeds
                tflm_interpreter_->Invoke();
                for(unsigned int i=0; i< tflm_interpreter_->outputs_size(); i++)
                {
                      outputs_.push_back(tflm_interpreter_->output(i));  
                }
                //tflm_interpreter_->ResetVariableTensors();
        }

#if TFLITE_PROFILER
        /// It starts measuring the execution time operator per operator.
        /// This function should be called before executing TFLite model inference(s).
        void LaunchProfiler(unsigned int batch_size);
        /// It prints out the execution time in µs and the operator name.
        /// This function should be called after executing TFLite model inference(s).
        void EndProfiler(unsigned int batch_size);
#endif  // TFLITE_PROFILER

        /// Enable 16-bits floating point precision
        void EnableFp16() {  }
        /// Enable probe mode
        void EnableProbeMode() {  }
        /// Get TFLite model inputs/outputs

        /// Get TFLite model outputs (shared by const& to avoid extra copying)
        const std::vector<TfLiteTensor *> & GetOutputs() const { return outputs_; }
        /// Get TFLite model config
        TFLMConfig GetConfig() const { return config_; }
        /// Reset the ML model (tensors' states)
        void ResetModel() { tflm_interpreter_->ResetVariableTensors(); }
};
}  // end namespace vcaudioml

#endif  // TFLITE_WRAPPER_HPP
