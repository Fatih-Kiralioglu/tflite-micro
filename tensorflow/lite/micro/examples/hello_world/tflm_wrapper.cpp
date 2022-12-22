/**
 *  Source code for the tflite_wrapper.hpp
 *
 */

#include "vcaudioml/tflite_wrapper.hpp"

/** std dependencies */
#include <algorithm>
#include <cstring>

#include "format.h"

namespace vcaudioml
{
/**
 * Main constructor
 */
TFLiteWrapper::TFLiteWrapper(TFLiteConfig config) : config_(config) {}

#if TFLITE_PROFILER
/**
 * @brief This function is enabling the TFLite profiler.
 * @details It starts measuring the execution time operator per operator.
 * This function should be called before executing TFLite model inference(s).
 * Move profiler to the constructor ?
 */
void TFLiteWrapper::LaunchProfiler(unsigned int batch_size)
{
    std::unique_ptr<tflite::profiling::BufferedProfiler> profiler(
        new tflite::profiling::BufferedProfiler(batch_size * tflite_interpreter_->nodes_size()));
    tflite_profiler_ = std::move(profiler);
    tflite_interpreter_->SetProfiler(tflite_profiler_.get());
    tflite_profiler_->StartProfiling();
}

/**
 * This function is printing out the execution time operator by operator.
 * It prints out the execution time in µs and the operator name.
 * This function should be called after executing TFLite model inference(s).
 *
 */
void TFLiteWrapper::EndProfiler(unsigned int batch_size)
{
    unsigned int multiplier = 0;
    tflite_profiler_->StopProfiling();
    auto profiler_events = tflite_profiler_->GetProfileEvents();
    const auto number_of_events = profiler_events.size();
    const auto n_ops = tflite_interpreter_->nodes_size();
    const auto subgraph = tflite_interpreter_->subgraph(0);  // support only single graph model
    for (unsigned int n_events = 0; n_events < number_of_events; n_events++)
    {
        auto ind_op = profiler_events[n_events]->event_metadata;
        const auto node_and_registration = subgraph->node_and_registration(ind_op);
        const auto registration = node_and_registration->second;

        if (n_events < n_ops)
        {
            TFLiteNodeProfiling profile_node = {
                0.0,
                static_cast<unsigned int>(ind_op),
                tflite::EnumNameBuiltinOperator(
                    static_cast<tflite::BuiltinOperator>(registration.builtin_code)),
            };
            profiling_buffer_.push_back(profile_node);
        }
        else
        {
            if (!(n_events % n_ops)) multiplier += 1;
            profiling_buffer_[n_events - (n_ops * multiplier)].inference_time +=
                (profiler_events[n_events]->end_timestamp_us -
                 profiler_events[n_events]->begin_timestamp_us);
        }
    }

    for (unsigned int ind_ops = 0; ind_ops < n_ops; ind_ops++)
    {
        profiling_buffer_[ind_ops].inference_time /= batch_size;
        fmt::print("Time (µs) {:.{}f}", profiling_buffer_[ind_ops].inference_time, 2);
        fmt::print("    Node_Index: {0}    Node_Name: {1}\n",
                   profiling_buffer_[ind_ops].ind,
                   profiling_buffer_[ind_ops].name);
    }
    fmt::print("\n");
}
#endif  // TFLITE_PROFILER

/**
 * @brief Function to build a TFLite interpreter using a TFLite model.
 * @details This function is called once at initialization time in your application.
 */
bool TFLiteWrapper::SetInterpreter()
{
    /// Use local model and interpreter
    /// Create the variable when you really start using it
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;

    /// Load a TFLite model and initialize
    /// Cmake can help using a environment variable
    /// If the model's size is 0, it means that there is no compiled object (to improve)
    /// Then, it loads the ML model as a flatbuffer binary file (*.tflite)
    if (!config_.model_size)
    {
        std::string filename = std::getenv("TFLITE_LOGI_PATH");  // To improve
        filename.append(PATH_SEPARATOR).append(config_.model_name);
        model = tflite::FlatBufferModel::BuildFromFile(filename.c_str());
    }
    /// Otherwise, it loads it as a compiled object
    else
    {
        model = tflite::FlatBufferModel::BuildFromBuffer(config_.model_buffer, config_.model_size);
    }
    /// Build the tflite interpreter
    tflite::ops::builtin::BuiltinOpResolver tflite_resolver;
    tflite::InterpreterBuilder(*model, tflite_resolver)(&(interpreter));

    if (!(interpreter))
    {
        /**
         *  @todo Add log error message
         */
        return false;
    }
    // Allocate inputs/outputs tensors here, as they will not be modified later on
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        /**
         *  @todo Add log error message
         */
        return false;
    }

    /// Support float_16 point if specified in the config
    interpreter->SetAllowFp16PrecisionForFp32(config_.fp16);
    /// Support multiple threads (useful when multiples subgraphs can be executed in parallel)
    if (config_.n_threads != -1) interpreter->SetNumThreads(config_.n_threads);

    if (config_.probe_mode)
    {
        /**
         *  @todo Add log message for model metadata
         */
    }
    /// Initialize model and interpreter at class level
    tflite_model_ = std::move(model);
    tflite_interpreter_ = std::move(interpreter);

    /// Initialise model input and output metadata
    if (!InitialiseAiBuffer()) return false;

    return true;
}

/**
 *  @brief It uses the tflite signature API to initialise inputs and outputs metadata.
 *  @details Extract the signature keys
 *  which contains the inputs and outputs tf.keras namings with the tflite node indexes.
 *  From the indexes, it then gets the size of the inputs and outputs (kBytes).
 *  The sizes are divided by 4, as it only supports float 32 bits format.
 *
 *  @return a boolean value to confirm if the initialisation went well (bool).
 */
bool TFLiteWrapper::InitialiseAiBuffer()
{
    /** Extracting ml model input and output metadata here */
    /// Only take the first signature key, do not consider any custom signature key
    auto signature_key = tflite_interpreter_->signature_keys()[0];
    auto signature_inputs = tflite_interpreter_->signature_inputs(signature_key->c_str());
    auto signature_outputs = tflite_interpreter_->signature_outputs(signature_key->c_str());
    config_.n_inputs = signature_inputs.size();
    config_.n_outputs = signature_outputs.size();
    /// Support ML model with input (or output) batch size to 1
    /// Loop through a std::map<node name, node index>
    /// where the "first" is the "node name" and the "second" is the "node index"
    for (auto const& input : signature_inputs)
    {
        /// Sanity check to ensure input use float_32
        const unsigned int data_type = tflite_interpreter_->tensor(input.second)->type;
        if (data_type != TfLiteType::kTfLiteFloat32)
        {
            fmt::print("Error : The input data {} does not have a float32 data type \n",
                       input.first);
            return false;
        }
        const unsigned int input_size = tflite_interpreter_->tensor(input.second)->bytes / 4;
        inputs_.push_back({input.first, input_size, input.second, nullptr});
    }
    for (auto const& output : signature_outputs)
    {
        /// Sanity check to ensure output use float_32
        const unsigned int data_type = tflite_interpreter_->tensor(output.second)->type;
        if (data_type != TfLiteType::kTfLiteFloat32)
        {
            fmt::print("Error : The output data {} does not have a float32 data type \n",
                       output.first);
            return false;
        }
        const unsigned int output_size = tflite_interpreter_->tensor(output.second)->bytes / 4;
        outputs_.push_back({output.first, output_size, output.second, nullptr});
    }
    return true;
}

/**
 *  @brief Function to set inputs, run tflite inference and set outputs.
 *  @details This function is called in the your main application loop (runtime).
 *  It supports and requires an input data with float_32 bits format.
 *  @param in data (const float*) (It expects an input float array).
 *
 *  leverage from the inheritance or type template?
 *
 */
void TFLiteWrapper::RunInterpreter(const float* input_data)
{
    /// Initialize the input(s)
    inputs_[0].data = tflite_interpreter_->typed_tensor<float>(inputs_[0].ind);
    std::memcpy(inputs_[0].data, input_data, sizeof(float) * inputs_[0].size);

    /// Run the inference
    tflite_interpreter_->Invoke();

    /// Extract and save the outputs
    for (unsigned int output_ind = 0; output_ind < config_.n_outputs; output_ind++)
    {
        outputs_[output_ind].data =
            tflite_interpreter_->typed_tensor<float>(outputs_[output_ind].ind);
    }
}

}  // end namespace vcaudioml
