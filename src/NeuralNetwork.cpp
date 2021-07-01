#include "NeuralNetwork.h"

#include <LinkedList.h>

#include "model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

const int kArenaSize = 8 * 1024;

NeuralNetwork::NeuralNetwork(size_t expectedOutputs) {
    this->expectedOutputs = expectedOutputs;

    error_reporter = new tflite::MicroErrorReporter();

    model = tflite::GetModel(tflite_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter, "Model provided is schema version %d not equal to supported version %d.", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    // This pulls in the operators implementations we need
    resolver = new tflite::MicroMutableOpResolver<10>();
    resolver->AddFullyConnected();
    resolver->AddMul();
    resolver->AddAdd();
    resolver->AddLogistic();
    resolver->AddReshape();
    resolver->AddQuantize();
    resolver->AddDequantize();

    tensor_arena = (uint8_t *)malloc(kArenaSize);
    if (!tensor_arena) {
        TF_LITE_REPORT_ERROR(error_reporter, "Could not allocate arena");
        return;
    }

    // Build an interpreter to run the model with.
    interpreter = new tflite::MicroInterpreter(model, *resolver, tensor_arena, kArenaSize, error_reporter);

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }

    size_t used_bytes = interpreter->arena_used_bytes();
    TF_LITE_REPORT_ERROR(error_reporter, "Used bytes %d\n", used_bytes);

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
}

float *NeuralNetwork::getInputBuffer() {
    return input->data.f;
}

LinkedList<float*> NeuralNetwork::predict() {
    TfLiteStatus invokeStatus = interpreter->Invoke();
    LinkedList<float*> outputList = LinkedList<float*>();
    if (invokeStatus != kTfLiteOk) return outputList;
    for(size_t i = 0; i < this->expectedOutputs; i++) outputList.add(&output->data.f[i]);
    return outputList;
}