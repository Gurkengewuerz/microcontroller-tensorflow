# Tensorflow Light Micro examples

This repository is an example how to use tensorflow with a microcontroller.
The example is using a `GY521` sensor to detect and recognise *standing*, *walking* and *running*.  
The code can recognise and also collect new data to a SD card. Everything is visible on a simple SSD1306 OLED display.

## ⚠ **Warnings** ⚠
### Model

The current model is currently not correctly trained. Because this is just an example i haven't collected enough data.

### On update

Please check on every update if the output by train.py equals the array `ACTIVITIES` in `main.cpp`. The indexes must match! Also check if the features are in the correct order.

## Repo structure

- `data` - this directory is for the data processing
    - `processed` - output directory for `train.py`
        - `model_data.cpp` - tensorflow lite model **auto generated** by `train.py` 
    - `raw_data` - this directory holds all raw data collected by the microcontroller
        - `*.csv` 
    - `requirements.txt` - python module requirements
    - `train.py` - python code to generate a tensorflow model
- `src` - microcontroller code
    - `NeuralNetwork.cpp` - Tensorflow Light Wrapper by [@atomic14](https://github.com/atomic14/tensorflow-lite-esp32/blob/master/firmware/src/NeuralNetwork.cpp)
    - `main.cpp` - main arduino framework code


## The Arduino_TensorFlowLite library

To get the newest tensor flow lite library for the Arduino framework just visit [this repository](https://github.com/antmicro/tensorflow-arduino-examples) and download the newest `tflite-micro` folder.