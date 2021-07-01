import glob
import os
import csv
import statistics
import math
import pandas
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.python.keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

def split_dataset(dataset: tf.data.Dataset, validation_data_fraction: float):
  """
  Splits a dataset of type tf.data.Dataset into a training and validation dataset using given ratio. Fractions are
  rounded up to two decimal places.
  @param dataset: the input dataset to split.
  @param validation_data_fraction: the fraction of the validation data as a float between 0 and 1.
  @return: a tuple of two tf.data.Datasets as (training, validation)
  """

  validation_data_percent = round(validation_data_fraction * 100)
  if not (0 <= validation_data_percent <= 100):
    raise ValueError("validation data fraction must be ∈ [0,1]")

  dataset = dataset.enumerate()
  train_dataset = dataset.filter(lambda f, data: f % 100 > validation_data_percent)
  validation_dataset = dataset.filter(lambda f, data: f % 100 <= validation_data_percent)

  # remove enumeration
  train_dataset = train_dataset.map(lambda f, data: data)
  validation_dataset = validation_dataset.map(lambda f, data: data)

  return train_dataset, validation_dataset

def process(source, destination, output):
  dir_path = os.path.dirname(os.path.realpath(__file__))
  RAW_DATA = os.path.join(source)
  PROCESSED_DATA = os.path.join(destination)

  os.chdir(dir_path)

  PROCESSED_FILE = os.path.join(PROCESSED_DATA, output)

  if os.path.isfile(PROCESSED_FILE):
    os.remove(PROCESSED_FILE)

  data = {"accelerometer": [], "accelerometer_max": [], "target": []}

  for file in glob.glob(source + "/**/*.csv", recursive=True):
    if RAW_DATA not in file:
      continue
    if "disabled" in file:
      continue

    print(file)
    with open(file) as csv_file:
      listAcc = {"x": [], "y": [], "z": []}
      listGyr = {"x": [], "y": [], "z": []}
      listGrv = {"x": [], "y": [], "z": []}

      csv_reader = csv.reader(csv_file, delimiter=";")
      line_count = 0
      counter = 0
      csv_file.seek(0)
      line_length = len(csv_file.readlines())
      csv_file.seek(0)
      for row in csv_reader:
        line_count += 1
        counter += 1
        if line_count == 1:
          continue
        
        millisTime = int(row[0])
        if counter <= 10:
          line_count -= 1
          continue
        if counter >= line_length - 5:
          line_count -= 1
          continue
        accelerometer = row[1:4]
        gyroscope = row[4:7]
        gravity = row[7:10]
        walkType = str(row[10]).lower()

        listAcc["x"].append(float(accelerometer[0]))
        listAcc["y"].append(float(accelerometer[1]))
        listAcc["z"].append(float(accelerometer[2]))

        listGyr["x"].append(float(gyroscope[0]))
        listGyr["y"].append(float(gyroscope[1]))
        listGyr["z"].append(float(gyroscope[2]))

        listGrv["x"].append(float(gravity[0]))
        listGrv["y"].append(float(gravity[1]))
        listGrv["z"].append(float(gravity[2]))

        if len(listAcc["x"]) < 10:
          continue

        processed_accelerometer = math.sqrt(statistics.mean(listAcc["x"])**2 + statistics.mean(listAcc["y"])**2 + statistics.mean(listAcc["z"])**2)
        processed_accelerometer_max = math.sqrt(max(listAcc["x"])**2 + max(listAcc["y"])**2 + max(listAcc["z"])**2)
        processed_gyroscope = math.sqrt(statistics.mean(listGyr["x"])**2 + statistics.mean(listGyr["y"])**2 + statistics.mean(listGyr["z"])**2)
        processed_gravity = math.sqrt(statistics.mean(listGrv["x"])**2 + statistics.mean(listGrv["y"])**2 + statistics.mean(listGrv["z"])**2)

        data["accelerometer"].append(processed_accelerometer)
        data["accelerometer_max"].append(processed_accelerometer_max)
        data["target"].append(walkType)
          
        listAcc = {"x": [], "y": [], "z": []}
        listGyr = {"x": [], "y": [], "z": []}
        listGrv = {"x": [], "y": [], "z": []}

      print(f"Processed {line_count} lines.")
  
  targets = data["target"]
  targetsSet = list(set(targets))
  del data["target"]
  features = list(data.keys())
  dataframe = pandas.DataFrame(data=data)

  # https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
  encoder = LabelEncoder()
  encoder.fit(targets)
  encoded_targets = encoder.transform(targets)
  # convert integers to dummy variables (i.e. one hot encoded)
  dummy_targets = np_utils.to_categorical(encoded_targets)

  for n, i in enumerate(targets):
    targets[n] = targetsSet.index(i)

  model = Sequential([
    Dense(8, activation="relu"),
    Dense(len(targetsSet), activation="sigmoid")
  ])
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  #print(model.summary())
 
  dataset = tf.data.Dataset.from_tensor_slices((
    tf.cast(dataframe[features].values, tf.float32),
    dummy_targets
  )).shuffle(1) 

  #for features_tensor, target_tensor in dataset:
  #  print(f"features:{features_tensor} target:{target_tensor}")

  training_dataset, validation_dataset = split_dataset(dataset, 0.2)

  model.fit(training_dataset.repeat(2).shuffle(10, reshuffle_each_iteration=True).batch(16), epochs=100)

  eval_result = model.evaluate(validation_dataset.shuffle(1).batch(5))
  print("Test loss:", eval_result[0])
  print("Test accuracy:", eval_result[1])

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  def representative_dataset_gen():
      for _ in range(10000):
          yield [
              np.array(
                  [np.random.uniform(), np.random.uniform()]
              , dtype=np.float32)
          ]
  converter.representative_dataset = representative_dataset_gen
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  tflite_quant_model = converter.convert()
  hexArray = "".join("0x{:02x},".format(x) for x in tflite_quant_model)
  open(PROCESSED_FILE, "w").write("//AUTOGENERATED DO NOT CHANGE\nunsigned char tflite_model[] = {" + hexArray[:-1] + "};\nunsigned int tflite_model_len = "+str(len(tflite_quant_model))+";\n")

  print(features)
  print(targetsSet)
  print("done")
  

def main():
  process("raw_data", "processed", "model_data.cpp")

if __name__ == "__main__":
    main()