#include <Arduino.h>
#include <GY521.h>
#include <LinkedList.h>
#include <SPI.h>
#include <SSD1306Ascii.h>
#include <SSD1306AsciiWire.h>
#include <SdFat.h>
#include <Wire.h>
#include <jled.h>

#include "NeuralNetwork.h"

#define SDCARD_CS 7
#define USER_KEY 0
#define SAMPLES_FILE "sample.csv"

SdFat sd;
FsFile f;

GY521 sensor(0x68);
SSD1306AsciiWire oled;

NeuralNetwork *nn;

volatile bool UKEYFlag = false;

JLed led = JLed(LED_BUILTIN).Blink(1000, 500).Forever();

const char *ACTIVITIES[] = {"walk", "run", "stand"};
uint8_t state = 0;
uint32_t lastSample = 0;
uint32_t printIt = 0;
uint32_t deleteIt = 0;

typedef struct XYZSensor {
    float xAccelerometer;
    float yAccelerometer;
    float zAccelerometer;
} XYZSensor_t;

LinkedList<XYZSensor_t> samples = LinkedList<XYZSensor_t>();

void UKeyISR() {
    UKEYFlag = true;
}

void setup() {
    SerialUSB.begin(MON_SPEED);
    for (uint8_t i = 0; i < 8; i++) {
        SerialUSB.print(".");
        delay(250);
    }
    SerialUSB.println("");

    nn = new NeuralNetwork(3);

    pinMode(USER_KEY, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(USER_KEY), UKeyISR, LOW);

    while (!sd.begin(SDCARD_CS)) {
        SerialUSB.println("SD Card not found");
        delay(1000);
    }
    SerialUSB.println("SD Card initialized");

    Wire.begin();

    oled.begin(&Adafruit128x32, 0x3C);
    oled.setFont(Adafruit5x7);
    oled.clear();
    oled.set1X();

    SerialUSB.println("Initializing GY521...");
    while (sensor.wakeup() == false) {
        SerialUSB.println("Could not connect to GY521");
        delay(1000);
    }
    SerialUSB.println("GY521 initalized");

    delay(100);
    sensor.setAccelSensitivity(0);  // 2g
    sensor.setGyroSensitivity(0);   // 250 degrees/s
    sensor.setThrottle(false);

    sensor.axe = -0.066;
    sensor.aye = -0.056;
    sensor.aze = -0.981;
    sensor.gxe = 0.437;
    sensor.gye = -3.150;
    sensor.gze = -0.366;
}

void loop() {
    led.Update();

    if (UKEYFlag) {
        SerialUSB.println("Button pressed!");
        UKEYFlag = false;
        state++;
        if (state >= 5) state = 0;

        oled.clear();
        if (state > 0 && state < 4) {
            oled.set2X();
            oled.println("Recording...");
            oled.println(ACTIVITIES[state - 1]);

            if (!f.isOpen()) {
                f = sd.open(SAMPLES_FILE, O_WRITE | O_CREAT | O_APPEND);
                if (!f) {
                    oled.set1X();
                    oled.println("failed to open");
                    oled.println(SAMPLES_FILE);
                } else {
                    f.seek(EOF);
                }
            }
        } else if (state == 4) {
            deleteIt = millis();

            oled.set1X();
            oled.println("stay here to delete");
            oled.println(SAMPLES_FILE);
        } else {
            oled.set1X();
            oled.println("collecting samples");

            if (f.isOpen()) f.close();
        }
    }

    if (state == 4) {
        const uint32_t currentMillis = millis();
        if (currentMillis - deleteIt > 5000) {
            if (f.remove()) {
                oled.println("deleted");
                delay(1000);
                NVIC_SystemReset();
            } else {
                oled.println("failed to deleted");
            }
            delay(500);
            state = 0;
        }
    } else if (state == 0) {
        if (samples.size() >= 200) {
            float ax = 0;
            float ay = 0;
            float az = 0;
            float maxX = -999;
            float maxY = -999;
            float maxZ = -999;

            for (int i = 0; i < samples.size(); i++) {
                XYZSensor_t val = samples.get(i);
                ax += val.xAccelerometer;
                ay += val.yAccelerometer;
                az += val.zAccelerometer;

                if (val.xAccelerometer > maxX) maxX = val.xAccelerometer;
                if (val.yAccelerometer > maxY) maxY = val.yAccelerometer;
                if (val.zAccelerometer > maxZ) maxZ = val.zAccelerometer;
            }

            ax /= samples.size();
            ay /= samples.size();
            az /= samples.size();

            samples.clear();

            float smvAvarage = sqrt(pow(ax, 2) + pow(ay, 2) + pow(az, 2));
            float smvMax = sqrt(pow(maxX, 2) + pow(maxY, 2) + pow(maxZ, 2));

            SerialUSB.print("smvAvarage: ");
            SerialUSB.println(smvAvarage);

            SerialUSB.print("smvMax: ");
            SerialUSB.println(smvMax);

            nn->getInputBuffer()[0] = smvAvarage;  // accelerometer
            nn->getInputBuffer()[1] = smvMax;      // accelerometer_max

            LinkedList<float *> result = nn->predict();
            oled.clear();
            for (int i = 0; i < result.size(); i++) {
                float *accuarcy = result.get(i);
                SerialUSB.print(ACTIVITIES[i]);
                SerialUSB.print(": ");
                SerialUSB.println(*accuarcy);

                oled.print(ACTIVITIES[i]);
                oled.print(": ");
                oled.println(*accuarcy);
            }

            SerialUSB.println();
        }

        if (millis() - lastSample >= 10) {
            lastSample = millis();
            printIt++;
            sensor.read();

            XYZSensor_t val;
            val.xAccelerometer = sensor.getAccelX();
            val.yAccelerometer = sensor.getAccelY();
            val.zAccelerometer = sensor.getAccelZ();

            samples.add(val);

            if (printIt >= 50) {
                printIt = 0;
                char buf[256];

                sprintf(buf, "%d:\tAccX: %s\tAccY: %s\tAccZ: %s\tGyrX: %s\tGyrY: %s\tGyrZ: %s\tTemperature: %s",
                        samples.size(),
                        String(val.xAccelerometer, 6).c_str(),
                        String(val.yAccelerometer, 6).c_str(),
                        String(val.zAccelerometer, 6).c_str(),
                        String(sensor.getGyroX(), 6).c_str(),
                        String(sensor.getGyroY(), 6).c_str(),
                        String(sensor.getGyroZ(), 6).c_str(),
                        String(sensor.getTemperature(), 2).c_str());
                SerialUSB.println(buf);
            }
        }
    } else {
        if (millis() - lastSample >= 100) {
            lastSample = millis();
            sensor.read();
            char buf[256];

            sprintf(buf, "%lu;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s",
                millis(),                                   // 0
                    String(sensor.getAccelX(), 6).c_str(),  // 1
                    String(sensor.getAccelY(), 6).c_str(),  // 2
                    String(sensor.getAccelZ(), 6).c_str(),  // 3
                    String(sensor.getGyroX(), 6).c_str(),   // 4
                    String(sensor.getGyroY(), 6).c_str(),   // 5
                    String(sensor.getGyroZ(), 6).c_str(),   // 6
                    String(sensor.getPitch(), 2).c_str(),   // 7
                    String(sensor.getYaw(), 2).c_str(),     // 8
                    String(sensor.getRoll(), 2).c_str(),    // 9
                    ACTIVITIES[state - 1]);                 // 10
            f.println(buf);
        }
    }
}