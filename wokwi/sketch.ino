#include "model.h" 
#include <eloquent_tinyml.h>

// 1. Tensor Arena & Model Config
#define N_INPUTS 12
#define N_OUTPUTS 1
#define TENSOR_ARENA_SIZE (10 * 1024) // 10KB as suggested by your friend/PDF

// Initialize the EloquentTinyML wrapper
Eloquent::TinyML::TfLite<N_INPUTS, N_OUTPUTS, TENSOR_ARENA_SIZE> ml;

// Hardware Pins
const int LED_PIN = 2;
const int BUZZER_PIN = 4;

// ==============================================================================
// ✅ PHASE 1 CONSTANTS (StandardScaler mean_ and scale_)
float feature_means[12] = {
    48.539499, 938.627649, 12942.453936, 80.049042, 203.586487, 
    184.467770, 670.021044, 100.594309, 491.463608, 15.970424, 
    1942.057528, 19754.257912
}; 

float feature_stds[12]  = {
    8.865367, 1.331344, 272.464305, 1083.383189, 2214.738556, 
    1976.305615, 1905.885439, 922.524245, 4265.661251, 14.359576, 
    7811.589055, 609.513156
};

// Dummy baseline inputs (Normal Environment)
float raw_inputs[12] = {45.0, 1012.5, 13000, 2.0, 1.0, 5.0, 400, 2.0, 0.5, 22.0, 50, 19000};
float scaled_inputs[12]; // Array to hold normalized data

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  // Initialize the model in one line!
  ml.begin(fire_model_data);
  Serial.println("✅ Model Initialized Successfully!");
}

void loop() {
  // 1. Preprocessing (StandardScaler)
  for (int i = 0; i < 12; i++) {
    scaled_inputs[i] = (raw_inputs[i] - feature_means[i]) / feature_stds[i];
  }

  // 2. Real-Time Performance Tracking (Latency)
  long start_time = micros();
  
  // Predict! (EloquentTinyML automatically handles the 8-bit Quantization under the hood)
  float prediction = ml.predict(scaled_inputs);
  
  long end_time = micros();
  long latency = end_time - start_time;

  // 3. Serial Plotter Visualization (Sensor Fusion)
  Serial.print("Input_Trend(Temp):"); 
  Serial.print(raw_inputs[9]); 
  Serial.print(",");
  
  Serial.print("Model_Confidence:"); 
  Serial.print(prediction * 100); // Scale up for visual graph
  Serial.print(",");

  Serial.print("Latency(us):"); 
  Serial.print(latency);
  Serial.print(",");

  Serial.print("Free_Memory(bytes):"); 
  Serial.println(ESP.getFreeHeap());

  // 4. Hardware High-Dimensional Trigger
  if (prediction > 0.5) {
    digitalWrite(LED_PIN, HIGH);
    digitalWrite(BUZZER_PIN, HIGH);
  } else {
    digitalWrite(LED_PIN, LOW);
    digitalWrite(BUZZER_PIN, LOW);
  }

  // 5. Simulate a "Fire" progressing over time
  raw_inputs[9] += 1.5;   // Temperature rises
  raw_inputs[10] += 150;  // TVOC rises
  
  if (raw_inputs[9] > 80.0) {
    raw_inputs[9] = 22.0; 
    raw_inputs[10] = 50.0;
  }

  delay(250); // Loop speed
}