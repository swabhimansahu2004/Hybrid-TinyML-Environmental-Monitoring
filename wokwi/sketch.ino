#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ── Arena: start at 10KB as per handover doc, safe for Wokwi ─────────────────
// Remove IRAM_ATTR — Wokwi's emulator doesn't support it and will crash silently
#define TENSOR_ARENA_SIZE (10 * 1024)
#define N_INPUTS 12

constexpr uint8_t LED_PIN    = 2;
constexpr uint8_t BUZZER_PIN = 4;

namespace {
  const tflite::Model*      tfl_model     = nullptr;
  tflite::MicroInterpreter* interpreter   = nullptr;
  TfLiteTensor*             input_tensor  = nullptr;
  TfLiteTensor*             output_tensor = nullptr;

  // NO IRAM_ATTR — causes silent crash in Wokwi emulator
  static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
}

// ── Exact ops from your model binary (confirmed) ──────────────────────────────
static tflite::MicroMutableOpResolver<3> resolver;

// ── ScalerParams in flash ─────────────────────────────────────────────────────
const float feature_means[N_INPUTS] PROGMEM = {
  48.539499f, 938.627649f, 12942.453936f, 80.049042f,
  203.586487f, 184.467770f, 670.021044f, 100.594309f,
  491.463608f, 15.970424f, 1942.057528f, 19754.257912f
};
const float feature_stds[N_INPUTS] PROGMEM = {
  8.865367f, 1.331344f, 272.464305f, 1083.383189f,
  2214.738556f, 1976.305615f, 1905.885439f, 922.524245f,
  4265.661251f, 14.359576f, 7811.589055f, 609.513156f
};

// ── Simulation state (matches handover doc feature order) ─────────────────────
// Humidity, Pressure, RawH2, NC2.5, NC1.0, PM2.5, eCO2, PM1.0, NC0.5, Temp, TVOC, RawEthanol
float raw_inputs[N_INPUTS] = {
  45.0f, 1012.5f, 13000.0f, 2.0f, 1.0f, 5.0f,
  400.0f, 2.0f, 0.5f, 22.0f, 50.0f, 19000.0f
};

float    in_scale,  out_scale;
int8_t   in_zp,     out_zp;
bool     model_ok = false;  // Guard flag — prevents loop() running if setup failed

inline void setAlert(bool on) {
  digitalWrite(LED_PIN,    on ? HIGH : LOW);
  digitalWrite(BUZZER_PIN, on ? HIGH : LOW);
}

void setup() {
  Serial.begin(115200);
  // Give Wokwi serial monitor time to attach
  delay(500);

  pinMode(LED_PIN,    OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  Serial.println(F("=== Fire Detection TinyML ==="));
  Serial.printf("Free heap at start: %u bytes\n", ESP.getFreeHeap());

  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddQuantize();

  tfl_model = tflite::GetModel(fire_model_data);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("ERROR: Schema version mismatch!"));
    return; // Don't while(true) — Wokwi can't show anything if halted
  }
  Serial.println(F("Model loaded OK"));

  static tflite::MicroInterpreter static_interpreter(
    tfl_model, resolver, tensor_arena, TENSOR_ARENA_SIZE
  );
  interpreter = &static_interpreter;

  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println(F("ERROR: AllocateTensors failed!"));
    Serial.println(F("Try increasing TENSOR_ARENA_SIZE to 15*1024"));
    return; // Return instead of halt so serial output stays visible
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);

  in_scale  = input_tensor->params.scale;
  in_zp     = input_tensor->params.zero_point;
  out_scale = output_tensor->params.scale;
  out_zp    = output_tensor->params.zero_point;

  Serial.printf("Arena used:    %u / %u bytes\n",
    interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);
  Serial.printf("Input  type:   %d (1=float32, 9=int8)\n", input_tensor->type);
  Serial.printf("Output type:   %d\n", output_tensor->type);
  Serial.printf("Free heap now: %u bytes\n", ESP.getFreeHeap());
  Serial.println(F("=== Starting inference loop ==="));
  Serial.println(F("Input_Temp,Model_Confidence,Latency_us,Free_Heap"));

  model_ok = true;
}

void loop() {
  if (!model_ok) {
    delay(2000);
    Serial.println(F("Setup failed — check errors above"));
    return;
  }

  // ── Write inputs ────────────────────────────────────────────────────────────
  if (input_tensor->type == kTfLiteFloat32) {
    for (int i = 0; i < N_INPUTS; i++) {
      input_tensor->data.f[i] =
        (raw_inputs[i] - pgm_read_float(&feature_means[i]))
        / pgm_read_float(&feature_stds[i]);
    }
  } else {
    for (int i = 0; i < N_INPUTS; i++) {
      float norm = (raw_inputs[i] - pgm_read_float(&feature_means[i]))
                   / pgm_read_float(&feature_stds[i]);
      int32_t q = (int32_t)roundf(norm / in_scale) + in_zp;
      input_tensor->data.int8[i] = (int8_t)constrain(q, -128, 127);
    }
  }

  uint32_t t0 = micros();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println(F("ERROR: Invoke failed!"));
    return;
  }
  uint32_t latency = micros() - t0;

  float prediction;
  if (output_tensor->type == kTfLiteFloat32) {
    prediction = output_tensor->data.f[0];
  } else {
    prediction = (output_tensor->data.int8[0] - out_zp) * out_scale;
  }

  bool fire = (prediction > 0.5f);
  setAlert(fire);

  // ── Serial Plotter format (CSV header printed once in setup) ────────────────
  Serial.printf("%.1f,%.4f,%lu,%u\n",
    raw_inputs[9],       // Temp — visible trend line
    prediction,          // Model confidence 0.0–1.0
    latency,             // Inference time in microseconds
    ESP.getFreeHeap()    // Heap stability check
  );

  // ── Simulate fire signature progressing ────────────────────────────────────
  raw_inputs[9]  += 1.5f;   // Temp rises
  raw_inputs[10] += 150.0f; // TVOC rises

  if (raw_inputs[9] > 80.0f) {
    raw_inputs[9]  = 22.0f;
    raw_inputs[10] = 50.0f;
  }

  delay(250);
}