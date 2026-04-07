#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define TENSOR_ARENA_SIZE (20 * 1024)
#define N_INPUTS 12

constexpr uint8_t LED_PIN    = 2;
constexpr uint8_t BUZZER_PIN = 4;

namespace {
  const tflite::Model*      tfl_model     = nullptr;
  tflite::MicroInterpreter* interpreter   = nullptr;
  TfLiteTensor*             input_tensor  = nullptr;
  TfLiteTensor*             output_tensor = nullptr;
  static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
}

static tflite::MicroMutableOpResolver<4> resolver;

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

float raw_inputs[N_INPUTS] = {
  45.0f, 1012.5f, 13000.0f, 2.0f, 1.0f, 5.0f,
  400.0f, 2.0f, 0.5f, 22.0f, 50.0f, 19000.0f
};

float  in_scale, out_scale;
int8_t in_zp,    out_zp;
bool   model_ok  = false;
bool   use_model = true;   // set false to test plotter with fake data only

inline void setAlert(bool on) {
  digitalWrite(LED_PIN,    on ? HIGH : LOW);
  digitalWrite(BUZZER_PIN, on ? HIGH : LOW);
}

void setup() {
  Serial.begin(9600);
       delay(500);   // give monitor time to attach

  pinMode(LED_PIN,    OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  // ── Fake-data smoke test: if plotter shows spikes, model is the problem ──
  for (int i = 0; i < 5; i++) {
    Serial.println("22.0,0.10,0,0");
    delay(200);
  }
               

  Serial.print("Free heap: ");
  Serial.println(ESP.getFreeHeap());

  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddQuantize();
  resolver.AddSoftmax();

  tfl_model = tflite::GetModel(fire_model_data);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Schema mismatch");
    use_model = false;
  } else {
    Serial.println("Model loaded OK");

    static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, TENSOR_ARENA_SIZE
    );
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
      Serial.println("ERROR: AllocateTensors failed - falling back to fake data");
      use_model = false;
    } else {
      input_tensor  = interpreter->input(0);
      output_tensor = interpreter->output(0);
      in_scale  = input_tensor->params.scale;
      in_zp     = input_tensor->params.zero_point;
      out_scale = output_tensor->params.scale;
      out_zp    = output_tensor->params.zero_point;

      Serial.print("Arena used: ");
      Serial.print(interpreter->arena_used_bytes());
      Serial.print(" / ");
      Serial.println(TENSOR_ARENA_SIZE);
      Serial.printf("Input  type: %d\n", input_tensor->type);
      Serial.printf("Output type: %d\n", output_tensor->type);
      model_ok = true;
      Serial.println("Model ready!");
    }
  }

  Serial.println("Temp,Confidence,Latency_us,FreeHeap");
}

void loop() {
  float prediction = 0.0f;
  uint32_t latency = 0;

  if (use_model && model_ok) {
    // ── Quantized or float input write ──────────────────────────────────────
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
      Serial.println("ERROR: Invoke failed");
      delay(250);
      return;
    }
    latency = micros() - t0;

    if (output_tensor->type == kTfLiteFloat32) {
      prediction = output_tensor->data.f[0];
    } else {
      prediction = (output_tensor->data.int8[0] - out_zp) * out_scale;
    }

  } else {
    // ── Fallback: fake data so plotter always shows something ───────────────
    float t = raw_inputs[9];
    prediction = (t > 60.0f) ? 0.85f : 0.05f;
    latency    = 999;
  }

  bool fire = (prediction > 0.5f);
  setAlert(fire);

  // ── CSV output for Serial Plotter ────────────────────────────────────────
  Serial.print(raw_inputs[9], 1);
  Serial.print(",");
  Serial.print(prediction, 4);
  Serial.print(",");
  Serial.print(latency);
  Serial.print(",");
  Serial.println(ESP.getFreeHeap());
  Serial.flush();

  // ── Simulate rising fire signature ───────────────────────────────────────
  raw_inputs[9]  += 1.5f;
  raw_inputs[10] += 150.0f;
  raw_inputs[6]  += 20.0f;
  raw_inputs[3]  += 0.5f;
  raw_inputs[5]  += 0.8f;
  raw_inputs[7]  += 0.3f;
  raw_inputs[8]  += 2.0f;
  raw_inputs[2]  += 50.0f;

  if (raw_inputs[9] > 120.0f) {
    raw_inputs[9]  = 22.0f;
    raw_inputs[10] = 50.0f;
    raw_inputs[6]  = 400.0f;
    raw_inputs[3]  = 2.0f;
    raw_inputs[5]  = 5.0f;
    raw_inputs[7]  = 2.0f;
    raw_inputs[8]  = 0.5f;
    raw_inputs[2]  = 13000.0f;
  }

  delay(250);
}