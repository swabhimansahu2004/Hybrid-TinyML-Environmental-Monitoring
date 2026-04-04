# Integrated Hybrid Optimization TinyML Framework for Environmental Monitoring

### Group: 44-09 | Siksha 'O' Anusandhan (Deemed to be University)

## 📌 Project Overview
This research focuses on bridging the "memory gap" between complex deep learning models and resource-constrained microcontrollers like the **ESP32**. We are developing a TinyML system capable of real-time, on-device detection of environmental hazards (fire and smoke) using a fusion of temperature, humidity, and chemical gas signatures.

The core of this project is a **Hybrid Optimization Pipeline** designed to reduce model size and latency while maintaining high accuracy for safety-critical applications.

---

## 🛠️ The Hybrid Optimization Strategy
To fit a high-performance model into the **520KB SRAM** of an ESP32, we employ three key techniques:

1.  **Knowledge Distillation:** Training a lightweight "Student" model to mimic the complex logic of a high-capacity "Teacher" model.
2.  **Iterative Pruning:** Systematically removing redundant neural connections (targeting 40-60% sparsity) to minimize memory footprint.
3.  **8-bit Quantization:** Converting 32-bit floating-point weights into 8-bit integers to drastically reduce the binary size for deployment.

---

## 📂 Repository Structure

| Folder | Description |
| :--- | :--- |
| [**/data**](./data) | Contains the .csv datasets . |
| [**/notebooks**](./notebooks) | Google Colab notebooks for Data Analysis, Teacher Training, and Hybrid Optimization. |
| [**/models**](./models) | Final exported `.tflite` files and C++ header files for ESP32 deployment. |
| [**/docs**](./docs) | Project proposal, research papers, and technical documentation. |

---

## 📊 Dataset Features
The model is trained on a multi-dimensional dataset including:
* **Thermal:** Temperature [C], Humidity [%].
* **Gases:** Raw H2, Raw Ethanol, TVOC [ppb], eCO2 [ppm].
* **Particulates:** PM1.0, PM2.5, and Number Concentrations (NC0.5, NC1.0, NC2.5).
* **Atmospheric:** Pressure [hPa].

---

## 🚀 Future Roadmap
- [ ] Phase 1: Data Cleaning and Correlation Analysis (Completed)
- [ ] Phase 2: Training the "Teacher" Model (Completed)
- [ ] Phase 3: Applying Hybrid Pruning and Distillation
- [ ] Phase 4: ESP32 Hardware Deployment and Real-time Testing

## 📈 Phase 1: Data Engineering & Correlation Analysis
Before training, we audited the dataset to ensure "Research Integrity." The goal was to ensure the model learns environmental physics rather than metadata patterns.

### Key Actions:
* **Feature Selection:** Removed `UTC`, `CNT`, and `Unnamed: 0`. These metadata columns showed high correlation with the target but would lead to "Data Leakage," making the model fail in real-world deployment.
* **Normalization:** Implemented `StandardScaler` to handle range differences between TVOC (0–60,000 ppb) and Temperature (~20°C).
* **Statistical Validation:** Verified a clean dataset with zero missing values and a healthy 71/29 class distribution.

### Feature Correlation Heatmap:
The heatmap below justifies our feature selection, showing the strong physical relationship between Humidity, Gas levels (Raw H2/Ethanol), and the Fire Alarm trigger.

*Note: High negative correlation in Raw Ethanol and positive correlation in Humidity are key drivers for the model.*

---

## 🧠 Phase 2: Teacher Model Training (The "Expert")
We developed a high-capacity "Teacher" model to serve as the gold standard for intelligence. This model acts as the "Source of Truth" for the subsequent Knowledge Distillation phase.

### Architecture:
* **Type:** Deep Neural Network (DNN)
* **Complexity:** ~30,000 parameters (Dense Layers: 64 -> 128 -> 64 -> 32 -> 1)
* **Optimization:** Adam Optimizer with Binary Cross-Entropy loss.
* **Regularization:** 20% Dropout to ensure generalization.

### Results:
* **Final Test Accuracy:** **99.84%**
* **Inference Logic:** The model successfully identifies "Collective Anomalies" by evaluating the fusion of 12 distinct sensor inputs.

### Performance Learning Curves:
The training logs show perfect convergence. The narrow gap between the training and validation lines confirms that the model is not overfitted and is ready for distillation.

---

### 📦 Artifacts Generated
| File | Purpose |
| :--- | :--- |
| `scaler.pkl` | Vital for ESP32; contains mean/std constants for real-time data scaling. |
| `teacher_model.h5` | The high-accuracy expert model used for Phase 3. |
| `training_log.csv` | Full epoch-by-epoch audit trail of the training process. |

---

## 🧠 Phase 3: Hybrid Optimization (Distillation & Pruning)
In this phase, we execute the core "Hybrid" strategy: shrinking the high-capacity **Teacher** model into an ultra-lean **Student** model that fits the ESP32's memory constraints without sacrificing the 99.8% safety-critical accuracy.

### 1. Knowledge Distillation (The "Student" Brain)
We moved from a 30,000-parameter architecture to a **16-8-1 "Student" architecture** (Student_A). Instead of training on raw data alone, the Student was trained to mimic the Teacher's probability distributions.
* **Student Complexity:** Only **353 parameters**.
* **Result:** The Student achieved a baseline accuracy of **99.88%**, successfully capturing the "Expert" logic in a fraction of the size.

### 2. Iterative Magnitude Pruning (The "Surgical" Snipping)
To further optimize for the ESP32's Flash and SRAM, we applied **Iterative Pruning** to remove internal redundancy.
* **Target Sparsity:** **50%** (Half of all neural connections were "zeroed out").
* **Schedule:** 10 Epochs of **Polynomial Decay** pruning, allowing the model to "re-learn" its fire detection rules as connections were removed.
* **The Outcome:** The model maintained a final test accuracy of **99.85%**, proving that 50% of the original connections were redundant "noise."

---

### 📊 Phase 3: Optimization Impact Report
This report highlights the "Memory Journey" from the unoptimized Expert to the production-ready Student.

| Metric | Teacher Model (Phase 2) | Distilled Student | Pruned Student (Final) |
| :--- | :--- | :--- | :--- |
| **Model Type** | 64-128-64-32-1 | 16-8-1 | **16-8-1 (50% Sparse)** |
| **Parameters** | ~30,000+ | 353 | **353** |
| **Test Accuracy** | 99.84% | 99.88% | **99.85%** |
| **Uncompressed Size** | ~120 KB | 23.30 KB | **17.53 KB** |
| **Zipped Size (Flash)** | ~80 KB | 12.0 KB | **2.50 KB** |
| **Total Reduction** | Baseline | 80% Reduction | **98% Total Reduction** |

### 🔍 Research Significance:
The **2.50 KB** footprint is the definitive proof of the **Integrated Hybrid Optimization Framework**. 
* **Sparsity Benefit:** By creating a sparse weight matrix, the model is highly compressible. 
* **Edge Readiness:** This allows the ESP32 to run the fire detection logic in **microseconds**, leaving the CPU free to handle WiFi communication and sensor polling simultaneously.

---

### 📦 Artifacts Generated (Phase 3)
| File | Purpose |
| :--- | :--- |
| `best_distilled_student.h5` | The initial dense Student model (353 params). |
| `student_A_pruned.h5` | The final 50% sparse model after structural stripping. |
| `pruning_summary.txt` | Audit log of weight magnitudes and sparsity ratios. |

---

## 📉 Phase 4: 8-bit Post-Training Quantization (PTQ) & Benchmarking
This final optimization stage bridges the gap between high-level Python models and low-level hardware execution. We convert 32-bit floating-point weights into **8-bit integers**, enabling the ESP32 to perform inference using high-speed integer arithmetic.

### 1. 8-bit Quantization (The Final Shrink)
Using a **Representative Dataset** of 100 samples to calibrate the dynamic range of our sensors, the model was converted to a TFLite flatbuffer format.
* **Final Binary Size:** **3.69 KB**
* **Format:** 8-bit Integer Quantized (`.tflite`)
* **Hardware Compatibility:** Fully compatible with TensorFlow Lite for Microcontrollers (TFLite Micro).

### 2. Research Benchmarking: The "Optimization Gap"
We conducted a comparative study between the original **Teacher** and the **Optimized Student** to validate the framework's efficiency.

| Metric | Teacher (Phase 2) | Optimized Student (Phase 4) | Improvement |
| :--- | :--- | :--- | :--- |
| **Storage (KB)** | ~120 KB | **3.69 KB** | **97% Reduction** |
| **Inference Latency** | ~5.20 ms | **< 0.05 ms (Simulated)** | **~100x Speedup** |
| **Accuracy** | 99.84% | **99.80% (Estimated)** | **Minimal Loss (<0.1%)** |

### 🔍 Research Insight:
The benchmarking proves that the **Integrated Hybrid Optimization Framework** successfully creates a "Pareto Optimal" model. We achieved a near **100x speedup** and massive memory savings, which are critical for the ESP32 to maintain a 100% duty cycle for fire monitoring while simultaneously handling WiFi/MQTT communication.

---

### 📦 Artifacts Generated (Phase 4)
| File | Purpose |
| :--- | :--- |
| `fire_model_optimized.tflite` | Final 8-bit quantized model binary. |
| `model.h` | **Deployment Header:** Contains the model as a C++ hex array for Wokwi/Arduino. |
| `Model_BenchMarking.ipynb` | The notebook containing latency and memory audit logs. |

---

## 🚀 Future Roadmap (Final Update)
- [x] Phase 1: Data Cleaning and Correlation Analysis (Completed)
- [x] Phase 2: Training the "Teacher" Model (Completed)
- [x] Phase 3: Applying Hybrid Pruning and Distillation (Completed)
- [x] Phase 4: 8-bit Quantization and Performance Benchmarking (Completed)
- [ ] **Phase 5: ESP32 Hardware Deployment (Wokwi Simulation Phase)**
