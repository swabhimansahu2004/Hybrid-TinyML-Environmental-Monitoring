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
| [**/data**](./data) | Contains the `smoke_detection_iot.csv` dataset featuring 15 physical and chemical sensors. |
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
- [ ] Phase 2: Training the "Teacher" Model
- [ ] Phase 3: Applying Hybrid Pruning and Distillation
- [ ] Phase 4: ESP32 Hardware Deployment and Real-time Testing
