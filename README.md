# 🧠 DLOR Project – Part 2: Transfer Learning & Android Deployment for Traffic Sign Recognition

![Python](https://img.shields.io/badge/Made%20with-Python-blue.svg?logo=python)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange?logo=tensorflow)
![Dataset](https://img.shields.io/badge/Dataset-GTSRB-yellowgreen)
![Model](https://img.shields.io/badge/Model-VGG19%20%7C%20MobileNet%20%7C%20ResNet50-blue)
![Deployment](https://img.shields.io/badge/Deployment-Android-success)
![Status](https://img.shields.io/badge/Progress-Completed-brightgreen)

This repository documents **Part 2** of the DLOR project, which builds upon Part 1’s foundational work by advancing into **transfer learning, model optimization, evaluation, and mobile deployment**. The goal is to build a highly accurate traffic sign recognition model ready for real-world use.

---

## 🎯 Objectives

- Implement multiple pre-trained CNN architectures (e.g. VGG19, ResNet50, MobileNet)
- Fine-tune models for GTSRB dataset classification
- Apply hyperparameter tuning using **Keras Tuner**
- Compare baseline and transfer learning results
- Deploy the best model to an Android app

---

## 🧠 Models Explored

| Model           | Accuracy (Before Tuning) | Accuracy (After Tuning) |
|----------------|--------------------------|-------------------------|
| **CNN (Custom)** | 92.77%                  | —                       |
| **MobileNet**    | ~93%                    | —                       |
| **ResNet50**     | ~94%                    | —                       |
| **VGG19**        | 94.48%                  | ✅ **97.22%** (Best)     |

---

## 🔧 Hyperparameter Tuning with Keras Tuner

- **Method**: RandomSearch
- **Parameters Tuned**:
  - Learning rate
  - Dropout rate
  - Number of dense units
- **Best model**: Fine-tuned VGG19

---

## 🚀 Android Deployment

- The trained VGG19 model (after tuning) was converted and integrated into an Android application.
- Real-time recognition tested with live images on-device.
- UI built for camera input and label prediction output.

> ✅ Successfully tested and deployed using Android Studio with TensorFlow Lite support.

---

## 🧪 Evaluation

- **Confusion Matrix**: Used to analyze model predictions vs. actual labels
- **Misclassified Samples**: Visualized for deeper error analysis
- **Balanced Dataset**: Addressed class imbalance with oversampling/undersampling techniques

---

## ⚠️ Challenges

- Overfitting in deeper models: resolved via Dropout and EarlyStopping
- Dataset imbalance: addressed through augmentation and class weights
- Real-time deployment: optimized model size for mobile performance

---

## 🚧 Future Improvements

- Implement **more augmentation**: e.g., perspective, occlusion
- Try **additional transfer learning models**
- Use **k-fold cross-validation** for robustness
- Optimize TFLite model for latency and size

---

## 📁 Repository Structure

```
├── Final_DLOR_Project_Part_2.ipynb      # Jupyter notebook with modeling and training
├── requirements.txt                     # Required Python libraries
├── DLOR_Project_Report.pptx             # Final project presentation/report
```

---

## 👤 Author

**Moanish Ashok Kumar**  
Applied AI Student
🔗 [LinkedIn](https://www.linkedin.com/in/moanish-ashok-kumar-086978272/)

---

