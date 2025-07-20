# 🏙️ Deep Learning for Building Image Classification & Generation

This project uses **Convolutional Neural Networks (CNNs)** in an **encoder-decoder architecture** to classify and generate images of buildings. The model is trained to learn the mapping between input sketches or noisy images and their corresponding realistic building photos.

---

## 🧠 Project Overview

The objective is twofold:
- **Classification**: Identify different types or styles of buildings from input images.
- **Image Generation**: Reconstruct realistic images from input sketches using encoder-decoder CNNs.

This approach leverages **deep learning** for visual feature extraction and transformation, enabling both recognition and reconstruction tasks.

---

## 🛠️ Technologies Used

- 🐍 **Python**  
- 🧠 **TensorFlow / Keras**  
- 🖼️ **OpenCV** for image preprocessing  
- 📊 **Matplotlib / Seaborn** for visualization  
- 🔧 **SSIM (Structural Similarity Index)** for evaluating image quality

---

## 🧬 Model Architecture

### 🔹 Encoder
- Multiple convolutional layers to extract spatial features
- Batch normalization and ReLU activation
- Downsampling via max pooling

### 🔹 Decoder
- Upsampling through transposed convolutions
- ReLU activations and skip connections (U-Net-like)
- Final layer with sigmoid activation for image reconstruction

---

## 📦 Dataset

- Real-world building images and corresponding sketches or noisy representations
- Preprocessing included resizing, normalization, and data augmentation techniques

---

## 📊 Results Summary

### ✅ **Image Realism**
- **SSIM score**: `0.7624`  
- Indicates a **moderate similarity** between generated and real images  
- Outputs are close to reality, but with **visible imperfections**

### ⚠️ **Common Errors & Artifacts**
- 🎨 **Color inconsistency**: Slight shifts in tones and hues  
- 🧱 **Shape distortion**: Some objects are misinterpreted or warped  
- 🖌️ **Loss of fine details**: Missing textures or shading precision  
- 💡 **Lighting issues**: Unrealistic or inconsistent illumination effects

---

## 🚀 Recommendations for Improvement

- **📈 Data Augmentation**: Apply rotations, zoom, flips, etc., to enrich training data  
- **⏳ Longer Training**: Increase number of epochs to allow finer feature learning  
- **🎯 Hyperparameter Tuning**: Adjust learning rate, loss functions, optimizers  
- **🧠 Advanced Architectures**:
  - Try **Pix2Pix** or **CycleGAN** for more realistic image-to-image translation  
  - Add extra layers or explore **PatchGAN** for detail enhancement  

---

## 🧾 Conclusion

The model demonstrates strong potential in generating plausible building images. While results are already acceptable, further improvements in training time, network architecture, and data variability can significantly enhance the quality and realism of generated outputs.
