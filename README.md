# SureSight

## Confidence-Guided Multi-Image Fusion for Diabetic Retinopathy Screening

SureSight is a lightweight deep learning framework for diabetic retinopathy (DR) screening from retinal fundus images.

The system combines:

- RETFoundGreen retinal image encoding
- transformer-based multi-image fusion
- patient-level confidence-guided prediction

to improve diagnostic reliability while allowing uncertain cases to be deferred for additional imaging or review.

---

## Motivation

Automated DR screening is especially valuable in settings where access to ophthalmological care is limited.

However, real-world fundus images can vary in illumination, focus, artifacts, positioning, lesion visibility

Traditional pipelines often use a separate image-quality classifier to reject poor-quality images. SureSight instead uses the diagnostic model's own confidence to determine whether a prediction is reliable.

---

## Pipeline

### 1. Image Encoding

Each retinal image is processed using a shared RETFoundGreen vision encoder to produce an image embedding.

### 2. Multi-Image Fusion

Embeddings from multiple fundus images of the same patient are combined using a **two-layer transformer encoder**.

The transformer supports variable numbers of input images through masking.

### 3. Patient-Level Prediction

The fused representation produces a single patient-level probability of diabetic retinopathy.

### 4. Confidence-Guided Deferral

Predictions are accepted only when sufficiently far from the decision boundary at 0.5:

`p < 0.5 - T` or `p > 0.5 + T`

Low-confidence patients are deferred rather than assigned an unreliable diagnosis.

---

## Compared Methods

SureSight evaluates three main strategies:

1. **Image-quality cascade**  
   A separate quality model filters images before DR diagnosis.

2. **Single-image confidence filtering**  
   Predictions are accepted or rejected using the DR model's confidence.

3. **Confidence-guided multi-image fusion**  
   Multiple images are fused before applying patient-level confidence filtering.

Multi-image fusion methods evaluated include:

- mean probability fusion
- max probability fusion
- transformer-based fusion

---

## Datasets

### mBRSET

Mobile Brazilian Multilabel Ophthalmological Dataset

- 5,164 original fundus images
- smartphone-based retinal imaging
- four images per patient

### BRSET

Brazilian Multilabel Ophthalmological Dataset

- 16,266 original fundus images
- tabletop clinical retinal imaging
- two images per patient
---
## Model

SureSight uses **RETFoundGreen**, a lightweight retinal foundation model based on a Vision Transformer.

The full fusion model contains approximately **25M parameters**, including approximately **3.6M parameters** in the multi-image transformer.

---

## Results

## Results

Transformer fusion consistently outperformed the single-image baseline and simple probability fusion.

### mBRSET Results

![mBRSET results](figures/mBRSET_graph.png)

### BRSET Results

![BRSET results](figures/brset_graph.png)

## Main Findings

## Main Findings

- Multi-image fusion improves patient-level DR prediction over single-image inference.
- Transformer fusion performs better overall than mean or max probability fusion.
- Model-confidence filtering consistently outperforms explicit image-quality filtering.
- Human image-quality labels are only weakly associated with downstream diagnostic performance.
- Increasing the number of available images improves the accuracy-coverage tradeoff.
- SureSight requires only one inference pass per image and can operate fully offline, supporting mobile deployment.
