# Crop Recommendation System

This repository contains the implementation of a **Smart Agricultural Framework** for soil image classification and crop recommendation. The project leverages deep learning techniques for farm land assessment and Random Forest for recommending suitable crops.

---

## Objectives

1. **Soil Image Classification**: Classify soil images using deep learning (Modified DenseNet) to assess farmland quality.
2. **Crop Recommendation**: Develop an automatic crop recommendation system using the Random Forest algorithm.
3. **Leaf Disease Detection**: Implement a deep learning framework for detecting leaf diseases and assessing the extent of damage caused to crops.
4. **Crop Yield Prediction**: Use deep learning to predict crop yields accurately.

---

## Datasets
### 1. **Crop Recommendation Dataset**
- **Source**: Kaggle ([Crop Recommendation Dataset](https://www.kaggle.com/datasets))
- **Description**: Contains features like soil type, pH value, rainfall, and temperature for recommending crops.

## Prerequisites

### Tools and Libraries
- Python 3.8+
- TensorFlow / PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Kaggle API


## Project Structure

├── data/
│   ├── soil_images/
│   ├── crop_recommendation.csv
│   ├── leaf_disease_images/
├── models/
│   ├── soil_classification_model.h5
│   ├── crop_recommendation_model.pkl
├── notebooks/
│   ├── soil_classification.ipynb
│   ├── crop_recommendation.ipynb
├── src/
│   ├── soil_classifier.py
│   ├── crop_recommender.py
├── README.md
├── requirements.txt

## Results

- **Crop Recommendation**: Achieved XX% precision using Random Forest.
- 
## Contributing

Contributions are welcome! Please fork this repository and submit a pull request.

## Acknowledgments

- **PlantVillage Dataset** for soil and leaf images.
- **Kaggle** for providing the crop recommendation dataset.
- Special thanks to mentors and collaborators for their valuable guidance.

---
