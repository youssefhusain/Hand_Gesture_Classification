# Hand Gesture Recognition using Machine Learning

## 📌 Project Overview
This project focuses on recognizing hand gestures using machine learning techniques. It processes hand landmark data, performs feature engineering, applies various classification models, and evaluates their performance to achieve optimal results.
![The main box](img.gif)
## 📂 Dataset
- The dataset consists of hand landmark coordinates (x, y) extracted from images.
- The dataset is preprocessed by normalizing coordinates relative to the wrist position and balancing classes using SMOTE.
- The preprocessed dataset is saved as `data_after_prepration.csv`.

## 🛠️ Preprocessing Steps
1. Load dataset from `hand_landmarks_data.csv`.
2. Remove unnecessary `z` coordinates.
3. Normalize coordinates relative to the wrist.
4. Perform data visualization using Seaborn.
5. Apply SMOTE to handle class imbalance.
6. Encode labels using `LabelEncoder`.
7. Split data into training, validation, and test sets.

## 📊 Machine Learning Models Used
The following models were implemented and evaluated:

1. **Random Forest** - Tuned using GridSearchCV.
2. **Support Vector Machine (SVM)** - Hyperparameter tuning performed.
3. **Extra Trees Classifier** - Optimized using cross-validation.
4. **K-Nearest Neighbors (KNN)** - Hyperparameter tuning applied.
5. **AdaBoost Classifier** - Utilized Decision Trees as weak learners.
6. **Stacking Classifier** - Combined multiple classifiers with Logistic Regression as meta-learner.
7. **Voting Classifier** - Combined multiple models for ensemble learning.

## 🔥 Performance Evaluation
Each model's accuracy was assessed using a test set. The final results:

| Model                  | Accuracy |
|------------------------|----------|
| Random Forest         | 95.5%   |
| SVM                   | 94%   |
| Extra Trees           | 96%   |
| KNN                   | 88.8%   |
| AdaBoost              | 94.4%   |
| Stacking Classifier   | 97.7%   |
| Voting Classifier     | 98%   |

## 📌 How to Run
1. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```
2. Run the preprocessing script:
   ```python
   python preprocess.py
   ```
3. Train the models:
   ```python
   python train_models.py
   ```
4. Evaluate results and visualize predictions.

## 🚀 Future Enhancements
- Implement deep learning models (CNNs) for improved accuracy.
- Optimize feature selection and engineering.
- Deploy as a real-time application using Flask or FastAPI.

## 🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## 📜 License
This project is open-source and available under the MIT License.

