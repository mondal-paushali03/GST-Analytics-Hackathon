# GST Analytics Hackathon 🚀  

A machine learning project developed for the **GST Analytics Hackathon**, focusing on detecting and classifying anomalies in GST-related datasets using advanced data preprocessing, feature engineering, and ensemble learning models.  

## 📊 Project Overview  

This project leverages **Random Forest Classifier** along with data transformation and feature selection techniques to build a robust model for classification. The workflow involves:  

- Data Cleaning & Preprocessing (handling missing values, outliers, and skewness)  
- Feature Engineering (mutual information gain & correlation-based feature selection)  
- Model Training with Random Forest  
- Model Evaluation using Accuracy, MSE, Classification Report, Confusion Matrix, and AUC-ROC Curve  

The aim is to enable **data-driven insights** into GST datasets for anomaly detection and classification tasks.  

---

## ⚙️ Tech Stack  

- **Python** (pandas, numpy, seaborn, matplotlib, scikit-learn)  
- **Machine Learning** – Random Forest Classifier  
- **Feature Engineering** – Mutual Information Gain, Power Transformation, Outlier Treatment  
- **Visualization** – Heatmaps, Histograms, Boxplots, Correlation Matrix, ROC & Confusion Matrix  

---

## 🔑 Key Steps  

### 1. Data Preprocessing  
- Missing values handled with **mean, median, mode, and forward fill**.  
- Skewness reduced using **Yeo-Johnson Power Transformation**.  
- Outliers treated using **IQR method**.  

### 2. Feature Engineering  
- Correlation heatmap to remove redundant variables.  
- **Mutual Information Gain** to rank features.  
- Dropped low-importance columns before training.  

### 3. Model Building  
- Random Forest Classifier tuned with:  
  ```python
  RandomForestClassifier(
      n_estimators=300, 
      max_depth=10, 
      min_samples_split=4
  )
## 4. Model Evaluation  

- **Training Accuracy:** ~ 98%  
- **Test Accuracy:** ~ 98%  
- **MSE:** ~ 0.024  
- **Classification Report** with Precision, Recall, F1-Score  

📌 **Confusion Matrix:**  
<img width="533" height="455" alt="image" src="https://github.com/user-attachments/assets/c2c6f067-e71d-4023-898e-32bd8010f7f4" />


📌 **AUC-ROC Curve:**  
<img width="613" height="470" alt="image" src="https://github.com/user-attachments/assets/cf2cf648-ebb7-4188-b43d-7636edd04c88" />


---

## 📈 Results  

- The model achieved a balanced performance across both classes.  
- ROC-AUC score highlighted strong discriminatory power.  
- Outlier and skewness handling improved model stability.  

---

## 🚀 Future Improvements  

- Try **XGBoost / LightGBM** for better performance.  
- Perform **hyperparameter tuning** using GridSearchCV.  
- Build a **dashboard (Streamlit/Power BI)** for better visualization.  
- Deploy the model using **Flask/Django**.  

---

## 👩‍💻 Author  

**Paushali Mondal**  
📌 Department of Computer Science, VIT Bhopal University  
🔗 [LinkedIn](https://linkedin.com/in/paushali-mondal-483ba4250) | [GitHub](https://github.com/mondal-paushali03)  

---
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
