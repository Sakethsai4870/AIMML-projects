# Network Intrusion Detection using Machine Learning

## Overview
This project implements a machine learning-based network intrusion detection system using the NSL-KDD dataset. The objective is to classify network traffic as either "Normal" or "Attack" by training and evaluating multiple machine learning models.

## Dataset
The **NSL-KDD** dataset is an improved version of the KDD99 dataset, used for anomaly detection in network traffic. It contains labeled records of normal and attack network activities.

## Models Implemented
The following machine learning models were trained and evaluated:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**

## Performance Evaluation Metrics
The models were compared using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **AUC-ROC Curve**

## Installation and Setup
### Prerequisites
Ensure you have the following installed:
- Python (>=3.7)
- Jupyter Notebook or any Python IDE
- Required libraries (install using the command below)

### Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/network-intrusion-detection.git
cd network-intrusion-detection
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preprocessing
The following preprocessing steps were performed:
1. **Label Encoding**: Converted categorical features (protocol, service, flag) into numeric values.
2. **Feature Scaling**: Standardized numeric features using `StandardScaler`.
3. **Feature Selection**: Used `SelectKBest` to select the top 20 most relevant features.
4. **Train-Test Split**: Split the dataset into training (80%) and testing (20%).

## Model Training and Evaluation
Each model was trained using the processed dataset, and hyperparameter tuning was performed using GridSearchCV (for Decision Tree Classifier). Performance metrics were calculated, and results were visualized using bar plots and ROC curves.

## Results
After evaluating the models, the best-performing model was identified based on the highest accuracy and F1-score.

## Visualization
- **Bar plots** comparing accuracy and F1-score across different models.
- **ROC curves** to analyze classification performance.

## Usage
To run the project:
```bash
python main.py
```
Or open and run the Jupyter Notebook:
```bash
jupyter notebook network_intrusion_detection.ipynb
```

## Future Improvements
- Implementing deep learning models (e.g., Neural Networks, LSTMs).
- Exploring more feature selection techniques.
- Optimizing hyperparameters further for better performance.

## Contributors
- **Your Name** - [GitHub Profile](https://github.com/yourusername)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

