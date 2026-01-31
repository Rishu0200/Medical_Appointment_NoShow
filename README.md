# Medical Appointment No-Show Prediction

A machine learning project that predicts whether patients will show up for their healthcare appointments, helping hospitals reduce no-shows and optimize scheduling.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Performance Metrics](#performance-metrics)
- [Deployment](#deployment)
- [Contributing](#contributing)

## 🎯 Overview

Healthcare facilities face significant challenges with patient no-shows, leading to wasted resources and reduced operational efficiency. This project uses machine learning to predict appointment no-shows based on patient demographics and appointment characteristics, enabling proactive interventions.

## ✨ Features

- **Data Pipeline**: Automated data ingestion from Google Cloud Storage
- **ML Model**: LightGBM classifier with hyperparameter optimization
- **Experiment Tracking**: MLflow integration for reproducible experiments
- **Web Interface**: Flask-based application for real-time predictions
- **Containerization**: Docker support for easy deployment
- **CI/CD**: Jenkins pipeline for automated testing and deployment

## 📁 Project Structure

Healthcare_appointment/
├── src/ # Core ML modules
│ ├── data_ingestion.py # Download and split data from GCS
│ ├── data_preprocessing.py # Data cleaning and feature engineering
│ ├── model_training.py # Model training and hyperparameter tuning
│ ├── logger.py # Logging configuration
│ └── custom_exception.py # Custom exception handling
├── pipeline/
│ └── training_pipeline.py # Orchestrates the entire ML pipeline
├── config/
│ ├── config.yaml # Configuration parameters
│ ├── paths_config.py # File paths configuration
│ └── model_params.py # Model hyperparameters
├── artifacts/
│ ├── raw/ # Raw data from GCS
│ ├── processed/ # Preprocessed datasets
│ └── models/ # Trained model files
├── templates/ # HTML templates for web UI
├── static/ # CSS and static assets
├── notebook/ # Jupyter notebooks for exploration
├── application.py # Flask web application
├── requirements.txt # Python dependencies
├── setup.py # Package setup configuration
├── Dockerfile # Docker image configuration
└── Jenkinsfile # CI/CD pipeline configuration


## 🚀 Installation

### Prerequisites
- Python 3.8+
- Google Cloud Storage credentials (for data ingestion)
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Healthcare_appointment



2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

**3. Install dependencies**
pip install -r requirements.txt

##💻 Usage
Training the Model
Run the complete ML pipeline:
python pipeline/training_pipeline.py

## Running the Web Application
python application.py

```
The Flask app will start at http://localhost:5000 where you can:

Enter patient details through the web form
Get real-time appointment show-up predictions
View prediction probability
Making Predictions
Input features:

Gender: Female (F) / Male (M)
Age: Patient age
Scholarship: Scholarship status (0/1)
Hypertension: Medical condition (0/1)
Diabetes: Medical condition (0/1)
Alcoholism: Medical condition (0/1)
Handcap: Handicap status (0/1)
SMS Received: SMS reminder received (0/1)
Neighbourhood: Appointment location
Date Difference: Days until appointment
🤖 Model Details
Algorithm
LightGBM (Light Gradient Boosting Machine)

Fast and efficient gradient boosting framework
Handles imbalanced classification well
Interpretable feature importance
Hyperparameter Tuning
Method: RandomizedSearchCV
Cross-validation: 5-fold CV
Optimization Metric: F1-Score
Number of iterations: Configurable in model_params.py
Key Parameters
Number of leaves (num_leaves)
Learning rate
Number of estimators
Min child samples
Subsample ratio
Column sampling
📊 Performance Metrics
Model evaluation metrics tracked via MLflow:

Accuracy: Overall prediction correctness
Precision: True positives among predicted positives
Recall: True positives among actual positives
F1-Score: Harmonic mean of precision and recall
Training runs are logged in mlruns directory for experiment comparison.

🐳 Deployment
Docker Deployment
Build the Docker image:

```bash
docker build -t healthcare-appointment:latest .
```
Run the container:
```bash
docker run -p 5000:5000 healthcare-appointment:latest
```
Jenkins CI/CD
Automated pipeline defined in Jenkinsfile handles:

Automated testing
Model training
Image building and pushing
Deployment to production
📦 Dependencies
Data Processing: pandas, numpy
ML: scikit-learn, LightGBM, imbalanced-learn
Experiment Tracking: MLflow
Web Framework: Flask
Cloud: google-cloud-storage
Others: PyYAML, python-dotenv, scipy
See requirements.txt for complete list with versions.

🔧 Configuration
Config Files
config.yaml: Main configuration (bucket name, file name, train ratio)
paths_config.py: File paths for data and models
model_params.py: LightGBM and hyperparameter tuning settings
Environment Variables
Create a .env file:
```bash
GOOGLE_APPLICATION_CREDENTIALS=<path-to-gcs-credentials>
```

📝 Logging
Comprehensive logging throughout the pipeline:

Logs saved in logs directory
Debug and error tracking
Pipeline execution monitoring
🤝 Contributing
Contributions are welcome! Please:

Create a feature branch
Make your changes
Test thoroughly
Submit a pull request
📧 Contact
For questions or support, contact: rishabhanand0200@gmail.com

📄 License
This project is licensed under the MIT License.

Last Updated: January 2026