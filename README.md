# ECG Classification with Hybrid CNN-LSTM

![ECG Classification](https://img.shields.io/badge/status-completed-green)

This project is an ECG (Electrocardiogram) signal classification system that detects **normal** vs **abnormal** heart signals using a hybrid CNN-LSTM deep learning model. The project includes:

- Data preprocessing and scaling
- Training/testing with a deep learning model
- A Flask web application for interactive ECG testing
- Visualization of ECG signals and classification results
- Generation of PDF reports and confusion matrices

---

## Project Structure

/data # ECG datasets (normal and abnormal CSV files)
/saved_models # Saved trained model and scaler files
/templates # Flask HTML templates (index.html, test.html, test_result.html)
app.py # Flask application main script
README.md # This documentation file

yaml
Copy
Edit

---

## Features

- Load and preprocess ECG datasets (187 sample points per record)
- Scale features using StandardScaler
- Use a pretrained Hybrid CNN-LSTM model for prediction
- Flask web app with UI to test samples and visualize results
- Detailed classification reports and confusion matrices
- Save reports and plots as PDF files

---

## Setup Instructions

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Install dependencies

Make sure you have Python 3.7+ installed. Then install packages:

bash
Copy
Edit
pip install -r requirements.txt
(Example requirements.txt should include: tensorflow, pandas, numpy, scikit-learn, flask, matplotlib, seaborn, joblib)

Prepare your data

Place the ECG CSV files (ptbdb_normal.csv and ptbdb_abnormal.csv) inside the /data folder.

Place your saved model and scaler

Put your saved model file (hybrid_cnn_lstm_final.keras) and scaler (scaler.save) inside /saved_models.

Run the Flask web app

bash
Copy
Edit
python app.py
Open your browser and go to:

cpp
Copy
Edit
http://127.0.0.1:5000/
You can test ECG samples from the dataset and view classification results with confidence scores and plots.

Usage Example
You can run batch testing or test a single ECG signal through the Flask UI.

Example single ECG sample format:

mathematica
Copy
Edit
1.00E+00 6.07E-01 3.84E-01 ... (187 float values) ... 1.00E+00
Results
Achieved 99.4% accuracy on test data.

Generated detailed classification reports and confusion matrix plots.

Visualized ECG signals colored by prediction correctness.

Contact
For questions or collaboration:

Email: your.email@example.com

GitHub: https://github.com/yourusername

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
ECG dataset from PhysioNet PTB Diagnostic ECG Database

TensorFlow Keras for deep learning framework

Flask for web app development

yaml
Copy
Edit
