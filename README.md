# Twitter Sentiment Analysis using Machine Learning

This project focuses on performing **sentiment analysis** on Twitter data using **machine learning techniques**. The dataset used is sourced from **Kaggle**, and the entire project is developed in **Google Colab** for easy collaboration and cloud-based processing.

## Features

- **Sentiment Classification**: Classifies tweets into positive or negative.
- **Machine Learning Models**: Uses various ML algorithms like Logistic Regression, Random Forest, and Naive Bayes for classification.
- **Data Preprocessing**: Includes data cleaning, tokenization, and vectorization (TF-IDF/CountVectorizer) of tweet text.
  

## Dataset

- **Source**: [Kaggle Twitter Sentiment Analysis Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- The dataset contains labeled tweets with corresponding sentiment labels (positive and negative).

## Technologies Used

- **Python**: Core programming language.
- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation and analysis.
- **re (Regular Expressions)**: For text cleaning and preprocessing.
- **NLTK (Natural Language Toolkit)**: For stopword removal and stemming.
- **Scikit-learn**:
  - **TfidfVectorizer**: For text vectorization.
  - **Train-Test Split**: To split data for training and testing.
  - **Logistic Regression**: For machine learning model implementation.
  - **Accuracy Score**: For model performance evaluation.

This reflects the libraries and tools directly used in your code snippet.


### Prerequisites

Ensure you have the following installed before running the project:

- Python 3.x
- Google Colab (no local installation required)

### Dataset

1. **Download the Kaggle Dataset**:
   - You can download the dataset directly using the Kaggle API.
   - First, upload your **Kaggle API token** (`kaggle.json`) to Google Colab:
     ```python
     from google.colab import files
     files.upload()  # Upload kaggle.json
     ```

2. **Configure Kaggle API**:
   ```python
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Download the Dataset**:
   - Use the Kaggle API to download the dataset:
     ```python
     !kaggle datasets download -d <dataset-identifier>
     ```

4. **Unzip the Dataset**:
   ```python
   !unzip <dataset.zip>
   ```

5. **Load the Dataset into a DataFrame**:
   ```python
   import pandas as pd
   df = pd.read_csv('<dataset.csv>')
   ```
This setup now reflects using the Kaggle API to access the dataset directly from Google Colab.


## Usage

1. **Data Preprocessing**: 
   - Remove special characters, URLs, and mentions.
   - Tokenize the text and apply vectorization.

2. **Model Training**:
   - Train different machine learning models like Logistic Regression, Random Forest, or Naive Bayes on the processed data.
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   ```

3. **Model Evaluation**:
   - Evaluate the model using accuracy, precision, recall, and F1-score.
   ```python
   from sklearn.metrics import accuracy_score
   accuracy = accuracy_score(y_test, y_pred)
   ```

## Screenshots

![Screenshot 2024-09-07 211944](https://github.com/user-attachments/assets/844fda0b-ebe3-4af0-8ea3-e6d4b06bd7c4)    ![Screenshot 2024-09-07 212045](https://github.com/user-attachments/assets/b310ae0a-4f56-48e8-900e-426963eb065b)


## Future Improvements

- Experiment with advanced models like **BERT** or **LSTM** for improved accuracy.
- Add more detailed sentiment classifications like **very positive** or **very negative**.
- Implement real-time sentiment analysis on live Twitter data.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests for new features or improvements.


