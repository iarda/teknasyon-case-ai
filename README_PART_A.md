# SMS Spam Detection - Documentation


## 1. My Approach & Thought Process
I followed a structured approach to ensure the model's accuracy and efficiency.

### Step 1: Understanding the Data
- The dataset consists of two columns:
  - `Message`: The SMS text.
  - `Label`: Either `ham` (not spam) or `spam`.
- The data is **multilingual** (English & Turkish), which required some extra preprocessing steps.
- Around **66% of messages were ham** and **34% were spam**, meaning the dataset is slightly imbalanced but still usable without resampling.

### Step 2: Data Preprocessing
To prepare the data for machine learning, I applied the following steps:

1. **Text Cleaning**  
   - Converted all text to lowercase.
   - Removed punctuation, special characters, and numbers.
   - Tokenized the text into individual words.

2. **Stopword Removal**  
   - Removed common stopwords in both English and Turkish to improve model performance.

3. **Stemming**  
   - Applied **Porter Stemming** to reduce words to their root form (e.g., "running" → "run").
  
4. **Label Encoding**  
   - Converted categorical labels (`ham`, `spam`) into numeric values (`0` for ham, `1` for spam`).
  
5. **Train-Test Split**  
   - Split the dataset into **80% training data** and **20% test data**, ensuring balanced class distribution.

### Step 3: Feature Engineering
Since machine learning models require numerical input, I converted the text data into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**:
- This method assigns higher weights to words that appear frequently in a message but not across all messages.
- I limited the vocabulary size to **5000 most frequent words** to keep the model efficient.

### Step 4: Model Training
I chose **Support Vector Machine (SVM)** with a **linear kernel** for classification:
- **Why SVM?**  
  - Works well for high-dimensional text data.
  - Less prone to overfitting compared to other traditional ML models.
- **Training Process:**  
  - I trained the **SVM model** on the TF-IDF-transformed training data.
  - Made predictions on the test data.

### Step 5: Model Evaluation
To assess the model’s performance, I used:
- **Accuracy**: Measures overall correctness.
- **Precision**: Measures how many predicted spam messages were actually spam.
- **Recall**: Measures how many real spam messages were detected.
- **F1-Score**: Balances precision and recall.

#### Results:
| Metric  | Score  |
|---------|--------|
| **Accuracy** | 98%  |
| **Precision (Spam)** | 98%  |
| **Recall (Spam)** | 95%  |
| **F1-Score** | 97%  |

The results show that the model is highly effective at distinguishing spam from ham.

---

## 2. Assumptions & Challenges
- I assumed that the dataset was **correctly labeled**.
- Since the dataset was **multilingual**, I manually added Turkish stopwords for better preprocessing.
- I relied on **TF-IDF features**, assuming they were sufficient for the classification task.
- I opted for **SVM over deep learning** due to its efficiency on smaller datasets.

---

## 3. Possible Improvements
While the model performs well, there are areas that could be improved:
- **Experiment with deep learning models** like BERT or LSTMs for better contextual understanding.
- **Use Named Entity Recognition (NER)** to filter out non-informative words.
- **Hyperparameter tuning** to further optimize the SVM model.
- **Balance the dataset** using oversampling techniques if needed.

---

## 4. Conclusion
I successfully developed an **SMS Spam Classifier** that achieves:  
*TF-IDF-based feature extraction*
*SVM-based classification*
*98% accuracy* in detecting spam  

This approach is efficient and can be further improved with advanced techniques if needed.
