from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the model
loaded_model = pickle.load(open('model.pkl', 'rb'))

# Load the dataset
dataframe = pd.read_csv('news.csv')

# Train/Test split
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Initialize TfidfVectorizer directly here
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
tfvect.fit(x_train)  # Fit the vectorizer on training data

def fake_news_det(news):
    # Vectorizing the input text
    vectorized_input_data = tfvect.transform([news])
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction[0]  # Return the first (and only) prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)
