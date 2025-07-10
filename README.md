Project Structure

emotion-detection/
│
├── data/
│
├── models/
│   ├── emotion_model.pkl
│   └── vectorizer.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── emotion_detect.py
├── requirements.txt
└── README.md

Installation & Setup

1. Clone the repository

2. Install dependencies
   pip install -r requirements.txt

3. Train the model
   python emotion_detect.py

4. Run the Flask app
   python app.py
   Open browser at http://127.0.0.1:5000
