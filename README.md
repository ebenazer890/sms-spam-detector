SMS Spam Detector (Naive Bayes)

Simple, pure-Python Multinomial Naive Bayes classifier for SMS spam detection. Train on the provided sample dataset or your own SMS corpus.

Files:
- sms_spam_detector/model.py : Vectorizer and Multinomial Naive Bayes implementation
- train.py : Train model and save to disk
- predict.py : Load saved model and predict a single message
- data/sms_sample_20.csv : Small sample dataset for quick experiments
- smoke_test.py : trains and evaluates model on sample dataset
- webapp.py : small aesthetic web UI (http://localhost:8000)
- requirements.txt : minimal dependencies (none required)

How to run:
1. Train:
   python train.py --data data/sms_sample_20.csv --model model.pkl
2. Predict:
   python predict.py --model model.pkl --message "Free entry in 2 a weekly competition"
3. Web UI:
   python webapp.py  # open http://localhost:8000

This project requires Python 3.8+.

Deploying with Docker on Render (Option B)
1) Push this repo to GitHub.
2) Render → New → Web Service → pick the repo.
3) Choose "Docker" environment. Render will read Dockerfile.
4) If prompted, set Build Command: (leave empty, Dockerfile handles it). Start Command: python webapp.py
5) Ensure env var PORT is provided (render.yaml sets 8000; Render also auto-sets PORT). The app already reads PORT.
6) Deploy. You’ll get a base URL like https://your-service.onrender.com.

Frontend (Netlify) reminder
- Host only the static UI on Netlify; point all fetch URLs in the JS to your Render API base (e.g., https://your-service.onrender.com/predict).
- A ready-to-deploy static UI is in webui/index.html. Set API_BASE at the top of that file to your Render URL, then deploy the webui folder to Netlify (drag/drop or `netlify deploy --dir webui --prod`).
SMS Spam Detector (Naive Bayes)

Simple, pure-Python Multinomial Naive Bayes classifier for SMS spam detection. Train on the provided sample dataset or your own SMS corpus.

Files:
- sms_spam_detector/model.py : Vectorizer and Multinomial Naive Bayes implementation
- train.py : Train model and save to disk
- predict.py : Load saved model and predict a single message
- data/sms_sample.csv : Small sample dataset for quick experiments
- smoke_test.py : trains and evaluates model on sample dataset
- requirements.txt : minimal dependencies (none required)

How to run:
1. Train:
   python train.py --data data/sms_sample.csv --model model.pkl
2. Predict:
   python predict.py --model model.pkl --message "Free entry in 2 a weekly competition" 

This project requires Python 3.8+.
