import pandas as pd
import requests

df = pd.read_csv('data/creditcard.csv')


legit_samples = df[df['Class'] == 0].sample(5, random_state=1)
fraud_samples = df[df['Class'] == 1].sample(2, random_state=1)

samples = pd.concat([legit_samples, fraud_samples])

for _, row in samples.iterrows():
    payload = row.drop('Class').to_dict()
    resp = requests.post("http://127.0.0.1:8000/predict", json=payload)
    result = resp.json()
    actual = "FRAUD" if row['Class'] == 1 else "legit"
    print(f"Actual: {actual:6} | Predicted prob: {result['fraud_probability']:.4f} | "
          f"is_fraud: {result['is_fraud']} | latency: {result['latency_ms']}ms")