import urllib.request, json

base = "http://localhost:8000"

# 1. Health
with urllib.request.urlopen(f"{base}/health") as r:
    print("HEALTH:", json.loads(r.read()))

# 2. Metrics
with urllib.request.urlopen(f"{base}/api/metrics") as r:
    m = json.loads(r.read())
    print("XGB accuracy:", m["xgboost"]["accuracy"])
    print("LR  accuracy:", m["logistic_regression"]["accuracy"])

# 3. Predict FAKE
fake_payload = json.dumps({
    "job_text": "Work from home data entry job! Earn 5000 dollars per week typing. No experience needed. Training provided. Send your bank details to apply immediately. Unlimited earning potential!",
    "model_name": "xgboost"
}).encode()
req = urllib.request.Request(f"{base}/api/predict", data=fake_payload,
                             headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req) as r:
    p = json.loads(r.read())
    print("PREDICT (fake text):", p["prediction"], "| conf:", round(p["confidence"]*100,1), "%", "| risk:", p["risk_level"])
    print("  Keywords found:", p["suspicious_keywords"])

# 4. Predict REAL
real_payload = json.dumps({
    "job_text": "Senior Software Engineer at Google. We are looking for experienced engineers with 5+ years of Python, distributed systems, and cloud infrastructure. Competitive salary, health benefits, and equity. Apply with resume and GitHub.",
    "model_name": "xgboost"
}).encode()
req2 = urllib.request.Request(f"{base}/api/predict", data=real_payload,
                              headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req2) as r:
    p2 = json.loads(r.read())
    print("PREDICT (real text):", p2["prediction"], "| conf:", round(p2["confidence"]*100,1), "%")

# 5. Stats
with urllib.request.urlopen(f"{base}/api/stats") as r:
    s = json.loads(r.read())
    print("STATS:", s["total_predictions"], "total |", s["fake_count"], "fake |", s["real_count"], "real")

# 6. History
with urllib.request.urlopen(f"{base}/api/history") as r:
    h = json.loads(r.read())
    print("HISTORY:", len(h), "records returned")

print("\n=== ALL ENDPOINTS PASSED ===")
