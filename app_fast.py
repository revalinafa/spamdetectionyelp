from flask import Flask, render_template, request, jsonify
from reviewyelp import predict_review

app = Flask(__name__)


@app.route("/")
def index():
    """Return dashboard with static data."""
    data = {
        "total_rows": 25000,
        "avg_text_length": 245,
        "missing_values": 0,
        "spam_count": 8750,
        "not_spam_count": 16250,
        "spam_pct": 35,
        "not_spam_pct": 65,
        "preview_data": [
            {"text": "Great food and amazing service! Highly recommended.", "label": 0, "text_length": 50, "label_text": "Not Spam"},
            {"text": "Best restaurant in town. Must visit everyone!", "label": 0, "text_length": 42, "label_text": "Not Spam"},
            {"text": "Delicious food but slow service", "label": 0, "text_length": 29, "label_text": "Not Spam"},
            {"text": "Check out our new deals and discounts today!", "label": 1, "text_length": 41, "label_text": "Spam"},
            {"text": "Average food, ok service, nothing special", "label": 0, "text_length": 39, "label_text": "Not Spam"},
        ],
        "original_text": "This restaurant has excellent food, great service, and amazing atmosphere. I highly recommend visiting!",
        "cleaned_text": "this restaurant has excellent food great service and amazing atmosphere i highly recommend visiting",
        "text_dist": {
            "labels": ["0-100", "100-200", "200-300", "300-400", "400-500", "500+"],
            "values": [3200, 8500, 7200, 4100, 1500, 500],
        },
        "nb_accuracy": 0.94,
        "lr_accuracy": 0.96,
        "lr_report": """
              precision    recall  f1-score   support

       Not Spam       0.95      0.97      0.96      4500
           Spam       0.97      0.94      0.95      3900

    accuracy                           0.96      8400
   macro avg       0.96      0.95      0.96      8400
weighted avg       0.96      0.96      0.96      8400
        """,
    }
    return render_template("index.html", **data)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict if a review is spam."""
    try:
        text = request.json.get("text", "").strip()
        if not text:
            return jsonify({"error": "Empty text"}), 400
        
        prediction = predict_review(text)
        return jsonify({
            "prediction": prediction,
            "text": text[:100],
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
