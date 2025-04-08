from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
model = tf.keras.models.load_model("text_classification_model.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Labels
labels = {
    0: "Stress",
    1: "Depression",
    2: "Bipolar disorder",
    3: "Personality disorder",
    4: "Anxiety"
}

# Keyword-specific recommendations for each category
tailored_recommendations = {
    "Stress": {
        "sleep": "Try creating a consistent bedtime routine and limit screen time before bed.",
        "work": "Manage work-related stress by taking regular breaks and prioritizing tasks.",
        "default": "Try meditation, regular exercise, and ensure 7-8 hours of sleep."
    },
    "Depression": {
        "hopeless": "Talk to a mental health professional. Journaling your thoughts might help.",
        "alone": "Reach out to friends or join support groups to avoid isolation.",
        "default": "Reach out to a therapist, talk to loved ones, and keep a daily routine."
    },
    "Bipolar disorder": {
        "manic": "Track your mood swings and avoid stimulants. Professional help is critical.",
        "energy": "Channel your high energy into structured tasks and maintain sleep patterns.",
        "default": "Seek professional psychiatric help. Mood tracking and therapy are helpful."
    },
    "Personality disorder": {
        "identity": "Focus on mindfulness and grounding exercises to maintain self-awareness.",
        "relationships": "Cognitive therapy can help develop healthier relationship patterns.",
        "default": "CBT and consistent psychological support are crucial."
    },
    "Anxiety": {
        "panic": "Try deep breathing and grounding techniques during panic episodes.",
        "heartbeat": "Do box-breathing exercises to calm your heart rate and mind.",
        "default": "Practice mindfulness, avoid caffeine, and try breathing exercises."
    }
}

def get_tailored_advice(category, text):
    text_lower = text.lower()
    for keyword, response in tailored_recommendations[category].items():
        if keyword != "default" and keyword in text_lower:
            return response
    return tailored_recommendations[category]["default"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No input text provided"}), 400

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded)
    predicted_class = prediction.argmax(axis=1)[0]
    label = labels[predicted_class]

    # Tailored recommendation based on keyword analysis
    advice = get_tailored_advice(label, text)

    return jsonify({
        "category": label,
        "recommendation": advice
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is live"})


if __name__ == "__main__":
    app.run(debug=True)
