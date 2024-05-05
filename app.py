from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
import pickle

app = Flask(__name__)

# Définissez les paramètres de votre modèle
vocab_size = 10000
embed_dim = 512
num_heads = 4
ff_dim = 512
maxlen = 100
num_classes = 8

# Charger le tokenizer depuis un fichier (à placer au début du fichier)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Charger le modèle transformateur entraîné (à la place du modèle SavedModel)
model = tf.keras.models.load_model("model")

# Fonction pour nettoyer le texte
def clean_text(text):
    text = re.sub(r'https?://\S+', '', text)   # Suppression des URLs
    text = re.sub(r'#\w+', '', text)           # Suppression des hashtags
    text = re.sub(r'@\w+', '', text)           # Suppression des mentions
    text = re.sub(r'\d+', '', text)            # Suppression des nombres
    text = re.sub(r'[^\w\s]', '', text)        # Suppression des caractères spéciaux
    return text.lower()

# Route pour la page d'accueil
@app.route('/')
def index():
    return render_template('home.html')

# Fonction pour prédire l'émotion à partir du texte
def predict_emotion_from_text(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    sequence_padded = pad_sequences(sequence, maxlen=maxlen)
    prediction = model.predict(sequence_padded)[0]
    return prediction

# Route pour la prédiction de l'émotion à partir de la phrase
@app.route('/traitement_texte', methods=['POST'])
def traitement_texte():
    # Récupérer le texte entré par l'utilisateur depuis le formulaire
    texte = request.form['texte']
    
    # Vérifier que le texte n'est pas vide
    if texte.strip() == "":
        return jsonify({'error': 'Le texte est vide'})
    
    # Prédiction de l'émotion
    try:
        resultat_emotion = predict_emotion_from_text(texte)
        # Mapper les indices de classe à des émotions
        emotions = {
            0: "Sadness",
            1: "Joy",
            2: "Love",
            3: "Anger",
            4: "Fear",
            5: "Surprise",
            6: "Neutral",
            7: "Contempt",
            8: "Disgust"
        }
        # Trouver l'indice de la classe avec la plus grande probabilité
        indice_emotion = resultat_emotion.argmax()
        # Récupérer l'émotion correspondante
        emotion_predite = emotions.get(indice_emotion, "Unknown")
        return jsonify({'emotion_prediction': emotion_predite})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
