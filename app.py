from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
import pickle
import random
from PIL import Image
import os

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

# Chemins vers les répertoires de chaque émotion
emotion_folders = {
    "sadness": "sad",
    "joy": "happy",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "neutral": "neutral",
    "contempt": "contempt",
    "disgust": "disgust"
}

# Chemin vers le répertoire des images
IMAGES_DIR = "images/train"

# Fonction pour prédire l'émotion à partir du texte
def predict_emotion_from_text(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    sequence_padded = pad_sequences(sequence, maxlen=maxlen)
    prediction = model.predict(sequence_padded)[0]
    return prediction

# Fonction pour choisir une image au hasard dans un répertoire
def choose_random_image_from_folder(folder_path):
    images = os.listdir(folder_path)
    random_image = random.choice(images)
    return os.path.join(folder_path, random_image)

# Fonction pour afficher une image correspondant à une émotion
def display_image_for_emotion(emotion):
    # Chemin vers le répertoire contenant les images de cette émotion
    emotion_folder_path = os.path.join(IMAGES_DIR, emotion_folders.get(emotion, ""))
    
    if os.path.isdir(emotion_folder_path):
        # Choix d'une image au hasard dans le dossier de cette émotion
        image_path = choose_random_image_from_folder(emotion_folder_path)
        
        # Affichage de l'image
        image = Image.open(image_path)
        image.show()
        
        # Affichage de la prédiction
        print("Prédiction d'émotion:", emotion)
    else:
        print("Le chemin vers le répertoire de cette émotion n'est pas spécifié.")

@app.route('/')
def index():
    return render_template('home.html')

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
            0: "sadness",
            1: "joy",
            2: "anger",
            3: "fear",
            4: "surprise",
            5: "neutral",
            6: "contempt",
            7: "disgust"
        }
        # Trouver l'indice de la classe avec la plus grande probabilité
        indice_emotion = resultat_emotion.argmax()
        # Récupérer l'émotion correspondante
        emotion_predite = emotions.get(indice_emotion, "Unknown")
        # Choix de l'image correspondant à l'émotion
        display_image_for_emotion(emotion_predite)
    
        return jsonify({'emotion_prediction': emotion_predite})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)