from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Criando a aplicação Flask
app = Flask(__name__)

# Carregar o modelo treinado
model = load_model('libras_model.h5')

# Função de pre-processamento da imagem
def preprocess_image(image):
    img = cv2.resize(image, (50, 50))  # Redimensionar para o tamanho correto
    img = np.expand_dims(img, axis=-1)  # Adicionar a dimensão do canal
    img = np.expand_dims(img, axis=0)  # Adicionar a dimensão do batch
    img = img / 255.0  # Normalizar a imagem
    return img

# Rota principal
@app.route('/')
def index():
    return render_template('index.html')

# Rota para fazer previsões
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo encontrado'}), 400

    file = request.files['file']
    
    # Verificar se o arquivo é uma imagem
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Ler a imagem do arquivo
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        # Pré-processamento da imagem
        img = preprocess_image(img)

        # Fazer a previsão
        prediction = model.predict(img)

        # Obter a classe prevista
        predicted_class = np.argmax(prediction)

        return jsonify({'predicted_class': int(predicted_class)})

    return jsonify({'error': 'Formato de imagem inválido'}), 400

# Rodar o servidor Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
