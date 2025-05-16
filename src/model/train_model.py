import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import cv2

# Carregar os dados (substitua pelos seus próprios caminhos para os dados)
X_train = np.load('data/processed/libras_dataset/X_train.npy')
X_val = np.load('data/processed/libras_dataset/X_val.npy')
y_train = np.load('data/processed/libras_dataset/y_train.npy')
y_val = np.load('data/processed/libras_dataset/y_val.npy')

# Definir o modelo (exemplo simples de CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(np.unique(y_train)), activation='softmax')  # Número de classes
])

# Compilar o modelo
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Salvar o modelo
model.save('libras_model.h5')

# Após o treinamento, você pode carregar o modelo e fazer previsões

# Carregar o modelo salvo
model = load_model('libras_model.h5')

# Função para fazer previsão com o modelo treinado
def predict_image(image_path):
    # Carregar a imagem para previsão
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Verifique se a imagem foi carregada corretamente
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem {image_path}")
        return None
    
    img = cv2.resize(img, (50, 50))  # Redimensionar para o tamanho correto
    img = np.expand_dims(img, axis=-1)  # Adicionar a dimensão do canal
    img = np.expand_dims(img, axis=0)  # Adicionar a dimensão do batch
    img = img / 255.0  # Normalizar a imagem
    
    # Fazer a previsão
    prediction = model.predict(img)

    # Obter a classe prevista
    predicted_class = np.argmax(prediction)
    return predicted_class

# Exemplo de uso da função de previsão (substitua pelo caminho da sua imagem)
img_path = 'data/raw/libras_dataset/Fold1/D/0.PNG'  # Substitua pelo caminho correto da imagem
predicted_class = predict_image(img_path)
if predicted_class is not None:
    print(f"A classe prevista para a imagem é: {predicted_class}")
