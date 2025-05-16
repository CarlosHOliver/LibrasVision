import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Caminho para o dataset
input_dir = 'data/raw/libras_dataset/'  # Caminho onde o dataset foi descompactado
output_dir = 'data/processed/libras_dataset/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Função para carregar e pré-processar as imagens
def load_and_process_data():
    images = []
    labels = []
    label_dict = {}  # Dicionário para mapear rótulos
    label_counter = 0

    # Loop para carregar as imagens de cada classe
    for label_folder in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label_folder)
        
        if os.path.isdir(label_path):
            if label_folder not in label_dict:
                label_dict[label_folder] = label_counter
                label_counter += 1

            # Loop para carregar cada imagem dentro da pasta da classe
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # Redimensionar para 64x64
                img_resized = cv2.resize(img, (64, 64)) / 255.0  # Normalização de pixels

                # Adicionar imagem e rótulo à lista
                images.append(img_resized)
                labels.append(label_dict[label_folder])

    # Converter listas em arrays numpy
    images = np.array(images)
    labels = np.array(labels)

    # Adicionar uma dimensão para as imagens (necessário para a CNN)
    images = np.expand_dims(images, axis=-1)

    return images, labels, label_dict

# Carregar e processar os dados
X, y, label_dict = load_and_process_data()

# Dividir os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Salvar os dados processados em formato .npy
np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

# Salvar o dicionário de labels
np.save(os.path.join(output_dir, 'label_dict.npy'), label_dict)

print("Pré-processamento concluído!")
