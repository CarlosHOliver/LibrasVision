import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Caminho para os diretórios de entrada e saída
input_dir = '/workspaces/LibrasVision/data/raw/libras_dataset/'
output_dir = '/workspaces/LibrasVision/data/processed/libras_dataset/'

# Verifique se o diretório de saída existe, caso contrário, crie-o
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Função para verificar se o arquivo é uma imagem válida
def is_valid_image(file_path):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.PNG']
    return any(file_path.lower().endswith(ext) for ext in valid_extensions)

# Função para carregar e processar as imagens
def load_and_process_data():
    data = []
    labels = []
    label_dict = {}  # Dicionário para mapear rótulos
    label_counter = 0

    # Loop para carregar as imagens de cada classe
    for label_folder in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label_folder)
        
        if os.path.isdir(label_path):
            print(f"Processando pasta: {label_folder}")
            # Mostrar os arquivos encontrados na pasta
            files_in_folder = os.listdir(label_path)
            print(f"Arquivos encontrados: {files_in_folder}")
            
            if label_folder not in label_dict:
                label_dict[label_folder] = label_counter
                label_counter += 1

            # Processar todas as imagens na pasta da classe
            for subfolder in files_in_folder:
                subfolder_path = os.path.join(label_path, subfolder)
                if os.path.isdir(subfolder_path):  # Verificar se é um diretório
                    image_files = os.listdir(subfolder_path)
                    for img_name in image_files:
                        img_path = os.path.join(subfolder_path, img_name)

                        # Verifique se o arquivo é uma imagem válida
                        if not is_valid_image(img_path):
                            continue  # Pular arquivos não-imagem

                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Leitura em escala de cinza

                        if img is None:
                            print(f"Erro ao carregar a imagem {img_path}. A imagem pode estar corrompida ou com caminho incorreto.")
                            continue  # Pular se a imagem não foi carregada corretamente

                        # Se necessário, redimensione a imagem para 50x50
                        img_resized = cv2.resize(img, (50, 50)) / 255.0  # Normalizando

                        # Adicionar imagem e rótulo à lista
                        data.append(img_resized)
                        labels.append(label_dict[label_folder])

    # Verificar se imagens foram carregadas corretamente
    if len(data) == 0:
        print("Erro: Nenhuma imagem válida foi carregada.")
        return None, None, None

    # Converter para numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Adicionar uma dimensão para as imagens (necessário para a CNN)
    data = np.expand_dims(data, axis=-1)  # Formato (n_samples, 50, 50, 1)

    return data, labels, label_dict

# Carregar e processar os dados
X, y, label_dict = load_and_process_data()

# Verifique se o conjunto de dados foi carregado corretamente
if X is None or y is None:
    print("Erro: Não foi possível carregar dados válidos.")
else:
    # Dividir os dados em treino e teste (80% treino, 20% teste)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Salvar os dados processados em formato .npy
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)

    # Salvar o dicionário de labels
    np.save(os.path.join(output_dir, 'label_dict.npy'), label_dict)

    print("Pré-processamento concluído!")
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Caminho para os diretórios de entrada e saída
input_dir = '/workspaces/LibrasVision/data/raw/libras_dataset/'
output_dir = '/workspaces/LibrasVision/data/processed/libras_dataset/'

# Verifique se o diretório de saída existe, caso contrário, crie-o
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Função para verificar se o arquivo é uma imagem válida
def is_valid_image(file_path):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.PNG']
    return any(file_path.lower().endswith(ext) for ext in valid_extensions)

# Função para carregar e processar as imagens
def load_and_process_data():
    data = []
    labels = []
    label_dict = {}  # Dicionário para mapear rótulos
    label_counter = 0

    # Loop para carregar as imagens de cada classe
    for label_folder in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label_folder)
        
        if os.path.isdir(label_path):
            print(f"Processando pasta: {label_folder}")
            # Mostrar os arquivos encontrados na pasta
            files_in_folder = os.listdir(label_path)
            print(f"Arquivos encontrados: {files_in_folder}")
            
            if label_folder not in label_dict:
                label_dict[label_folder] = label_counter
                label_counter += 1

            # Processar todas as imagens na pasta da classe
            for subfolder in files_in_folder:
                subfolder_path = os.path.join(label_path, subfolder)
                if os.path.isdir(subfolder_path):  # Verificar se é um diretório
                    image_files = os.listdir(subfolder_path)
                    for img_name in image_files:
                        img_path = os.path.join(subfolder_path, img_name)

                        # Verifique se o arquivo é uma imagem válida
                        if not is_valid_image(img_path):
                            continue  # Pular arquivos não-imagem

                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Leitura em escala de cinza

                        if img is None:
                            print(f"Erro ao carregar a imagem {img_path}. A imagem pode estar corrompida ou com caminho incorreto.")
                            continue  # Pular se a imagem não foi carregada corretamente

                        # Se necessário, redimensione a imagem para 50x50
                        img_resized = cv2.resize(img, (50, 50)) / 255.0  # Normalizando

                        # Adicionar imagem e rótulo à lista
                        data.append(img_resized)
                        labels.append(label_dict[label_folder])

    # Verificar se imagens foram carregadas corretamente
    if len(data) == 0:
        print("Erro: Nenhuma imagem válida foi carregada.")
        return None, None, None

    # Converter para numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Adicionar uma dimensão para as imagens (necessário para a CNN)
    data = np.expand_dims(data, axis=-1)  # Formato (n_samples, 50, 50, 1)

    return data, labels, label_dict

# Carregar e processar os dados
X, y, label_dict = load_and_process_data()

# Verifique se o conjunto de dados foi carregado corretamente
if X is None or y is None:
    print("Erro: Não foi possível carregar dados válidos.")
else:
    # Dividir os dados em treino e teste (80% treino, 20% teste)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Salvar os dados processados em formato .npy
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)

    # Salvar o dicionário de labels
    np.save(os.path.join(output_dir, 'label_dict.npy'), label_dict)

    print("Pré-processamento concluído!")
