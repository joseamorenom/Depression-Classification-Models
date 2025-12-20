import gensim.downloader as api
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# --- 1. CONFIGURACIÓN ---

# Nombre del modelo pre-entrenado en el repositorio de Gensim.
# El script lo descargará automáticamente.
MODEL_NAME = 'word2vec-google-news-300'

# Ruta donde están los TEXTOS segmentados y organizados por splits.
INPUT_TEXT_PATH = '/home/invitadogita/Documentos/Trabajo de grado/Fase 2 - Enfoques unimodales/2. Análisis 10 Palabras (dataset)/Texto'

# Ruta donde se guardará la nueva carpeta de representaciones de Word2Vec.
OUTPUT_FEATURES_PATH = '/home/invitadogita/Documentos/Trabajo de grado/Fase 2 - Enfoques unimodales/3. Representaciones de los Segmentos/Texto/Word2Vec'

# --- 2. CARGA DEL MODELO Y FUNCIONES ---

def load_word2vec_model(model_name):
    """
    Descarga (si es necesario) y carga el modelo Word2Vec usando gensim.downloader.
    """
    print(f"Cargando modelo Word2Vec: '{model_name}'...")
    print("Nota: La primera vez, esto descargará el modelo (aprox. 1.6 GB) y puede tardar.")
    try:
        model = api.load(model_name)
        print("Modelo cargado exitosamente. ✅")
        return model
    except Exception as e:
        print("\n" + "="*70)
        print("¡ERROR CRÍTICO! No se pudo descargar o cargar el modelo Word2Vec.")
        print("Por favor, verifica tu conexión a internet.")
        print(f"Detalle del error: {e}")
        print("="*70)
        return None

def text_to_vector(text, model):
    """Convierte una cadena de texto en un vector promedio usando Word2Vec."""
    words = text.split()
    # Obtenemos los vectores solo de las palabras que existen en el vocabulario del modelo.
    word_vectors = [model[word] for word in words if word in model.key_to_index]
    
    # Si el texto está vacío o no contiene palabras del vocabulario, devolvemos un vector de ceros.
    if not word_vectors:
        return np.zeros(model.vector_size)
        
    # Calculamos el promedio de todos los vectores de palabras.
    return np.mean(word_vectors, axis=0)

# --- 3. PROCESO PRINCIPAL DE EXTRACCIÓN ---

# Cargar el modelo una sola vez al inicio
w2v_model = load_word2vec_model(MODEL_NAME)
if w2v_model is None:
    exit() # Detener el script si no se pudo cargar el modelo

# Buscar todos los archivos .txt de los segmentos
text_files = glob.glob(os.path.join(INPUT_TEXT_PATH, '**', '*.txt'), recursive=True)
if not text_files:
    print(f"¡ADVERTENCIA! No se encontraron archivos .txt en {INPUT_TEXT_PATH}")
    exit()

print(f"\nSe encontraron {len(text_files)} segmentos de texto. Iniciando extracción de vectores...")

# Procesar cada archivo de texto
for txt_path in tqdm(text_files, desc="Procesando Textos con Word2Vec"):
    try:
        # Leer el contenido del archivo de texto
        with open(txt_path, 'r', encoding='utf-8') as f:
            segment_text = f.read()
            
        # Convertir el texto a su vector promedio
        vector = text_to_vector(segment_text, w2v_model)
        
        # --- Lógica para guardar la salida en la ruta correcta ---
        relative_path = os.path.relpath(txt_path, INPUT_TEXT_PATH)
        csv_relative_path = os.path.splitext(relative_path)[0] + '.csv'
        output_csv_path = os.path.join(OUTPUT_FEATURES_PATH, csv_relative_path)
        
        output_dir = os.path.dirname(output_csv_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear y guardar el DataFrame
        feature_names = [f'feature_{i}' for i in range(w2v_model.vector_size)]
        df_output = pd.DataFrame([vector], columns=feature_names)
        df_output.to_csv(output_csv_path, index=False)

    except Exception as e:
        print(f"\n  -> Error procesando el archivo {txt_path}: {e}")

print("\n¡PROCESO FINALIZADO! Todas las representaciones de Word2Vec han sido generadas. ✅")
