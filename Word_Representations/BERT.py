import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# --- 1. CONFIGURACIÓN ---

# Ruta donde están los TEXTOS segmentados y organizados por splits
INPUT_TEXT_PATH = '/home/invitadogita/Documentos/Trabajo de grado/Fase 2 - Enfoques unimodales/2. Análisis 10 Palabras (dataset)/Texto'

# Ruta BASE donde se guardarán las nuevas carpetas de representaciones
OUTPUT_BASE_PATH = '/home/invitadogita/Documentos/Trabajo de grado/Fase 2 - Enfoques unimodales/3. Representaciones de los Segmentos/Texto_Profundo'

# Diccionario de los modelos que vamos a procesar
# La clave es el nombre de la carpeta de salida, el valor es el modelo en Hugging Face
MODELS_TO_PROCESS = {
    "Bert_Base": "bert-base-uncased",
    "Bert_Emociones": "bhadresh-savani/bert-base-uncased-emotion"
}

# --- 2. FUNCIÓN PRINCIPAL DE EXTRACCIÓN ---

def extract_features_for_model(model_hf_name, output_folder_name):
    """
    Extrae representaciones de BERT para todos los textos y los guarda en
    la carpeta de salida especificada.
    """
    print("\n" + "="*60)
    print(f" INICIANDO EXTRACCIÓN PARA EL MODELO: {model_hf_name}")
    print("="*60)

    # --- Cargar el modelo y tokenizador específicos ---
    print("Cargando modelo y tokenizador desde Hugging Face... (puede tardar)")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_hf_name)
        model = BertModel.from_pretrained(model_hf_name)
        # Obtenemos la dimensión de salida del modelo
        output_dim = model.config.hidden_size
        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"¡ERROR CRÍTICO! No se pudo cargar el modelo '{model_hf_name}'. Verifica el nombre y tu conexión a internet.")
        print(f"Detalle: {e}")
        return

    # --- Buscar todos los archivos de texto ---
    text_files = glob.glob(os.path.join(INPUT_TEXT_PATH, '**', '*.txt'), recursive=True)
    if not text_files:
        print(f"¡ADVERTENCIA! No se encontraron archivos .txt en {INPUT_TEXT_PATH}")
        return

    print(f"Se encontraron {len(text_files)} segmentos de texto. Iniciando extracción...")

    # --- Procesar cada archivo de texto ---
    for txt_path in tqdm(text_files, desc=f"Procesando con {output_folder_name}"):
        try:
            # Leer el contenido del archivo de texto
            with open(txt_path, 'r', encoding='utf-8') as f:
                segment_text = f.read()

            # Si el texto está vacío, guardar un vector de ceros
            if not segment_text.strip():
                embedding = np.zeros(output_dim)
            else:
                # Tokenizar el texto y prepararlo para el modelo
                inputs = tokenizer(segment_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                
                # Extraer la representación
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Usamos el embedding del token [CLS] (el primero) como representación de toda la oración
                embedding = outputs.last_hidden_state[0, 0, :].numpy()

            # --- Guardar la salida en un archivo .csv ---
            relative_path = os.path.relpath(txt_path, INPUT_TEXT_PATH)
            csv_relative_path = os.path.splitext(relative_path)[0] + '.csv'
            output_csv_path = os.path.join(OUTPUT_BASE_PATH, output_folder_name, csv_relative_path)
            
            output_dir = os.path.dirname(output_csv_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Crear y guardar el DataFrame
            feature_names = [f'feature_{i}' for i in range(embedding.shape[0])]
            df_output = pd.DataFrame([embedding], columns=feature_names)
            df_output.to_csv(output_csv_path, index=False)

        except Exception as e:
            print(f"\n  -> Error procesando el archivo {txt_path}: {e}")

    print(f"\nExtracción para '{output_folder_name}' completada. ✅")


# --- 3. ORQUESTADOR PRINCIPAL ---

if __name__ == "__main__":
    # Iterar sobre el diccionario de modelos y ejecutar la extracción para cada uno
    for folder_name, model_name in MODELS_TO_PROCESS.items():
        extract_features_for_model(model_name, folder_name)
    
    print("\n¡PROCESO FINALIZADO! Todas las representaciones de texto profundo han sido generadas.")
