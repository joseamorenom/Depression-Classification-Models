import torch
import librosa
# --- CAMBIO 1: Importamos la herramienta correcta ---
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm

# --- 1. CONFIGURACIÓN ---

# Ruta donde están los AUDIOS segmentados y organizados por splits
INPUT_AUDIO_PATH = '/home/invitadogita/Documentos/Trabajo de grado/Fase 2 - Enfoques unimodales/2. Análisis 10 Palabras (dataset)/Audio'

# Ruta BASE donde se guardarán las nuevas carpetas de representaciones
OUTPUT_BASE_PATH = '/home/invitadogita/Documentos/Trabajo de grado/Fase 2 - Enfoques unimodales/3. Representaciones de los Segmentos'

# La pareja de modelos correcta
MODELS_TO_PROCESS = {
    "Wav2Vec_XLSR_Base": "facebook/wav2vec2-large-xlsr-53",
    "Wav2Vec_XLSR_Emociones": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
}

# --- 2. FUNCIÓN PRINCIPAL DE EXTRACCIÓN ---

def extract_features_for_model(model_hf_name, output_folder_name):
    """
    Extrae representaciones de Wav2Vec para todos los audios y los guarda en
    la carpeta de salida especificada, replicando la estructura de carpetas.
    """
    print("\n" + "="*70)
    print(f"== INICIANDO EXTRACCIÓN PARA EL MODELO: {model_hf_name}")
    print("="*70)

    print("Cargando modelo y extractor de características... (puede tardar)")
    try:
        # --- CAMBIO 2: Usamos Wav2Vec2FeatureExtractor en lugar de Wav2Vec2Processor ---
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_hf_name)
        model = Wav2Vec2Model.from_pretrained(model_hf_name)
        print("Modelo y extractor cargados exitosamente. ✅")
    except Exception as e:
        print(f"¡ERROR CRÍTICO! No se pudo cargar el modelo '{model_hf_name}'.")
        print(f"Detalle: {e}")
        return

    audio_files = glob.glob(os.path.join(INPUT_AUDIO_PATH, '**', '*.wav'), recursive=True)
    if not audio_files:
        print(f"¡ADVERTENCIA! No se encontraron archivos .wav en {INPUT_AUDIO_PATH}")
        return

    print(f"Se encontraron {len(audio_files)} segmentos de audio. Iniciando extracción...")

    for audio_path in tqdm(audio_files, desc=f"Procesando con {output_folder_name}"):
        try:
            speech_array, _ = librosa.load(audio_path, sr=16000)
            input_values = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
            
            with torch.no_grad():
                outputs = model(input_values)
                hidden_states = outputs.last_hidden_state
            
            embedding = torch.mean(hidden_states, dim=1).squeeze().numpy()

            relative_path = os.path.relpath(audio_path, INPUT_AUDIO_PATH)
            csv_relative_path = os.path.splitext(relative_path)[0] + '.csv'
            output_csv_path = os.path.join(OUTPUT_BASE_PATH, output_folder_name, csv_relative_path)
            
            output_dir = os.path.dirname(output_csv_path)
            os.makedirs(output_dir, exist_ok=True)
            
            feature_names = [f'feature_{i}' for i in range(embedding.shape[0])]
            df_output = pd.DataFrame([embedding], columns=feature_names)
            df_output.to_csv(output_csv_path, index=False)
        except Exception as e:
            print(f"\n  -> Error procesando el archivo {audio_path}: {e}")

    print(f"\nExtracción para '{output_folder_name}' completada. ✅")

# --- 3. ORQUESTADOR PRINCIPAL ---

if __name__ == "__main__":
    for folder_name, model_name in MODELS_TO_PROCESS.items():
        extract_features_for_model(model_name, folder_name)
    
    print("\n¡PROCESO FINALIZADO! Todas las representaciones de audio profundo han sido generadas.")
