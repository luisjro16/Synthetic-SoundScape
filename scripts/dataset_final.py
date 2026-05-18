import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os
import random
import torch
from tqdm import tqdm
from panns_inference import SoundEventDetection, labels

ESC50_AUDIO_PATH = "../data/ESC-50"
ESC50_METADATA_PATH = "../data/ESC-50/esc50.csv"
OUTPUT_AUDIO_DIR = "../synthetic_scenes_real/audio"
OUTPUT_METADATA_FILE = "../synthetic_scenes_real/meta.csv"
OUTPUT_FEATURES_FILE = "../synthetic_scenes_real/master_features.npz"
DATA_LABEL = "../data/label.csv"

N_PUROS_PER_CLASS = 1667   
N_HYBRID_PER_PAIR = 1667
SAMPLE_RATE = 32000
DURATION_SAMPLES = SAMPLE_RATE * 5

os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

bio_indices = []
antro_indices = []
geo_indices = []

dummy_df = pd.read_csv(DATA_LABEL)
mapa_classe = pd.Series(dummy_df.Classe.values, index=dummy_df.Som).to_dict()

for i, label_name in enumerate(labels):
    categoria = mapa_classe.get(label_name)

    if categoria == 'Biofonia':
        bio_indices.append(i)
    elif categoria == 'Antropofonia':
        antro_indices.append(i)
    elif categoria == 'Geofonia':
        geo_indices.append(i)

#  (VALIDAÇÃO E CONVERSÃO) 

def convert_to_3_channels(features_527, bio_idxs, antro_idxs, geo_idxs):
    """Converte a saída bruta do PANNs (527) para as 3 Fonias."""
    # .max(axis=1) pega o valor máximo daquele grupo de colunas para cada frame
    p_bio = features_527[:, bio_idxs].max(axis=1)
    p_antro = features_527[:, antro_idxs].max(axis=1)
    p_geo = features_527[:, geo_idxs].max(axis=1)
    return np.stack([p_bio, p_antro, p_geo], axis=1)

def validate_pure(features_527, target_idx, bio_idxs, antro_idxs, geo_idxs):
    """Valida se existe APENAS UM dominantes claro."""
    p_bio = features_527[:, bio_idxs].max()
    p_antro = features_527[:, antro_idxs].max()
    p_geo = features_527[:, geo_idxs].max()
    
    scores = [p_bio, p_antro, p_geo]
    winner = np.argmax(scores)
    
    if winner == target_idx and scores[winner] > 0.4:
        return True
    return False

def validate_hybrid(features_527, idx_A, idx_B, idx_Noise, bio_idxs, antro_idxs, geo_idxs):
    """Valida se existem DOIS dominantes claros."""
    p_bio = features_527[:, bio_idxs].max()
    p_antro = features_527[:, antro_idxs].max()
    p_geo = features_527[:, geo_idxs].max()
    
    scores = [p_bio, p_antro, p_geo]
    
    if (scores[idx_A] > scores[idx_Noise]) and \
       (scores[idx_B] > scores[idx_Noise]) and \
       (scores[idx_A] > 0.25) and (scores[idx_B] > 0.25):
        return True
    return False

def load_audio(p):
    try:
        s, _ = librosa.load(p, sr=SAMPLE_RATE, mono=True)
        if len(s) < DURATION_SAMPLES: s = np.pad(s, (0, DURATION_SAMPLES-len(s)))
        else: s = s[:DURATION_SAMPLES]
        
        # Normalização RMS para garantir que 10% de volume seja realmente 10% de energia
        rms = np.sqrt(np.mean(s**2)) + 1e-10
        return s * (0.1 / rms)
    except: return np.zeros(DURATION_SAMPLES)

def generate_clean_mix(target_class, file_map, output_filename):
    """
    Gera um mix onde a 'target_class' tem OBRIGATORIAMENTE entre 60% e 90% de presença.
    O restante é dividido entre ruído das outras classes.
    """
    final_signal = np.zeros(DURATION_SAMPLES)
    
    dominant_prop = random.uniform(0.6, 0.9)
    remaining_prop = 1.0 - dominant_prop
    
    noise_prop = remaining_prop / 2
    
    proportions = {'bio': noise_prop, 'antro': noise_prop, 'geo': noise_prop}
    proportions[target_class] = dominant_prop 
    
    for category in ['bio', 'antro', 'geo']:
        if proportions[category] > 0.01: 
            f = random.choice(file_map[category])
            s = load_audio(f)
            final_signal += s * proportions[category]
            

    max_val = np.max(np.abs(final_signal))
    if max_val > 1.0: final_signal /= max_val
    sf.write(output_filename, final_signal, SAMPLE_RATE)
    return proportions

def generate_hybrid_mix(class_A, class_B, file_map, output_filename):
    """
    Gera uma mistura híbrida: 45% A + 45% B + 10% Ruído C
    """
    final_signal = np.zeros(DURATION_SAMPLES)
    
    prop_A = random.uniform(0.40, 0.48)
    prop_B = random.uniform(0.40, 0.48)
    prop_rest = 1.0 - (prop_A + prop_B)
    
    all_classes = ['bio', 'antro', 'geo']
    class_rest = [c for c in all_classes if c != class_A and c != class_B][0]
    
    proportions = {class_A: prop_A, class_B: prop_B, class_rest: prop_rest}
    
    for category in all_classes:
        f = random.choice(file_map[category])
        s = load_audio(f)
        final_signal += s * proportions[category]
            
    max_val = np.max(np.abs(final_signal))
    if max_val > 1.0: final_signal /= max_val
    sf.write(output_filename, final_signal, SAMPLE_RATE)
    return proportions

def generate_all_mix(file_map, output_filename):
    """Gera uma mistura ~33% de cada classe."""
    final_signal = np.zeros(DURATION_SAMPLES)
    
    raw_props = np.random.dirichlet((10, 10, 10)) 
    
    proportions = {
        'bio': raw_props[0], 
        'antro': raw_props[1], 
        'geo': raw_props[2]
    }
    
    for category in ['bio', 'antro', 'geo']:
        f = random.choice(file_map[category])
        s = load_audio(f)
        final_signal += s * proportions[category]
            
    max_val = np.max(np.abs(final_signal))
    if max_val > 1.0: final_signal /= max_val
    sf.write(output_filename, final_signal, SAMPLE_RATE)
    return proportions

def validate_all(features_527, bio_idxs, antro_idxs, geo_idxs):
    """Valida se as TRÊS classes estão presentes."""
    p_bio = features_527[:, bio_idxs].max()
    p_antro = features_527[:, antro_idxs].max()
    p_geo = features_527[:, geo_idxs].max()
    
    if (p_bio > 0.15) and (p_antro > 0.15) and (p_geo > 0.15):
        return True
    return False

def run_master_factory(file_map):
    
    # Inicializa PANNs
    print("Iniciando Monitor PANNs...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    monitor = SoundEventDetection(checkpoint_path=None, device=device)
    
    metadata = []
    features_list = []
    filenames_list = []
    labels_list = [] 
    
    # --- DADOS PUROS 
    print("\n>>> Cenas Puras")
    pure_classes = {0: 'bio', 1: 'antro', 2: 'geo'}
    
    for label_code, name in pure_classes.items():
        print(f" -> Gerando {name.upper()}...")
        count = 0
        pbar = tqdm(total=N_PUROS_PER_CLASS)
        
        while count < N_PUROS_PER_CLASS:
            fname = f"pure_{name}_{count:04d}.wav"
            fpath = os.path.join(OUTPUT_AUDIO_DIR, fname)
            
            if not os.path.exists(fpath):
                generate_clean_mix(name, file_map, fpath)
            
            try:
                y_audio, _ = librosa.load(fpath, sr=32000, mono=True)
                tensor = torch.tensor(y_audio[None, :]).to(device)
                
                with torch.no_grad(): 
                    output = monitor.inference(tensor)
                
                # Verifica se é dict (padrão panns_inference)
                if isinstance(output, dict) and 'framewise_output' in output:
                    raw_array = output['framewise_output'] 
                else:
                    raw_array = output 
                
                # Remove dimensão do batch 
                feat_527 = raw_array[0] 
                
                if validate_pure(feat_527, label_code, bio_indices, antro_indices, geo_indices):
                    feat_3ch = convert_to_3_channels(feat_527, bio_indices, antro_indices, geo_indices)
                    features_list.append(feat_3ch)
                    filenames_list.append(fname)
                    labels_list.append(label_code)
                    
                    metadata.append({'filename': fname, 'scene_label': label_code})
                    count += 1
                    pbar.update(1)
                else:
                    os.remove(fpath) 
            except Exception as e: 
                if os.path.exists(fpath): os.remove(fpath)
        pbar.close()

    # DADOS HÍBRIDOS (3, 4, 5) 
    print("\n>>> Cenas Híbridas")
    hybrids = [
        ('bio', 'antro', 3, 0, 1, 2),
        ('bio', 'geo',   4, 0, 2, 1),
        ('antro', 'geo', 5, 1, 2, 0)
    ]
    
    for c1, c2, label, idxA, idxB, idxN in hybrids:
        print(f" -> Gerando {c1.upper()} + {c2.upper()}...")
        count = 0
        pbar = tqdm(total=N_HYBRID_PER_PAIR)
        
        while count < N_HYBRID_PER_PAIR:
            fname = f"hybrid_{c1}_{c2}_{count:04d}.wav"
            fpath = os.path.join(OUTPUT_AUDIO_DIR, fname)

            if not os.path.exists(fpath):
                generate_hybrid_mix(c1, c2, file_map, fpath)
            
            # Valida
            try:
                y_audio, _ = librosa.load(fpath, sr=32000, mono=True)
                tensor = torch.tensor(y_audio[None, :]).to(device)
                
                with torch.no_grad(): 
                    output = monitor.inference(tensor)
                
                if isinstance(output, dict) and 'framewise_output' in output:
                    raw_array = output['framewise_output']
                else:
                    raw_array = output
                
                feat_527 = raw_array[0] 
                
                if validate_hybrid(feat_527, idxA, idxB, idxN, bio_indices, antro_indices, geo_indices):
                    feat_3ch = convert_to_3_channels(feat_527, bio_indices, antro_indices, geo_indices)
                    features_list.append(feat_3ch)
                    filenames_list.append(fname)
                    labels_list.append(label)
                    
                    metadata.append({'filename': fname, 'scene_label': label})
                    count += 1
                    pbar.update(1)
                else:
                    os.remove(fpath)
            except Exception as e:
                if os.path.exists(fpath): os.remove(fpath)
        pbar.close()
        
    # DADOS COMPLETOS (6) 
    print("\n>>> Cena Completa")
    print("Pulando")
    # count = 0
    # pbar = tqdm(total=N_HYBRID_PER_PAIR)
        
    # while count < N_HYBRID_PER_PAIR:
    #     fname = f"complete_{count:04d}.wav"
    #     fpath = os.path.join(OUTPUT_AUDIO_DIR, fname)

    #     if not os.path.exists(fpath):
    #         generate_all_mix(file_map, fpath)
        
    #     # Valida
    #     try:
    #         y_audio, _ = librosa.load(fpath, sr=32000, mono=True)
    #         tensor = torch.tensor(y_audio[None, :]).to(device)
            
    #         with torch.no_grad(): 
    #             output = monitor.inference(tensor)
            
    #         if isinstance(output, dict) and 'framewise_output' in output:
    #             raw_array = output['framewise_output']
    #         else:
    #             raw_array = output
            
    #         feat_527 = raw_array[0] 
            
    #         if validate_all(feat_527, bio_indices, antro_indices, geo_indices):
    #             feat_3ch = convert_to_3_channels(feat_527, bio_indices, antro_indices, geo_indices)
    #             features_list.append(feat_3ch)
    #             filenames_list.append(fname)
    #             labels_list.append(6)
                
    #             metadata.append({'filename': fname, 'scene_label': 6})
    #             count += 1
    #             pbar.update(1)
    #         else:
    #             os.remove(fpath)
    #     except Exception as e:
    #         if os.path.exists(fpath): os.remove(fpath)
    pbar.close()


    pd.DataFrame(metadata).to_csv(OUTPUT_METADATA_FILE, index=False)
    
    X_padded = []
    for x in features_list:
        if len(x) < 501: x = np.pad(x, ((0, 501-len(x)), (0,0)), 'constant')
        else: x = x[:501]
        X_padded.append(x)
        
    np.savez_compressed(
        '../synthetic_scenes_real/master_features_all.npz', 
        X=np.array(X_padded, dtype=np.float32), 
        y=np.array(labels_list, dtype=np.int32),
        filenames=np.array(filenames_list)
    )
    print(f"Salvo em {OUTPUT_FEATURES_FILE}")
    

if __name__ == "__main__":

    df_final = pd.read_csv("../data/esc50_label.csv")

    fmap = {'bio': [], 'antro': [], 'geo': []}
    for i, r in df_final.iterrows():
        p = os.path.join(ESC50_AUDIO_PATH, r["arquivo"])
        p = p.replace('\\', '/')

        if r["classe"] == "Biofonia": fmap['bio'].append(p)
        elif r["classe"] == "Antropofonia": fmap['antro'].append(p)
        elif r["classe"] == "Geofonia": fmap['geo'].append(p)
    
    run_master_factory(fmap)