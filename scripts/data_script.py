import os
import pandas as pd
import json

ESC50_METADATA_PATH = "esc50.csv"
DATA_LABEL = "labels.csv"
ESC50_AUDIO_PATH = "audio"
MANUAL_MAP_PATH = "manual_class_map.json"
ESC50_AUDIO_PATH = "../data/ESC-50"
ESC50_METADATA_PATH = "../data/ESC-50/esc50.csv"
OUTPUT_AUDIO_DIR = "../synthetic_scenes_real/audio"
OUTPUT_METADATA_FILE = "../synthetic_scenes_real/meta.csv"
OUTPUT_FEATURES_FILE = "../synthetic_scenes_real/master_features.npz"
DATA_LABEL = "../data/label.csv"

def normalize_name(name: str) -> str:
    return name.replace("_", " ").title()

def load_manual_map():
    if os.path.exists(MANUAL_MAP_PATH):
        with open(MANUAL_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_manual_map(m):
    with open(MANUAL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":

    df_esc50 = pd.read_csv(ESC50_METADATA_PATH)
    df_label = pd.read_csv(DATA_LABEL)

    # Mapa automático
    label_map = {
        row["Som"].strip().lower(): row["Classe"]
        for _, row in df_label.iterrows()
    }

    # Mapa manual (persistente)
    manual_map = load_manual_map()

    rows = []

    for _, r in df_esc50.iterrows():
        audio_path = os.path.join(ESC50_AUDIO_PATH, r["filename"]).replace("\\", "/")

        esc_category = r["category"]
        normalized = normalize_name(esc_category)
        key = normalized.lower()

        classe = label_map.get(key)

        if classe is None:
            classe = manual_map.get(key)

        if classe is None:
            print(f"\nCategoria desconhecida: '{normalized}'")
            print("Escolha a classe:")
            print("[1] Biofonia")
            print("[2] Antropofonia")
            print("[3] Geofonia")
            print("[Enter] Ignorar")

            choice = input(">> ").strip()

            if choice == "1":
                classe = "Biofonia"
            elif choice == "2":
                classe = "Antropofonia"
            elif choice == "3":
                classe = "Geofonia"
            else:
                classe = "Desconhecido"

            # memoriza se não for desconhecido
            if classe != "Desconhecido":
                manual_map[key] = classe
                save_manual_map(manual_map)

        rows.append({
            "arquivo": r["filename"],
            "categoria_esc50": esc_category,
            "categoria_normalizada": normalized,
            "classe": classe,
            "caminho_audio": audio_path
        })

    df_final = pd.DataFrame(rows)
    df_final.to_csv("esc50_soundscape_map.csv", index=False)

    print("\n✔ Tabela final salva com sucesso!")
