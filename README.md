# SoundScape Generator

This repository contains experiments related to the generation and analysis of **ecological soundscapes**.
Most of the files in this project correspond to **tests, experiments, and learning**, while the **main functional script** is:

```
scripts/dataset_final.py

```

This script implements a **complete pipeline for generating synthetic acoustic scenes** using data from the ESC-50 dataset and automatic validation with **PANNs (Pretrained Audio Neural Networks)**.

---

# Objective

The objective of the script is to generate **labeled synthetic soundscapes**, combining three main sound categories:

* **Biophony** → biological sounds (animals, insects, etc.)
* **Anthropophony** → human or urban sounds
* **Geophony** → non-biological natural sounds (wind, rain, water)

These scenes are used to **create datasets for ecological soundscape classification**.

---

# Script Pipeline

The process implemented in `dataset_final.py` follows these steps:

1. **Metadata Loading**
* Reads `.csv` files containing the ESC-50 dataset labels.
* Associates each sound class with one of the three phony categories.


2. **Class Preparation**
* Mapping of the 527 classes detected by the PANNs model to:
* Biophony
* Anthropophony
* Geophony




3. **Synthetic Acoustic Scene Generation**

The script generates different types of scenes:

### Pure Scenes

A single dominant phony type.

Example:

```
Dominant Biophony + light noise from other classes

```

Typical distribution:

```
60% – 90% dominant class
10% – 40% noise from other classes

```

---

### Hybrid Scenes

Mixture of two dominant categories.

Example:

```
45% Biophony
45% Geophony
10% Anthropophony

```

Generated combinations:

* Biophony + Anthropophony
* Biophony + Geophony
* Anthropophony + Geophony

---

4. **Automatic Validation with PANNs**

Each generated scene is analyzed by a **Sound Event Detection (PANNs)** model.

The model verifies if the scene actually contains the expected patterns.

If the validation fails:

```
file is discarded
new scene is generated

```

This ensures **quality in the final dataset**.

---

5. **Feature Extraction**

The PANNs model generates **temporal features for 527 sound classes**.

These features are converted into **3 main channels**:

```
Biophony
Anthropophony
Geophony

```

The final result has the format:

```
[time, 3 channels]

```

---

6. **Dataset Export**

The script generates three types of output:

### Generated Audio

```
synthetic_scenes_real/audio/

```

`.wav` files containing the synthetic scenes.

---

### Metadata

```
synthetic_scenes_real/meta.csv

```

Contains:

```
filename
scene_label

```

---

### Training Features

```
synthetic_scenes_real/master_features_all.npz

```

This file contains:

```
X → extracted features
y → scene labels
filenames → file identification

```

---

# Expected Directory Structure

To run the script, the project must have a structure similar to:

```
project/
│
├── scripts
│   └── dataset_final.py
│
├── data
│   ├── ESC-50
│   │   ├── audio files
│   │   └── esc50.csv
│   │
│   └── esc50_label.csv
│
└── synthetic_scenes_real

```

---

# Dependencies

Main libraries used:

```
librosa
numpy
pandas
torch
soundfile
tqdm
panns-inference

```

Recommended installation:

```
pip install librosa numpy pandas torch soundfile tqdm panns-inference

```

---

# Execution

To generate the synthetic dataset:

```
python scripts/dataset_final.py

```

The script will:

1. Load ESC-50 data
2. Generate synthetic scenes
3. Validate with PANNs
4. Extract features
5. Save the final dataset

---

# Note

This repository contains several additional files related to:

* experiments
* tests
* notebooks
* auxiliary scripts

These files **are not part of the main pipeline** and have been kept solely for study and development purposes.

The **only consolidated script in the project currently is**:

```
scripts/dataset_final.py

```

---

# License

Academic and experimental use.