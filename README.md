# SoundScape Generator

Este repositório contém experimentos relacionados à geração e análise de **paisagens sonoras ecológicas (soundscapes)**.
Grande parte dos arquivos presentes no projeto corresponde a **testes, experimentos e aprendizado**, enquanto o **script principal funcional** é:

```
scripts/dataset_final.py
```

Esse script implementa um **pipeline completo para geração de cenas acústicas sintéticas** utilizando dados do dataset ESC-50 e validação automática com **PANNs (Pretrained Audio Neural Networks)**.

---

# Objetivo

O objetivo do script é gerar **paisagens sonoras sintéticas rotuladas**, combinando três categorias principais de som:

* **Biofonia** → sons biológicos (animais, insetos, etc.)
* **Antropofonia** → sons humanos ou urbanos
* **Geofonia** → sons naturais não biológicos (vento, chuva, água)

Essas cenas são utilizadas para **criação de datasets para classificação de soundscapes ecológicos**.

---

# Pipeline do Script

O processo implementado em `dataset_final.py` segue as seguintes etapas:

1. **Carregamento de metadados**

   * Lê arquivos `.csv` contendo os rótulos do dataset ESC-50.
   * Associa cada classe sonora a uma das três categorias de fonia.

2. **Preparação das classes**

   * Mapeamento das 527 classes detectadas pelo modelo PANNs para:

     * Biofonia
     * Antropofonia
     * Geofonia

3. **Geração de cenas acústicas sintéticas**

O script gera diferentes tipos de cenas:

### Cenas Puras

Um único tipo dominante de fonia.

Exemplo:

```
Biofonia dominante + ruído leve das outras classes
```

Distribuição típica:

```
60% – 90% classe dominante
10% – 40% ruído de outras classes
```

---

### Cenas Híbridas

Mistura de duas categorias dominantes.

Exemplo:

```
45% Biofonia
45% Geofonia
10% Antropofonia
```

Combinações geradas:

* Biofonia + Antropofonia
* Biofonia + Geofonia
* Antropofonia + Geofonia

---

4. **Validação automática com PANNs**

Cada cena gerada é analisada por um modelo de **Sound Event Detection (PANNs)**.

O modelo verifica se a cena realmente contém os padrões esperados.

Se a validação falhar:

```
arquivo é descartado
nova cena é gerada
```

Isso garante **qualidade no dataset final**.

---

5. **Extração de features**

O modelo PANNs gera **features temporais de 527 classes sonoras**.

Essas features são convertidas para **3 canais principais**:

```
Biofonia
Antropofonia
Geofonia
```

O resultado final possui o formato:

```
[tempo, 3 canais]
```

---

6. **Exportação do dataset**

O script gera três tipos de saída:

### Áudio gerado

```
synthetic_scenes_real/audio/
```

Arquivos `.wav` contendo as cenas sintéticas.

---

### Metadados

```
synthetic_scenes_real/meta.csv
```

Contém:

```
filename
scene_label
```

---

### Features para treinamento

```
synthetic_scenes_real/master_features_all.npz
```

Esse arquivo contém:

```
X → features extraídas
y → rótulos das cenas
filenames → identificação dos arquivos
```

---

# Estrutura mínima esperada

Para executar o script, o projeto deve possuir uma estrutura semelhante a:

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

# Dependências

Principais bibliotecas utilizadas:

```
librosa
numpy
pandas
torch
soundfile
tqdm
panns-inference
```

Instalação recomendada:

```
pip install librosa numpy pandas torch soundfile tqdm panns-inference
```

---

# Execução

Para gerar o dataset sintético:

```
python scripts/dataset_final.py
```

O script irá:

1. Carregar os dados do ESC-50
2. Gerar cenas sintéticas
3. Validar com PANNs
4. Extrair features
5. Salvar o dataset final

---

# Observação

Este repositório contém vários arquivos adicionais relacionados a:

* experimentos
* testes
* notebooks
* scripts auxiliares

Esses arquivos **não fazem parte do pipeline principal** e foram mantidos apenas para fins de estudo e desenvolvimento.

O **único script consolidado do projeto atualmente é**:

```
scripts/dataset_final.py
```

---

# Licença

Uso acadêmico e experimental.
