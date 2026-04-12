## Criação de Uma Base de Dados e Treinamento da Rede YOLO.


---

![bairesDev](https://github.com/user-attachments/assets/3613b0ea-8e0e-45c6-9e16-4165b2efc812)

**Bootcamp BairesDev — Machine Learning Training | Ministrado pela DIO**

---

# 🎯 treinaRedeYolo — Base de Dados e Treinamento da Rede YOLO

![GitHub repo size](https://img.shields.io/github/repo-size/Santosdevbjj/treinaRedeYolo)
![GitHub contributors](https://img.shields.io/github/contributors/Santosdevbjj/treinaRedeYolo)
![GitHub last commit](https://img.shields.io/github/last-commit/Santosdevbjj/treinaRedeYolo)
![Python Version](https://img.shields.io/badge/python-3.10-blue)

> *A diferença entre um projeto e um portfólio é o contexto que você coloca nele.*

---

## 1. 🎯 Problema de Negócio

Modelos de detecção de objetos como o YOLO são amplamente usados em aplicações reais — inspeção de qualidade industrial, monitoramento de segurança, contagem de itens em prateleiras — mas **a maior barreira de entrada não é o algoritmo: é a preparação do dataset**.

Rotular imagens manualmente, converter anotações para o formato correto do YOLO, dividir o dataset de forma reprodutível e garantir que imagens e labels estejam sempre sincronizados são etapas que frequentemente consomem mais tempo que o próprio treinamento. Erros nessa pipeline — como imagens sem label correspondente ou divisão inconsistente — invalidam os resultados do modelo silenciosamente.

O desafio é construir uma **pipeline de preparação e treinamento reprodutível**, que automatize desde a conversão de anotações LabelMe até a divisão treino/validação, o treinamento com YOLOv8 e a avaliação com métricas padronizadas.

---

## 2. 🏢 Contexto

O projeto foi desenvolvido no **Bootcamp BairesDev — Machine Learning Training**, como aplicação prática de construção de dataset e treinamento de redes de detecção de objetos com YOLOv8 (Ultralytics).

A pipeline cobre quatro responsabilidades distintas, cada uma encapsulada em um módulo independente:

**Conversão de anotações:** O `scripts/labelme_to_yolo.py` transforma JSONs gerados pelo LabelMe em arquivos `.txt` no formato YOLO — cada linha contendo `class_id x_center y_center width height` normalizados por dimensão da imagem.

**Divisão do dataset:** O `scripts/split_dataset.py` divide imagens e labels em treino/validação com ratio configurável (padrão 80/20), seed fixo para reprodutibilidade e verificação de integridade — arquivos sem label correspondente são sinalizados e ignorados ao invés de causar erros silenciosos.

**Treinamento no Colab:** O notebook `colab/train_yolo_ultralytics_colab.ipynb` carrega o dataset via upload, treina com YOLOv8n pré-treinado (50 épocas, `imgsz=640`) e executa validação e inferência dentro do mesmo ambiente.

**Avaliação de métricas:** O módulo `src/metrics.py` calcula Acurácia, Precisão, Recall e F1-Score via scikit-learn, além de gerar matriz de confusão (seaborn heatmap) e relatório detalhado por classe.

---

## 3. 📐 Premissas da Análise

As seguintes premissas delimitam o escopo desta implementação:

- O formato de anotação de entrada é **LabelMe JSON** (polígonos ou bounding boxes). O `labelme_to_yolo.py` extrai o bounding box a partir dos pontos do polígono calculando `xmin/xmax/ymin/ymax` e convertendo para coordenadas normalizadas do YOLO.
- Labels que referenciam classes **não presentes em `configs/classes.txt`** são ignorados silenciosamente durante a conversão. Isso garante que anotações de classes fora do escopo não corrompam o dataset.
- A divisão treino/validação usa **`random.seed(42)` fixo** para garantir que a mesma execução do `split_dataset.py` produza sempre a mesma partição — requisito de reprodutibilidade científica.
- O `split_dataset.py` verifica a existência do arquivo `.txt` correspondente para cada imagem antes de copiar. Imagens sem label são **sinalizadas no console e ignoradas**, evitando que o YOLO receba imagens sem anotação.
- O notebook de treinamento usa **YOLOv8n** (nano) — o menor modelo da família YOLOv8 — adequado para o ambiente Colab gratuito. Modelos maiores (`yolov8s`, `yolov8m`) exigem GPU mais robusta.
- O módulo `src/metrics.py` usa as implementações do **scikit-learn** (não do próprio YOLO), pois o objetivo é padronizar a avaliação independentemente do framework de detecção.

---

## 4. 🛠️ Estratégia da Solução

A pipeline foi organizada em quatro etapas executáveis de forma sequencial e independente:

**Etapa 1 — Rotulagem e Conversão (`scripts/labelme_to_yolo.py`)**
O script lê todos os arquivos `.json` do diretório de entrada, extrai as coordenadas de cada shape anotado no LabelMe, calcula as coordenadas normalizadas do bounding box e grava arquivos `.txt` no formato YOLO. A lista de classes é lida de `configs/classes.txt`, garantindo que os IDs de classe sejam consistentes com `dataset/data.yaml`. O mapeamento é por índice de linha — `classe1` vira ID 0, `classe2` vira ID 1.

**Etapa 2 — Divisão do Dataset (`scripts/split_dataset.py`)**
O script varre `dataset/raw/images/` (`.jpg` e `.png`), embaralha com seed fixo, calcula o índice de corte pelo `split_ratio` e copia imagens e labels para as pastas de destino. A função `move_files` verifica a existência do `.txt` correspondente antes de cada cópia — integridade garantida por design.

**Etapa 3 — Treinamento no Colab (`colab/train_yolo_ultralytics_colab.ipynb`)**
O notebook instala `ultralytics`, faz upload do dataset compactado, descompacta, carrega `yolov8n.pt`, treina com `model.train(data="dataset/data.yaml", epochs=50, imgsz=640, batch=16)`, valida com `model.val()` e demonstra inferência em uma imagem de validação. O `dataset/data.yaml` configura os paths de treino/validação e o número de classes (`nc: 2`).

**Etapa 4 — Avaliação de Métricas (`notebooks/calculo_metricas.ipynb` + `src/metrics.py`)**
O notebook demonstra o uso completo do módulo de métricas com vetores `y_true` e `y_pred` simulados. Em produção, esses vetores viriam das predições do modelo. As quatro funções do módulo são chamadas em sequência: `calcular_metricas()` → `exibir_metricas()` → `plotar_matriz_confusao()` → `gerar_relatorio()`.

---

## 5. 💡 Insights Técnicos

**Por que extrair bounding box dos pontos do polígono LabelMe e não usar a anotação retangular diretamente?**
LabelMe permite anotar tanto com retângulos quanto com polígonos livres. O `labelme_to_yolo.py` usa `bbox_from_points` — que calcula `xmin/xmax/ymin/ymax` a partir de qualquer lista de pontos — tornando o conversor compatível com ambos os tipos de anotação sem tratamento especial para cada caso.

**Por que usar `shutil.copy` em vez de `shutil.move` no `split_dataset.py`?**
`copy` preserva os arquivos originais em `data/raw/`, permitindo reexecutar a divisão com diferentes `split_ratio` ou `seed` sem precisar restaurar o dataset. A decisão de manter os dados brutos intocáveis é uma prática de engenharia de dados que evita a perda irreversível de anotações.

**Por que o `data.yaml` usa paths relativos ao invés de absolutos?**
Paths absolutos fazem o `data.yaml` funcionar apenas na máquina onde foi criado. Com paths relativos (`./dataset/images/train`), o mesmo arquivo funciona tanto no Colab (após upload do zip) quanto localmente, sem edição manual.

**Por que scikit-learn para métricas e não as métricas nativas do YOLOv8?**
O YOLOv8 calcula métricas de detecção (mAP@0.5, mAP@0.5:0.95) que consideram IoU entre bounding boxes — apropriadas para avaliar o detector. O `src/metrics.py` com scikit-learn calcula métricas de classificação (Acurácia, Precisão, Recall, F1) sobre as predições de classe — úteis para avaliar separadamente a qualidade das classificações após a detecção. São camadas de avaliação complementares, não redundantes.

---

## 6. 📊 Resultados

O projeto entrega quatro artefatos funcionais e encadeáveis:

| Artefato | Entrega |
|---|---|
| `scripts/labelme_to_yolo.py` | Conversor LabelMe JSON → YOLO `.txt` para qualquer dataset |
| `scripts/split_dataset.py` | Divisão treino/val reprodutível com verificação de integridade |
| `colab/train_yolo_ultralytics_colab.ipynb` | Pipeline de treinamento YOLOv8n completa no Colab |
| `src/metrics.py` + `notebooks/calculo_metricas.ipynb` | Avaliação padronizada com 4 métricas + matriz de confusão + relatório por classe |

Com o dataset de demonstração (`nc=2`, `classes: ['classe1', 'classe2']`), a pipeline executa end-to-end em menos de 30 minutos no Colab gratuito — incluindo conversão, divisão, treinamento (50 épocas) e avaliação.

---

## 7. 🔭 Próximos Passos

- Adicionar **data augmentation** no pipeline de preparação com `albumentations` — rotação, flip horizontal, variações de brilho — antes da divisão treino/val.
- Implementar **validação cruzada** na divisão do dataset (`scripts/split_dataset.py`) com suporte a K-Fold.
- Estender o `colab/train_yolo_ultralytics_colab.ipynb` para comparar YOLOv8n vs YOLOv8s em termos de mAP × tempo de treinamento.
- Integrar as métricas do scikit-learn (`src/metrics.py`) diretamente ao callback de treinamento do YOLOv8, gerando relatório automático ao final de cada experimento.
- Publicar o modelo treinado no **Hugging Face Hub** e expor inferência via API REST com FastAPI.

---

## 💻 Requisitos

### Hardware

| Recurso | Mínimo | Recomendado |
|---|---|---|
| CPU | Dual-Core | Quad-Core |
| RAM | 4 GB | 8 GB |
| GPU | — | NVIDIA (Colab T4 gratuito) |
| Disco | 2 GB livres | 10 GB livres |

### Software

| Dependência | Uso |
|---|---|
| Python 3.10+ | Linguagem principal |
| numpy | Arrays e operações matemáticas |
| opencv-python | Leitura e manipulação de imagens |
| pandas | Manipulação tabular (análises exploratórias) |
| matplotlib / seaborn | Visualizações e matriz de confusão |
| scikit-learn | Métricas de classificação |
| ultralytics | YOLOv8 — treinamento e inferência |

---

## 📂 Estrutura do Projeto

```bash
treinaRedeYolo/
│── requirements.txt              # numpy, opencv, pandas, matplotlib, seaborn, sklearn
│── .gitignore                    # data/raw/, data/processed/, venv/, __pycache__/
│── README.md
│
├── configs/
│   └── classes.txt               # Uma classe por linha — define os IDs do YOLO
│
├── dataset/
│   └── data.yaml                 # Paths treino/val + nc=2 + names=['classe1','classe2']
│
├── colab/
│   └── train_yolo_ultralytics_colab.ipynb  # Upload zip → YOLOv8n 50 épocas → val → predict
│
├── notebooks/
│   └── calculo_metricas.ipynb    # split_dataset → calcular_metricas → plotar → relatorio
│
├── scripts/
│   ├── labelme_to_yolo.py        # LabelMe JSON → YOLO .txt (qualquer tipo de shape)
│   └── split_dataset.py          # 80/20 com seed=42 + verificação de integridade
│
└── src/
    └── metrics.py                # calcular_metricas, exibir, plotar_matriz, gerar_relatorio
```

---

## ⚙️ Como Executar

### 🔹 Google Colab (Treinamento)

1. Compacte seu dataset no formato:
```bash
zip -r dataset.zip dataset/
```
2. Abra `colab/train_yolo_ultralytics_colab.ipynb` no Colab.
3. Execute as células sequencialmente — upload, treino, validação e inferência.

### 🔹 Localmente (Pipeline Completa)

**1. Clonar o repositório:**

```bash
git clone https://github.com/Santosdevbjj/treinaRedeYolo.git
cd treinaRedeYolo
```

**2. Criar e ativar ambiente virtual:**

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
```

**3. Instalar dependências:**

```bash
pip install -r requirements.txt
pip install ultralytics        # para treinamento local
```

**4. Converter anotações LabelMe → YOLO:**

```bash
python scripts/labelme_to_yolo.py \
    --json_dir data/labelme_json \
    --out_labels_dir data/labels \
    --classes configs/classes.txt
```

**5. Dividir dataset em treino/validação:**

```bash
python scripts/split_dataset.py \
    --images_dir data/raw/images \
    --labels_dir data/raw/labels \
    --output_dir dataset \
    --split_ratio 0.8 \
    --seed 42
```

**6. Treinar localmente (requer GPU):**

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="dataset/data.yaml", epochs=50, imgsz=640, batch=16)
```

**7. Calcular métricas no notebook:**

```bash
jupyter notebook notebooks/calculo_metricas.ipynb
```

---

## 📖 Notebooks

| Notebook | Objetivo |
|---|---|
| `colab/train_yolo_ultralytics_colab.ipynb` | Pipeline completa de treinamento YOLOv8 no Colab |
| `notebooks/calculo_metricas.ipynb` | Divisão do dataset + avaliação com todas as métricas |

---

## 📌 Aprendizados

**1. Preparar o dataset é mais trabalhoso que treinar o modelo — e mais importante.** A maioria dos problemas de performance do YOLO não está na arquitetura ou nos hiperparâmetros: está em labels inconsistentes, imagens sem anotação ou divisão treino/val contaminada. Construir o `split_dataset.py` com verificação de integridade (`label_file.exists()`) antes de cada cópia foi uma decisão que previne horas de debugging.

**2. O conversor precisa ser agnóstico ao tipo de anotação.** LabelMe permite anotar com retângulos e polígonos no mesmo projeto. Implementar `bbox_from_points` — que calcula o bounding box a partir de qualquer lista de pontos — tornou o `labelme_to_yolo.py` compatível com ambos os casos sem ramificações no código. Aprendi que robustez a variações do dado de entrada é mais valiosa do que otimização para o caso feliz.

**3. Métricas de detecção e métricas de classificação respondem perguntas diferentes.** O YOLOv8 calcula mAP considerando a sobreposição espacial entre boxes (IoU). O `src/metrics.py` com scikit-learn avalia se a classe atribuída à detecção está correta. Um modelo pode ter mAP alto e F1 baixo — detecta bem onde o objeto está, mas erra qual objeto é. Ter as duas camadas de avaliação no mesmo repositório tornou isso explícito.

---

## 📜 Licença

Este projeto está sob a licença MIT. Sinta-se livre para usar, modificar e distribuir.

---

## 📬 Contato

[![Portfólio Sérgio Santos](https://img.shields.io/badge/Portfólio-Sérgio_Santos-111827?style=for-the-badge&logo=githubpages&logoColor=00eaff)](https://portfoliosantossergio.vercel.app)
[![LinkedIn Sérgio Santos](https://img.shields.io/badge/LinkedIn-Sérgio_Santos-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/santossergioluiz)

---

