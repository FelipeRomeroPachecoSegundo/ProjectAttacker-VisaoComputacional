# ProjectAttacker-VisaoComputacional

Este projeto implementa uma versão simplificada do ataque físico **ProjAttacker**, utilizando projeção luminosa simulada para enganar modelos de reconhecimento facial.  
O ataque gera uma perturbação projetável no rosto do atacante, fazendo com que o sistema o reconheça como outra pessoa (vítima).

O pipeline utiliza **MediaPipe FaceMesh**, **mapa de profundidade sintético**, **simulação de projeção de luz**, **simulação de câmera**, e um modelo pré-treinado de embeddings faciais (**InceptionResnetV1 – facenet-pytorch**).

---

## 📌 Funcionalidades

- Extração de landmarks faciais (MediaPipe FaceMesh)
- Construção de uma máscara pseudo-3D com mapa de profundidade
- Simulação diferenciável de projeção de luz (LRF simplificada)
- Simulação de captura por câmera (blur + ruído)
- Uso de modelo pré-treinado de reconhecimento facial
- Otimização iterativa da perturbação adversarial
- Geração de imagem final + timeline de evolução do ataque

---

## 🛠️ Pré-requisitos

- Python **3.12**
- As dependências estão no `requirements.txt`

Crie um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## 📷 Preparação das imagens

Você deve fornecer duas imagens:

- **Imagem do atacante** (quem tentará “virar” outra pessoa)
- **Imagem da vítima** (identidade alvo)

Recomendações:

- Rosto bem visível  
- De frente  
- Boa iluminação  
- Arquivos `.jpg` ou `.png`

Exemplo:

```
attacker_face.jpg
michael.jpeg
```

---

## ▶️ Executando o ataque

Use o comando abaixo:

```bash
python attack_mediapipe_mask.py \
  --attacker attacker_face.jpg \
  --victim michael.jpeg \
  --output output_obama.png \
  --steps 2000 \
  --frames 4 \
  --lr 0.05 \
  --reg-weight 0.01 \
  --max-eps 0.3
```

### Parâmetros

| Parâmetro | Descrição |
|----------|-----------|
| `--attacker` | Caminho da imagem do atacante |
| `--victim` | Caminho da imagem da vítima |
| `--output` | Nome do arquivo final (imagem gerada) |
| `--steps` | Número de iterações de otimização |
| `--frames` | Frames da timeline |
| `--lr` | Learning rate |
| `--reg-weight` | Regularização L2 |
| `--max-eps` | Intensidade máxima da perturbação |
| `--device` | `"cuda"` ou `"cpu"` |

Caso queira rodar apenas no CPU:

```bash
python attack_mediapipe_mask.py ... --device cpu
```

---

## 📁 Saídas geradas

O script produz automaticamente:

- **`<output>`.png** – imagem final adversarial  
- **`<output>_mask_attacker.png`** – máscara quadrilátero no atacante  
- **`<output>_mask_victim.png`** – máscara quadrilátero na vítima  
- **`<output>_timeline.png`** – evolução do ataque ao longo das iterações  

---

## 📄 Referência do Artigo Original

Este projeto é inspirado no artigo:

**ProjAttacker: A Configurable Physical Adversarial Attack for Face Recognition via Projector**  
https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_ProjAttacker_A_Configurable_Physical_Adversarial_Attack_for_Face_Recognition_via_CVPR_2025_paper.pdf

---

## ⚠️ Aviso ético

Este código foi desenvolvido exclusivamente para fins acadêmicos e de pesquisa, com o objetivo de demonstrar vulnerabilidades de sistemas de reconhecimento facial.  
**Nunca utilize este método para finalidade maliciosa ou ilegal.**

---

## 👨‍💻 Autor

**Felipe Romero Pacheco Segundo**
