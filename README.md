# 🧠 Crowd Counting and Localization

A comparative deep learning project for real-time crowd counting and localization using models like **CCTrans**, **CrowdDiff**, **DMCount++**, and **Multi-Scale Attention Networks**, with applications in public safety, urban monitoring, and disaster response.

## 🚀 Features

- 📸 Real-time crowd density estimation from live video feeds  
- 🧩 Multiple architectures: CCTrans (ViT), CrowdDiff (Diffusion), DMCount++, SE Net, YOLOv8  
- 🧠 Attention mechanisms and multi-scale feature extraction  
- 🎯 Precise individual localization using DBSCAN clustering  
- 📈 Evaluation using MAE and RMSE across dense and sparse datasets  

## 📊 Models Compared

| Model                        | MAE   | RMSE  | Strengths                                                                 |
|-----------------------------|-------|-------|---------------------------------------------------------------------------|
| CCTrans (ViT-based)         | 71.36 | 94.35 | Global context, multi-scale features, high computation                    |
| CrowdDiff (Diffusion)       | 67.66 | 91.99 | Clean density maps, multi-hypothesis fusion, slower inference            |
| Multi-Scale Attention Net   | 32.64 | 62.54 | Accurate in sparse scenes, struggles in dense crowds                     |
| DMCount++                   | 27.58 | 42.04 | Great on dense crowds, efficient context capture                         |
| DMCount++ + Self Attention  | 25.00 | 35.21 | Robust to scale, complex but powerful                                    |
| P2P Net (No finetuning)     | 52.44 | 85.78 | Simple architecture, not great for dense scenes                          |
| CSRNet                      | 51.25 | 74.36 | Strong features, adds uncertainty estimation                             |
| SENet                       | 122.89| 154.54| Lightweight but lacks spatial understanding                              |
| YOLOv8                      |   —   |   —   | Very fast, limited accuracy due to NMS and scaling issues                |

## 📁 Dataset

We used a custom dataset called **crowd_wala_dataset**, available for download:

📦 [Download crowd_wala_dataset (Google Drive)](https://drive.google.com/file/d/1Y8yy8Ksy86fX7wOoUDmymHvTjUsDb_cP/view?usp=sharing)

After downloading, extract it into a `data/` directory at the root of the repository.

```bash
mkdir -p data/
unzip crowd_wala_dataset.zip -d data/
```

## 🛠️ Setup

```bash
# Clone the repository
git clone https://github.com/TaherMerchant25/Crowd-Counting-Model-Analysis.git
cd Crowd-Counting-Model-Analysis

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## 🧪 Run Models

```bash
# Example: Run DMCount++ with self-attention
python run_dmcount_plus.py --config configs/dmcount_self_attention.yaml
```

## 🔍 Use Case

The models are deployable in emergency response scenarios for identifying stranded individuals, assessing crowd density in real-time, and preventing stampedes during large events.

## 📷 Live Demo

```bash
python live_demo.py --model dmcount_self_attention.pth
```

## 👥 Contributors

- Aaarat Chadda

- Nipun Yadav

- Taher Merchant

## © 2025 | For academic and research use only.

```CSS

Let me know if you want to include model weights, Colab notebook links, or a license section.

```
