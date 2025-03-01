# Optimized BERT-Based Information Retrieval System
This is an optimized information retrieval system using hybrid approaches that combine TF-IDF (sparse retrieval) with DistilBERT and Q-BERT (dense retrieval). The goal is to balance accuracy and efficiency for large-scale unstructured textual datasets.


## Table of Contents
1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Methodology](#methodology)
4. [Installation](#installation)
5. [Usage](#usage) 
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)


## Overview
Modern information retrieval systems require both semantic understanding and computational efficiency . This project explores the use of hybrid models that integrate:

- **TF-IDF** : For exact keyword matching.
- **DistilBERT/Q-BERT** : For dense semantic retrieval.

The study evaluates different configurations (e.g., learning rates, activation functions) to optimize performance metrics such as Precision , Recall , F1-Score , MRR , and nDCG , while minimizing inference time and memory usage.


## Objectives
- Optimize BERT-based models (DistilBERT and Q-BERT) for scalable text retrieval .
- Implement a hybrid approach combining sparse (TF-IDF) and dense (BERT) methods.
- Evaluate trade-offs between accuracy and efficiency in real-world applications.
- Fine-tune hyperparameters (e.g., learning rate, activation functions) to maximize performance.


## Methodology
1. **Data Preprocessing**
Tokenization, lowercasing, and stopword removal.
Encoding queries and headlines using TF-IDF and BERT-based embeddings .
2. **Model Selection**
- DistilBERT : A smaller, faster version of BERT (~40% reduction in size).
- Q-BERT : A quantized version of BERT (~50% reduction in size).
3. **Hybrid Approach**
Combine TF-IDF (for exact matches) with BERT-based embeddings (for semantic similarity).
Use cosine similarity to rank retrieved headlines.
4. **Hyperparameter Optimization**
Experimented with learning rates (1e-4, 5e-5, etc.) and activation functions (ReLU, Softmax, etc.).
Fine-tuned on domain-specific data to improve MRR by up to 20%.
5. **Evaluation Metrics**
Accuracy Metrics : Precision, Recall, F1-Score, MRR, nDCG.
Efficiency Metrics : Inference Time, Memory Usage.


## Installation

### Prerequisites
- Python 3.8+
- GPU-enabled environment (optional, but recommended for faster training)

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Step 1: Clone the Repository
```bash
git clone https://github.com/harsh16629/Optimized-BERT-Based-Information-Retrieval-System.git
cd Optimized-BERT-Based-Information-Retrieval-System
```

### Step 2: Run the Script
1. To run the hybrid retrieval system:
```bash
python main.py --model [choose your model] --dataset path/to/your/dataset.csv --query "Your query here"
```
2. Command-Line Arguments
- ```--model```: Choose between distilbert or qbert.
- ```--dataset```: Path to the dataset file (CSV format).
- ```--query```: Input query for retrieval.

3. Example
```bash
python main.py --model distilbert --dataset world_news.csv --query "AI advancements"
```

## Results
This work is currently in progress and is being refined rigorously before being submitted for a peer-reviewed analysis and publication. The table below shows the current best results generated with our custom datasets and hardware.

|Model     |Precision|Recall|F1-Score|MRR |nDCG|Inference Time(s)|Memory Usage(MB)|
|----------|---------|------|--------|----|----|-----------------|----------------|
|DistilBERT|0.88     |0.9   |0.89    |0.87|0.92|0.3              |1500            |
|Q-BERT    |0.86     |0.88  |0.87    |0.85|0.9 |0.35             |1000            |


## Contributing
Contributions to this project are most welcome! 
To contribute:

- Fork the repository.
- Create a new branch (git checkout -b feature/YourFeature).
- Commit your changes (git commit -m "Add YourFeature").
- Push to the branch (git push origin feature/YourFeature).
- Open a pull request.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
