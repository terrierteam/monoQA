# monoQA
This repo contains code and data for EMNLP 2022 paper "monoQA: Multi-Task Learning of Reranking and Answer Extraction for Open-Retrieval Conversational Question Answering"
## Prerequisites

Install dependencies:

```bash
git clone https://github.com/thunlp/ConvDR.git
cd monoQA
pip install -r requirements.txt
```
## Data Preparation

By default, we expect raw data to be stored in `./datasets/raw` and processed data to be stored in `./datasets`:

```bash
mkdir datasets
```

### OR-QuAC

#### OR-QuAC files download

Download necessary OR-QuAC files and store them into `./datasets/or-quac`:

```bash
mkdir datasets/or-quac
cd datasets/or-quac
wget https://ciir.cs.umass.edu/downloads/ORConvQA/all_blocks.txt.gz
wget https://ciir.cs.umass.edu/downloads/ORConvQA/qrels.txt.gz
gzip -d *.txt.gz
mkdir preprocessed
cd preprocessed
wget https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/train.txt
wget https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/test.txt
wget https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/dev.txt
```

