# PyTerrier_monoQA

This is the [PyTerrier](https://github.com/terrier-org/pyterrier) plugin for the [monoQA]() ranking and answer extracting approaches

## Installation

This repostory can be installed using Pip.

```bash
    pip install --upgrade git+https://github.com/terrierteam/monoQA.git
```
    
You can then import the package in Python after importing pyterrier:

```python
import pyterrier as pt
pt.init()
import pyterrier_dr
```

## Inference

For instance, to load up a monoQA model,

```python
from pyterrier_monoQA import MonoQA
monoqa = MonoQA(top_k=10) # loads 9meo/monoQA by default
```

To use monoQA to rerank and extract an answer.

```python
import pandas as pd
monoqa(pd.DataFrame([
    {"qid":"1", "docno":"8172" ,"query":"measurement of dielectric constant of liquids by the use of microwave techniques","text":"microwave spectroscopy  includes chapters on spectroscope technique\\nand design on measurements on gases liquids and solids on nuclear\\nproperties on molecular structure and on further possible applications\\nof microwaves\\n"},
    {"qid":"2", "docno":"4330","query":"mathematical analysis and design details of waveguide fed microwave radiations","text":"a step by step method for designing waveguides and oscillatory systems\\n","score":-0.4900564551,"rank":0,"answer":"step by step method for designing waveguides and oscillatory systems"}
 ]))
# qid	docno	query	                                                text	                                                answer	                                                score	       rank
#   1	8172	measurement of dielectric constant of liquids ...	microwave spectroscopy includes chapters on s...	microwave spectroscopy includes chapters on sp...	-0.289365	0
#   2	4330	mathematical analysis and design details of wa...	a step by step method for designing waveguides...	a step by step method for designing waveguides...	-0.585294	0
``` 

## Building monoQA pipelines


```python
import pyterrier as pt
from pyterrier_monoQA import MonoQA
monoqa = MonoQA(top_k=10) # loads 9meo/monoQA by default


dataset = pt.get_dataset("irds:vaswani")
bm25 = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index(), wmodel="BM25")
mono_pipeline = bm25 >> pt.text.get_text(dataset, "text") >> monoqa    
```

Note that both approaches require the document text to be included in the dataframe (see [pt.text.get_text](https://pyterrier.readthedocs.io/en/latest/text.html#pyterrier.text.get_text)).

monoQA has the following options:
 - `model` (default: `'9meo/monoQA'`). HGF model name. Defaults to a version trained on OR-QuAC.
 - `tok_model` (default: `'9meo/monoQA'`). HGF tokenizer name.
 - `batch_size` (default: `4`). How many documents to process at the same time.
 - `text_field` (default: `text`). The dataframe attribute in which the document text is stored.
 - `verbose` (default: `True`). Show progress bar.
 - `top_k` (default: `5`). Top k document for answer generator.
 
 ## Examples

Checkout out the notebooks, even on Colab: 
    - Vaswani [[Github](https://github.com/terrierteam/monoQA/blob/main/pyterrier_monoqa_vaswani.ipynb)] [[Colab](https://colab.research.google.com/github/terrierteam/monoQA/blob/main/pyterrier_monoqa_vaswani.ipynb)]
    

## Implementation Details

We use a PyTerrier transformer to score documents using a T5 model.

Sequences longer than the model's maximum of 512 tokens are silently truncated. Consider splitting long texts
into passages and aggregating the results ([examples](https://pyterrier.readthedocs.io/en/latest/text.html#working-with-passages-rather-than-documents)).

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


## References

  - <a id="Macdonald20"/>Craig Macdonald, Nicola Tonellotto. Declarative Experimentation inInformation Retrieval using PyTerrier. Craig Macdonald and Nicola Tonellotto. In Proceedings of ICTIR 2020. https://arxiv.org/abs/2007.14271

## Credits

- Sarawoot Kongyoung, University of Glasgow
