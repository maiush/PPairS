# PPairS: Pairwise Preference Search with Linear Probing

[![DOI](https://zenodo.org/badge/777825592.svg)](TODO) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements PPairS, introduced in [**Aligning Language Model Evaluators with Human Judgement**](TODO), a dissertation submitted for the degree of *Master of Research in Environmental Data Science*.

## Overview

Large Language Models (LLM's) are capable automatic evaluators, highly suited for problems in which large datasets of text samples are evaluated on a numerical or Likert scale e.g., scoring factual accuracy or the quality of generated natural language. However, LLM's are still sensitive to prompt design and exhibit biases in such a way that their judgements may be *misaligned* with human assessments. <a href="https://github.com/cambridgeltl/PairS">Pairwise Preference Search</a> (PairS) is a search method designed to exploit LLMs' capabilities at conducting pairwise comparisons instead, in order to partially circumvent the issues with direct-scoring methods, however this approach still relies heavily on prompt design. As an alternative, we make use of concept-based interpretability methods and introduce **Pairwise Preference Search with Linear Probing** (PPairS), which further develops PairS through the construction of contrast pair embeddings and the use of linear probing to directly align an evaluator with human assessments. PPairS achieves state-of-the-art performance on text evaluation tasks and domain-specific problems of fact-checking and uncertainty estimation. Furthermore, it reduces the financial and computational constraints on automatic evaluation using LLM's, as it allows for peformance competitive with frontier, proprietary models using cheaper, smaller, open-weights models.

## Repo Structure

```
PPairS/
├── src/           
│    ├── dev
│    │    ├── benchmarks/       # text evaluation benchmarks - Experiment 1
│    │    ├── sciencefeedback/  # fact checking - Experiment 2
│    │    └── climatex/         # assessing expert confidence in climate science - Experiment 3
│    └── PPairS                 # core of PPairS - including prompt templates and sorting  
├── README.md                   # project overview and instructions
├── pyproject.toml              # config file for the PPairS package
└── thesis.pdf                  # mres dissertation introducing this work
```

## Data

TODO

## Installation

1. Clone the repository:
  ```bash
  git clone https://github.com/maiush/PPairS.git
  cd PPairS
  ```

2. [Optional] Set up a Python environment e.g., using Anaconda:
  ```bash
  conda env create -n ppairs python=3.12 -y
  conda activate ppairs
  ```

3. [Optional - for replication] Download our experiment data [here](TODO):
  ```bash
  wget <TODO>
  unzip -qq <TODO>.zip
  rm <TODO>.zip
  ```

4. Set up a path to your / our data (**important**):
  ```bash
  cd PPairS
  echo data_storage = <PATH_TO_YOUR_DATA> > src/dev/constants.py
  ```

5. Install PPairS:
  ```bash
  cd PPairS
  pip install .
  ```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```
@misc{maiya2024ppairs,
  author       = {Sharan Maiya},
  title        = {{Pairwise Preference Search with Linear Probing (PPairS)}},
  year         = 2024,
  institution  = {University of Cambridge},
  howpublished = {\url{https://github.com/maiush/PPairS}}
}
```

## Funding

This work was support by the [UKRI Centre for Doctoral Training in Application of Artificial Intelligence to the study of Environmental Risks](https://ai4er-cdt.esc.cam.ac.uk/) [EP/S022961/1].

## Contact

For any queries or information, contact [Sharan Maiya](maito:sm2783@cam.ac.uk).

---

<p align="middle">
  <a href="https://ai4er-cdt.esc.cam.ac.uk/"><img src="assets/ai4er_logo.png" width="15%"/></a>
  <a href="https://www.cam.ac.uk/"><img src="assets/cambridge_logo.png" width="56%"/></a>
</p>