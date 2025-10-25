# MEDIC: Model for Explainable Diagnosis using Interpretable Concepts


[![Preprint - arXiv](https://img.shields.io/badge/Open_Access-CEUR_WS-blue)](https://ceur-ws.org/Vol-4059/paper6.pdf
)
[![Preprint - arXiv](https://img.shields.io/badge/Presented_at-EXPLIMED_ECAI_2025-red)](https://sites.google.com/view/explimed-2025)

[**Jacek Karolczak**](https://github.com/jkarolczak),
[**Jerzy Stefanowski**](https://www.cs.put.poznan.pl/jstefanowski/) <br>
Poznan Universtiy of Technology

## Abstract

The ability to interpret machine learning model decisions is critical in such domains as healthcare, where trust in
model predictions is as important as their accuracy. Inspired by the development of prototype parts-based Deep
Neural Networks in computer vision, we propose a new model for tabular data, specifically tailored to medical
records, that requires discretization of diagnostic result norms. Unlike the original vision models that rely on the
spatial structure, our method employs trainable patching over features describing a patient, to learn meaningful
prototypical parts from structured data. These parts are represented as binary or discretized feature subsets. This
allows the model to express prototypes in human-readable terms, enabling alignment with clinical language
and case-based reasoning. Our proposed neural network is inherently interpretable and offers interpretable
concept-based predictions by comparing the patient’s description to learned prototypes in the latent space of
the network. In experiments, we demonstrate that the model achieves classification performance competitive to
widely used baseline models on medical benchmark datasets, while also offering transparency, bridging the gap
between predictive performance and interpretability in clinical decision support.

# Reproducibility

All results presented in the paper were produced using `experiment.py` script shared in this repository. To reproduce results, you just have to clone the repository, install requirements:

```shell
pip install -r requirements.txt
```

Then you will be able to run the script:

```shell
python experiment.py [EXPERIMENT_NAME]
```

To see available experiments, investigate the `experiment.py` file - it is easy to follow. In case of ambiguities or issues, please contact us via email or create an issue in this repository.

## Using MEDIC? Cite us!

```bibtex
@inproceedings{karolczak2025medic,
	author="{Karolczak, Jacek and Stefanowski, Jerzy}",
	title="{An Interpretable Prototype Parts-based Neural Network for Medical Tabular Data}",
	year="2025",
	booktitle="{Proceedings of the Second Workshop on Second Workshop on Explainable Artificial Intelligence for the Medical Domain (EXPLIMED 2025) co-located with 28th European Conference on Artificial Intelligence (ECAI 2025), Bologna, Italy, October 25, 2025}",
	publisher="{CEUR}",
	series="{CEUR Workshop Proceedings}",
	url="https://ceur-ws.org/Vol-4059/paper6.pdf"
}
```