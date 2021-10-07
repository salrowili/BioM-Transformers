# BioM-Transformers: Building Large Biomedical Language Models with BERT, ALBERT and ELECTRA. 

[Paper link](https://www.aclweb.org/anthology/2021.bionlp-1.24)

# Abstract

The impact of design choices on the performance
of biomedical language models recently
has been a subject for investigation. In
this paper, we empirically study biomedical
domain adaptation with large transformer models
using different design choices. We evaluate
the performance of our pretrained models
against other existing biomedical language
models in the literature. Our results show that
we achieve state-of-the-art results on several
biomedical domain tasks despite using similar
or less computational cost compared to other
models in the literature. Our findings highlight
the significant effect of design choices on
improving the performance of biomedical language
models.

# Pre-Trained Models ( PyTorch )

- BioM-ELECTRA-Base-Discriminator [Link](https://huggingface.co/sultan/BioM-ELECTRA-Base-Discriminator)
- BioM-ELECTRA-Base-Generator [Link](https://huggingface.co/sultan/BioM-ELECTRA-Base-Generator)
- BioM-ELECTRA-Large-Discriminator [Link](https://huggingface.co/sultan/BioM-ELECTRA-Large-Discriminator)
- BioM-ELECTRA-Large-Generator [Link](https://huggingface.co/sultan/BioM-ELECTRA-Large-Generator)
- BioM-ALBERT-xxlarge [Link](https://huggingface.co/sultan/BioM-ALBERT-xxlarge)
- BioM-ALBERT-xxlarge-PMC [Link](https://huggingface.co/sultan/BioM-ALBERT-xxlarge-PMC) ( +PMC for 64k steps)
- BioM-ELECTRA-Large-SQuAD2 [Link](https://huggingface.co/sultan/BioM-ELECTRA-Large-SQuAD2)
- BioM-ALBERT-xxlarge-SQuAD2 [Link](https://huggingface.co/sultan/BioM-ALBERT-xxlarge-SQuAD2)


# Pre-Trained LM Models ( TensorFlow )
| Model | Corpus | Vocab | Batch Size | Training Steps | Link |
| --- | --- | --- | --- | ---  | --- |
| BioM-ELECTRA-Base | PubMed Abstracts | 29K PubMed  | 1024 | 500K |  [link](https://drive.google.com/file/d/1-DOBjAim8MHqWSoBPLOv6InYHvdXg6Ru/view?usp=sharing) |
| BioM-ELECTRA-Large | PubMed Abstracts | 29K PubMed  | 4096 | 434K |  [link](https://drive.google.com/file/d/1-60kzBf7X8Y5XiZPdNIQHql82zpOYEnE/view?usp=sharing) |
| BioM-BERT-Large | PubMed Abstracts + PMC | 30K EN Wiki + Books Corpus  | 4096 | 690K |  [link](https://drive.google.com/file/d/1-FxVP98uIPgBNamojahPiRBcyhoKVAwS/view?usp=sharing) 
| BioM-ALBERT-xxlarge | PubMed Abstracts | 30K PubMed | 8192 | 264k | [link](https://drive.google.com/file/d/1-ARTcFGuEj8X9FEUK8adCslsSHTCoPVR/view?usp=sharing) |
| BioM-ALBERT-xxlarge-PMC | PubMed Abstracts + PMC |30K  PubMed  | 8192 | +64k | [link](https://drive.google.com/file/d/1-8oS2Gv97wUFJE52YsCFCORk2ZX8KZNU/view?usp=sharing) |




# SQuAD Fine-Tuned Checkpoints ( TensorFlow )
| Model | Exact Match (EM) | F1 Score  |  Link |
| --- | --- | --- |--- |
| BioM-ELECTRA-Base-SQuAD2 |  81.35 | 84.20 |[Link](https://drive.google.com/file/d/1z0mNdYGVeg7NTBR7SZJvt3oZwLBwi7C4/view?usp=sharing)
| BioM-ELECTRA-Large-SQuAD2 | 85.48 | 88.27 | [Link](https://drive.google.com/file/d/1-KvvN-0tjkMmxCRbRGiuO5ln5KBjJ47e/view?usp=sharing)
| BioM-ELECTRA-Large-MNLI-SQuAD2 | 85.24 | 88.01 | [Link](https://drive.google.com/file/d/18bW62OWAAwH4Gyo1htwresD3RDSfBFeO/view?usp=sharing)
| BioM-ALBERT-xxlarge-SQuAD2 |  83.86 | 86.99 |[Link](https://drive.google.com/file/d/1-HHLPXIyPm_fXTNQ-CxnoSiDUx8KwkgP/view?usp=sharing/)
| BioM-ALBERT-xxlarge-MNLI-SQuAD2 |  84.35 | 87.31 | [Link](https://drive.google.com/file/d/1-G793O1JtFPAgQTJIg_nUG4Z_Q8Zt4Gw/view?usp=sharing)

We implement transferability between MNLI and SQuAD, which was explained in details by [(Jeong, et al., 2020)](https://arxiv.org/abs/2007.00217). We detailed our particpiation in BioASQ9B in this [Paper](http://ceur-ws.org/Vol-2936/paper-14.pdf). To check the performance of our systems (UDEL-LAB) from the official BioASQ leaderboard visit http://participants-area.bioasq.org/results/9b/phaseB/ .

# GluonNLP (MXNet) Checkpoints

More information about GlounNLP https://github.com/dmlc/gluon-nlp

| Model |  Link |
| --- | --- |
| BioM-ELECTRA-Base | [Link](https://drive.google.com/file/d/1kebW-CfKw31UkhpLP53h8UuLqzaIqpWl/view?usp=sharing)
| BioM-ELECTRA-Large |  [Link](https://drive.google.com/file/d/1u49i9H8fAh1m5DGTZwBZBaI-iQTLwB6b/view?usp=sharing)


| Model | Exact Match (EM) | F1 Score  | Link |
| --- | --- | --- |--- |
| BioM-ELECTRA-Base-SQuAD2 | 80.93 | 83.86 | [Link](https://drive.google.com/file/d/1XFEtCucddN66461ggdHfxCuevBhy1tCk/view?usp=sharing)
| BioM-ELECTRA-Large-SQuAD2 | 85.34 | 88.09 | [Link](https://drive.google.com/file/d/1EUlGKhsn8vpCzv3DXx1ckl8MX9_GpY2_/view?usp=sharing)



# Colab Notebook Examples

BioM-ELECTRA-LARGE on NER and ChemProt Task [Link](https://github.com/salrowili/BioM-Transformers/blob/main/examples/Example_of_NER_and_ChemProt_Task_on_TPU.ipynb)

BioM-ELECTRA-Large on SQuAD2.0 and BioASQ7B Factoid tasks [Link](https://github.com/salrowili/BioM-Transformers/blob/main/examples/Example_of_SQuAD2_0_and_BioASQ7B_tasks_with_BioM_ELECTRA_Large_on_TPU.ipynb)

BioM-ALBERT-xxlarge on SQuAD2.0 and BioASQ7B Factoid tasks [Link](https://github.com/salrowili/BioM-Transformers/blob/main/examples/Example_of_SQuAD2_0_and_BioASQ7B_tasks_with_BioM_ALBERT_xxlarge_on_TPU.ipynb)

Text_Classification Task With HuggingFace Transformers and PyTorchXLA [Link](https://github.com/salrowili/BioM-Transformers/blob/main/examples/Fine_Tuning_Biomedical_Models_on_Text_Classification_Task_With_HuggingFace_Transformers_and_PyTorch_XLA.ipynb) (80.74 micro F1 score on ChemProt task)


# Acknowledgment

We would like to acknowledge the support we have from Tensorflow Research Cloud (TFRC) team to grant us access to TPUv3 units.


# Citation

BioM-Transfomers Model

```bibtex
@inproceedings{alrowili-shanker-2021-biom,
title = "{B}io{M}-Transformers: Building Large Biomedical Language Models with {BERT}, {ALBERT} and {ELECTRA}",
author = "Alrowili, Sultan and
Shanker, Vijay",
booktitle = "Proceedings of the 20th Workshop on Biomedical Language Processing",
month = jun,
year = "2021",
address = "Online",
publisher = "Association for Computational Linguistics",
url = "https://www.aclweb.org/anthology/2021.bionlp-1.24",
pages = "221--227",
abstract = "The impact of design choices on the performance of biomedical language models recently has been a subject for investigation. In this paper, we empirically study biomedical domain adaptation with large transformer models using different design choices. We evaluate the performance of our pretrained models against other existing biomedical language models in the literature. Our results show that we achieve state-of-the-art results on several biomedical domain tasks despite using similar or less computational cost compared to other models in the literature. Our findings highlight the significant effect of design choices on improving the performance of biomedical language models.",
}
```

```bibtex
@article{alrowili2021large,
  title={Large biomedical question answering models with ALBERT and ELECTRA},
  author={Alrowili, Sultan and Shanker, K},
  url = "http://ceur-ws.org/Vol-2936/paper-14.pdf",
  journal={CLEF (Working Notes)},
  year={2021}
}
```
