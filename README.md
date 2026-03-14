<p align="center">
  <img src="./assets/logo.png" alt="TIB-SID logo" width="150"/>
</p>

The **TIB Subject Indexing Dataset (TIB-SID)** is a bilingual benchmark for **extreme multi-label text classification (XMTC)** over real library records, designed for **domain classification** and **GND-based subject indexing**. The dataset combines a large, structured, authority-controlled label space with long-tail sparsity, cross-lingual variation, and real-world domain imbalance, making it substantially closer to operational library cataloging than standard text classification benchmarks.

## ✨ At a glance

- **136,569** library records in **JSON-LD** with predefined **train / dev / test** benchmark splits
- **Languages:** English and German
- [**28 domains**](28_domains_list.csv)
- **Record types:** article, book, conference, report, thesis

## ⬇️ Download

Download the dataset here: [data](./library-records-dataset/data)

## 🔗 Related Links

TIB-SID was introduced through the **LLMs4Subjects** shared tasks organized in 2025. More than 12 LLM-based systems were developed and evaluated on the dataset by participating teams worldwide. The shared task websites provide additional context, task details, and leaderboard results.

- [LLMs4Subjects @ SemEval](https://sites.google.com/view/llms4subjects)
- [LLMs4Subjects @ GermEval](https://sites.google.com/view/llms4subjects-germeval/)

## 📖 Citation

If **TIB-SID** useful for your research or project, please consider citing it.

The main dataset paper is listed below. It has been **accepted to [LREC 2026](https://lrec2026.info/)**, and the official proceedings citation will be added here as soon as it is available.

```bibtex
@misc{dsouza2026extrememultilabeltextclassification,
      title={An Extreme Multi-label Text Classification (XMTC) Library Dataset: What if we took "Use of Practical AI in Digital Libraries" seriously?}, 
      author={Jennifer D'Souza and Sameer Sadruddin and Maximilian Kähler and Andrea Salfinger and Luca Zaccagna and Francesca Incitti and Lauro Snidaro and Osma Suominen},
      year={2026},
      eprint={2603.10876},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.10876}, 
}
```

If you would also like to cite the shared task that introduced the broader benchmark setting, please use:

```bibtex
@InProceedings{dsouza-EtAl:2025:SemEval2025,
author    = {D'Souza, Jennifer and Sadruddin, Sameer and Israel, Holger and Begoin, Mathias and Slawig, Diana},
title     = {SemEval-2025 Task 5: LLMs4Subjects - LLM-based Automated Subject Tagging for a National Technical Library's Open-Access Catalog},
booktitle = {Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025)},
month     = {August},
year      = {2025},
address   = {Vienna, Austria},
publisher = {Association for Computational Linguistics},
pages     = {1082--1095},
url       = {https://aclanthology.org/2025.semeval2025-1.139}
}
```

## ⭐ Acknowledgements

This work was supported by the [NFDI4DataScience initiative](https://www.nfdi4datascience.de/) (DFG, German Research Foundation, Grant ID: 460234259) and the [TIB – Leibniz Information Centre for Science and Technology](https://www.tib.eu/en/). We also gratefully acknowledge the subject specialists at TIB who contributed to the curated human evaluation of this work.


## ⚖️ License

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg