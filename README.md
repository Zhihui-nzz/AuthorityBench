# AuthorityBench: Benchmark for LLM Authority Perception

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)


## 📖 Overview
Three sub-datasets:
- **DomainAuth**: 10K web domains annotated with PageRank-based authority scores.
- **EntityAuth**: 22K entities (across basketball, movies, and songs) annotated with popularity-based authority (Wikipedia sitelinks).
- **RAGAuth**: 120 yes/no queries paired with retrieved documents of varying authority, designed to evaluate the practical impact of authority perception on downstream RAG tasks.

---

## 📂 Repository Structure

```text
AuthorityBench/
├── dataset/
│   ├── DomainAuth/      # 10K web domains with fine/coarse-grained authority labels
│   ├── EntityAuth/      # 22K entities across three domains
│   └── RAGAuth/         # 120 QA pairs with retrieved documents of varying authority
├── judge_and_eval/
│   ├── Judges/          # Implementations of different LLM-as-a-Judge paradigms
│   │   ├── Judges_on_DomainAuth/
│   │   └── Judges_on_EntityAuth/
│   ├── eval/            # Scripts to compute Spearman's ρ, Kendall's τ, and accuracy
│   └── run_rag/         # Code for evaluating authority-aware filtering in RAG
└── README.md
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).
