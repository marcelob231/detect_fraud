# Python project for case of study in Machine Learning for fraud detection

## Structure
 ```bash
detect_fraud/
├── src/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # Flask app principal
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── api.py              # Rotas da API
│   │       └── web.py              # Rotas web (se houver interface)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py           # Classe base para todos os modelos
│   │   ├── decision_tree.py        # Árvore de decisão
│   │   ├── neural_network.py       # Redes neurais
│   │   ├── svm.py                  # Support Vector Machine
│   │   ├── clustering.py           # Clustering (K-means, DBSCAN)
│   │   └── isolation_forest.py     # Isolation Forest
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # Carregamento de dados CSV
│   │   ├── preprocessor.py         # Pré-processamento
│   │   └── validator.py            # Validação de dados
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Métricas de avaliação
│   │   ├── comparator.py           # Comparação entre modelos
│   │   └── visualizer.py           # Visualizações e gráficos
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py               # Configurações
│   │   ├── logger.py               # Sistema de logs
│   │   └── helpers.py              # Funções auxiliares
│   └── experiments/
│       ├── __init__.py
│       ├── experiment_runner.py    # Executor de experimentos
│       └── results/                # Resultados dos experimentos
├── data/
│   ├── raw/                        # Dados brutos (CSV original)
│   ├── processed/                  # Dados processados
│   └── results/                    # Resultados e métricas salvas
├── tests/
│   ├── test_basic.py
│   ├── test_data/
│   ├── test_models/
│   └── test_evaluation/
├── docs/                           # Documentação
├── notebooks/                      # Jupyter notebooks para análise
├── config/
│   ├── development.py
│   ├── production.py
│   └── testing.py
└── requirements.txt

```

## How to use it

1. Active enviroment:
   ```bash
   source .venv/Scripts/activate

2. Run application:
   ```bash
   python src/app/main.py
