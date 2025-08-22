# `README.md` 

# A High-Resolution Digital Twin of the Global Production Network

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type Checking: mypy](https://img.shields.io/badge/type_checking-mypy-blue)](http://mypy-lang.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%23025596?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-blue.svg?style=flat&logo=networkx&logoColor=white)](https://networkx.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-150458.svg?style=flat&logo=python-social-auth&logoColor=white)](https://www.statsmodels.org/stable/index.html)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.12315-b31b1b.svg)](https://arxiv.org/abs/2508.12315)
[![Research](https://img.shields.io/badge/Research-Supply%20Chain%20Economics-green)](https://github.com/chirindaopensource/global_production_network_mapping)
[![Discipline](https://img.shields.io/badge/Discipline-Network%20Science%20%26%20Econometrics-blue)](https://github.com/chirindaopensource/global_production_network_mapping)
[![Methodology](https://img.shields.io/badge/Methodology-Network%20Inference-orange)](https://github.com/chirindaopensource/global_production_network_mapping)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/global_production_network_mapping)

**Repository:** `https://github.com/chirindaopensource/global_production_network_mapping`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"Deciphering the global production network from cross-border firm transactions"** by:

*   Neave O'Clery
*   Ben Radcliffe-Brown
*   Thomas Spencer
*   Daniel Tarling-Hunter

The project provides a complete, end-to-end computational framework for transforming massive-scale, firm-level transaction data into a high-resolution, computable "digital twin" of the global production economy. It implements the paper's novel network inference algorithm, a full suite of network and econometric analyses, and a comprehensive set of validation and robustness checks. The goal is to provide a transparent, robust, and computationally efficient toolkit for researchers and policymakers to replicate, validate, and extend the paper's findings on global supply chain structures and economic diversification.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: main_analysis_orchestrator](#key-callable-main_analysis_orchestrator)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the methodologies presented in the 2025 paper "Deciphering the global production network from cross-border firm transactions." The core of this repository is the iPython Notebook `global_production_network_mapping_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings, from initial data validation to the final execution of a full suite of robustness checks.

The study of global supply chains has been historically constrained by a lack of granular data. This project implements the paper's innovative approach, which leverages a massive dataset of 1 billion firm-to-firm transactions to infer a detailed, product-level production network.

This codebase enables users to:
-   Rigorously validate and cleanse massive-scale transaction and firm metadata.
-   Implement the core network inference algorithm to build a weighted, directed product network.
-   Analyze the network's structure using community detection and centrality measures.
-   Perform multi-pronged validation of the inferred network against external benchmarks and statistical null models.
-   Engineer network-based econometric features to predict national economic diversification.
-   Execute the full Probit regression analysis with fixed effects.
-   Conduct a comprehensive suite of robustness checks to test the stability of the findings.

## Theoretical Background

The implemented methods are grounded in network science, econometrics, and economic complexity theory.

**1. Network Inference from Revealed Preference:**
The core of the methodology is to infer an input-output link from product `i` to product `j` not from direct input tables, but from the observed behavior of firms. The weight of a link, `A_ij`, is a measure of "excess purchase" or revealed preference. It is calculated as the ratio of the probability that a producer of `j` buys `i` to the baseline probability that any firm buys `i`.

$$
A_{i,j} = \frac{|S_i^j|/|S_j|}{|S_i^\dagger|/|S|}
$$

An `A_ij > 1` indicates that producers of `j` have a revealed preference for input `i`, suggesting a production linkage. This method effectively filters out ubiquitous inputs (like packaging) and highlights specific, technologically relevant connections.

**2. Network Density and Economic Diversification:**
The project implements the "density" metric, a concept from economic complexity that measures a country's existing capabilities relevant to a new product. The network-derived upstream and downstream densities measure the proportion of a target product's key suppliers or customers, respectively, that a country already has a comparative advantage in.

$$
d_{p,c} = \frac{\sum_{j \in J_p} I(A_{p,j}) \cdot M_{j,c}}{\sum_{j \in J_p} I(A_{p,j})}
$$

where `J_p` is the set of top-k downstream partners of product `p`, and `M_j,c` is an indicator of country `c`'s presence in product `j`.

**3. Probit Model for Diversification:**
The final analysis uses a Probit model to test the hypothesis that higher network density predicts the probability of a country developing a new export capability in a product. The model includes country and product fixed effects to control for unobserved heterogeneity.

$$
R_{p,c} = \Phi(\alpha + \beta_d d_{p,c} + \gamma_p + \eta_c)
$$

## Features

The provided iPython Notebook (`global_production_network_mapping_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Task-Based Architecture:** The entire pipeline is broken down into 11 distinct, modular tasks, from data validation to robustness checks.
-   **High-Performance Data Engineering:** Utilizes vectorized `pandas` and `numpy` operations to efficiently process and transform large datasets.
-   **Efficient Network Inference:** Implements the core `A_ij` formula and sparsification rules using performant, vectorized calculations.
-   **State-of-the-Art Network Analysis:** Employs the `leidenalg` library for robust community detection and `networkx` for standard centrality measures.
-   **Rigorous Statistical Validation:** Includes a parallelized Monte Carlo simulation framework for testing subgraph modularity against the configuration model.
-   **Professional-Grade Econometrics:** Implements the Probit model with fixed effects using `statsmodels`, including correct calculation of Average Marginal Effects for interpretation.
-   **Comprehensive Robustness Suite:** A full suite of advanced robustness checks to analyze the framework's sensitivity to parameters, temporal windows, and methodological choices.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Input Data Validation (Task 1):** Ingests and rigorously validates all raw data and configuration files.
2.  **Data Preprocessing (Task 2):** Cleanses the transaction log and performs firm entity resolution.
3.  **Firm Classification (Task 3):** Identifies significant producer and purchaser sets for each product.
4.  **Network Inference (Task 4):** Computes the `A_ij` matrix and constructs the network objects.
5.  **Structural Analysis (Task 5):** Performs community detection and topological validation.
6.  **Centrality Calculation (Task 6):** Computes Betweenness and Hub Score centralities.
7.  **Network Validation (Task 7):** Validates the network against external data and a statistical null model.
8.  **Feature Engineering (Task 8):** Computes Rpop, the diversification outcome, and network density metrics.
9.  **Econometric Analysis (Task 9):** Estimates the final Probit models.
10. **Orchestration (Task 10):** Provides a master function to run the entire end-to-end pipeline.
11. **Robustness Analysis (Task 11):** Provides a master function to run the full suite of robustness checks.

## Core Components (Notebook Structure)

The `global_production_network_mapping_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the 11 major tasks.

## Key Callable: main_analysis_orchestrator

The central function in this project is `main_analysis_orchestrator`. It orchestrates the entire analytical workflow, providing a single entry point for running the baseline study replication and the advanced robustness checks.

```python
def main_analysis_orchestrator(
    transactions_log_frame: pd.DataFrame,
    firm_metadata_frame: pd.DataFrame,
    # ... other data inputs
    base_manifest: Dict[str, Any],
    run_robustness_checks: bool = True,
    # ... other robustness configurations
) -> Dict[str, Any]:
    """
    Serves as the top-level entry point for the entire research project.
    """
    # ... (implementation is in the notebook)
```

## Prerequisites

-   Python 3.8+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `networkx`, `statsmodels`, `scikit-learn`, `python-igraph`, `leidenalg`, `joblib`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/global_production_network_mapping.git
    cd global_production_network_mapping
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy networkx statsmodels scikit-learn python-igraph leidenalg joblib
    ```

## Input Data Structure

The pipeline requires four `pandas` DataFrames and two Python dictionaries with specific structures, which are rigorously validated by the first task.
1.  `transactions_log_frame`: Contains transaction-level data.
2.  `firm_metadata_frame`: Contains firm-level metadata.
3.  `comtrade_exports_frame`: Contains country-product level export data.
4.  `country_data_frame`: Contains country-level population data.
5.  `supply_chains_definitions_dict`: Defines product sets for validation.
6.  `replication_manifest`: A comprehensive dictionary controlling all parameters.

A fully specified example of all inputs is provided in the main notebook.

## Usage

The `global_production_network_mapping_draft.ipynb` notebook provides a complete, step-by-step guide. The core workflow is:

1.  **Prepare Inputs:** Load your data `DataFrame`s and define your configuration dictionaries. A complete template is provided.
2.  **Execute Pipeline:** Call the master orchestrator function.

    ```python
    # This single call runs the baseline analysis and all configured robustness checks.
    final_results = main_analysis_orchestrator(
        transactions_log_frame=transactions_df,
        firm_metadata_frame=firms_df,
        comtrade_exports_frame=comtrade_df,
        country_data_frame=country_df,
        supply_chains_definitions_dict=supply_chains,
        base_manifest=replication_manifest,
        run_robustness_checks=True,
        parameter_grid=param_grid,
        methods_to_test=methods_list
    )
    ```
3.  **Inspect Outputs:** Programmatically access any result from the returned dictionary. For example, to view the temporal robustness results:
    ```python
    temporal_df = final_results['robustness_results']['temporal_robustness']
    print(temporal_df.head())
    ```

## Output Structure

The `main_analysis_orchestrator` function returns a single, comprehensive dictionary with two top-level keys:
-   `baseline_results`: A dictionary containing all artifacts from the primary study replication (network objects, analysis DataFrames, econometric models, etc.).
-   `robustness_results`: A dictionary containing the summary DataFrames from each of the executed robustness checks.

## Project Structure

```
global_production_network_mapping/
│
├── global_production_network_mapping_draft.ipynb  # Main implementation notebook
├── requirements.txt                                 # Python package dependencies
├── LICENSE                                          # MIT license file
└── README.md                                        # This documentation file
```

## Customization

The pipeline is highly customizable via the `replication_manifest` dictionary and the arguments to the `main_analysis_orchestrator`. Users can easily modify all relevant parameters for the baseline run or define custom scenarios for the robustness checks.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{oclery2025deciphering,
  title={Deciphering the global production network from cross-border firm transactions},
  author={O'Clery, Neave and Radcliffe-Brown, Ben and Spencer, Thomas and Tarling-Hunter, Daniel},
  journal={arXiv preprint arXiv:2508.12315},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Python Implementation of "Deciphering the global production network from cross-border firm transactions". 
GitHub repository: https://github.com/chirindaopensource/global_production_network_mapping
```

## Acknowledgments

-   Credit to Neave O'Clery, Ben Radcliffe-Brown, Thomas Spencer, and Daniel Tarling-Hunter for their innovative and clearly articulated research.
-   Thanks to the developers of the scientific Python ecosystem (`numpy`, `pandas`, `scipy`, `networkx`, `statsmodels`, etc.) for their powerful open-source tools.

--

*This README was generated based on the structure and content of `global_production_network_mapping_draft.ipynb` and follows best practices for research software documentation.*
