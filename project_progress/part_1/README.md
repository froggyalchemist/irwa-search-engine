# Part 1 - Text Processing and Exploratory Data Analysis

This part prepares the fashion product collection for indexing and explores its
content and distributions. All preprocessing, statistics, and visualizations are
implemented in [preprocessing.ipynb](preprocessing.ipynb).

In the [report](IRWA-2026-u212910-u103898-u231623-part-1.pdf) we explain in detail how [preprocessing.ipynb](preprocessing.ipynb) processes the documents as well as the reasoning behind our decisions. It also includes the figures from our exploratory data analysis and the insights we extracted from them.

## Authors

- Ramon Casas i Luque
- Marina Velasco Barriuso
- Shaoya Zhang

## Repository structure

Paths below are relative to the repository root:

```text
📁 data/
├── fashion_products_dataset.json     # Input dataset
├── fashion_products.csv              # Generated product information
└── index_info.csv                    # Generated tokens for each indexed field
📁 project_progress/
└── 📁 part_1/
    ├── preprocessing.ipynb           # Preprocessing and EDA implementation
    └── README.md                     # Setup and reproduction instructions
```

## To reproduce the results

1. **Set up the environment**

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install required packages
python -m pip install pip install -r requirements.txt
```

2. **Make sure `fashion_products_dataset.json` is inside `data/` before running the notebook**.
3. **Run `preprocessing.ipynb`**.

Running the notebook will overwrite `data/fashion_products.csv` and
`data/index_info.csv`.

- `fashion_products.csv` contains the product fields, preserving original text
  and structured information while storing the cleaned numerical values.
- `index_info.csv` contains `pid` and token lists for `title`, `description`,
  `brand`, `category`, `sub_category`, `seller`, and `product_details`. Fields
  remain separate to support future field-specific indexing and weighting.


