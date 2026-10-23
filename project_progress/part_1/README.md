# Part 1 Text Processing and Exploratory Data Analysis

This part prepares the fashion product collection for indexing and explores its
content and distributions. All preprocessing, statistics, and visualizations are
implemented in [preprocessing.ipynb](preprocessing.ipynb).

## Authors

- Ramon Casas i Luque
- Marina Velasco Barriuso
- Shaoya Zhang

## Files and data

Paths below are relative to the repository root:

```text
data/
  fashion_products_dataset.json   # Input dataset supplied by the instructor
  fashion_products.csv           # Generated product information
  index_info.csv                 # Generated tokens for each indexed field
project_progress/part_1/
  preprocessing.ipynb             # Preprocessing and EDA implementation
  README.md                      # Setup and reproduction instructions
```

Place the instructor's dataset at `data/fashion_products_dataset.json` before
running the notebook. The supplied collection contains 28,080 products. The
generated CSV files are recreated by the notebook; they are not required inputs.

## Environment setup

The notebook records Python 3.12.3 as its execution environment. Use Python 3.12
and create a virtual environment from the repository root:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the packages used by this notebook and the Jupyter interface:

```bash
python -m pip install pandas nltk matplotlib wordcloud spacy tqdm ipykernel notebook
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
```

The repository's [requirements.txt](../../requirements.txt) lists the pinned
project dependencies. The command above installs only the packages needed to run
Part 1, together with the notebook interface. The spaCy model is a separate
download. Initial package, model, and stopword downloads require internet access.

## Reproduce the results

1. Activate the environment containing the packages above.
2. From the repository root, start Jupyter in the notebook directory:

   ```bash
   cd project_progress/part_1
   python -m notebook
   ```

3. Open `preprocessing.ipynb` and select the kernel from that environment. If
   using VS Code instead, select the same environment and ensure the notebook's
   working directory is `project_progress/part_1/`.
4. Restart the kernel and run all cells from top to bottom. Later cells depend
   on variables and functions created earlier.
5. Save the notebook to retain the regenerated tables and figures.

The notebook uses `data_dir = "../../data/"`. This path is relative to
`project_progress/part_1/`, **not** the repository root. Running with the wrong
working directory will cause a file-not-found error.

Running the notebook overwrites `data/fashion_products.csv` and
`data/index_info.csv`. Figures and other EDA results appear inline in the
notebook. The NER section processes unique nonblank descriptions and then maps
the predictions back to products; it can take longer than the other EDA sections.

## Processing workflow

1. Load the JSON collection, inspect missing values including blank strings, and
   check that product identifiers are present and unique.
2. Convert prices and ratings to numeric values, and discounts to proportions
   (`0.69` represents 69%). Keep stock availability as a boolean.
3. Lowercase, tokenize, remove English stopwords, and apply Porter stemming to
   each searchable field separately. For `product_details`, include all values
   without merging dictionary keys, so repeated keys do not overwrite values.
4. Preserve product information in `df` and store field-specific token lists in
   `index_df`, linked by `pid`.
5. Export the two CSV files and perform EDA.

The EDA covers category and subcategory frequencies; price, rating, and discount
distributions; brand and seller product counts; stock availability; processed
text lengths; term occurrences and document frequencies; word clouds; and an
exploratory NER analysis of original descriptions.

Product-level statistics use `df`; token-based analyses use `index_df`. Numerical
distributions exclude missing values separately for each field rather than
removing whole products or imputing means. Brand and seller rankings count
products, not sales.

## Output files

- `fashion_products.csv` contains the product fields, preserving original text
  and structured information while storing the cleaned numerical values.
- `index_info.csv` contains `pid` and token lists for `title`, `description`,
  `brand`, `category`, `sub_category`, `seller`, and `product_details`. Fields
  remain separate to support future field-specific indexing and weighting.

CSV stores lists and dictionaries as strings. If reusing `index_info.csv`, parse
the token columns back into lists before indexing:

```python
import ast
import pandas as pd

index_df = pd.read_csv("../../data/index_info.csv")
for column in index_df.columns.drop("pid"):
    index_df[column] = index_df[column].map(ast.literal_eval)
```

This example assumes the same working directory as the notebook. The notebook
itself builds the token lists directly from the JSON input.

## Common setup issues

- **`ModuleNotFoundError`:** install packages in the environment selected as the
  notebook kernel, then restart that kernel.
- **spaCy cannot find `en_core_web_sm`:** run
  `python -m spacy download en_core_web_sm` in the selected environment.
- **NLTK cannot find `stopwords`:** run
  `python -m nltk.downloader stopwords` in the selected environment.
- **Dataset not found:** check the input filename and the working directory
  described above.

Part 1 runs independently of the Flask application and does not use a Groq API
key. The report contains the interpretation of results and the reasoning behind
the preprocessing and analysis decisions.
