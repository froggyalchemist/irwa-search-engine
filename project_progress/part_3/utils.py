"""utils.py
This module contains functions we defined in Part 2 
related to filtering and ranking, as well as useful

"""
import ast
import os
import pandas as pd
import numpy as np
from numpy import linalg as la
import math
from collections import defaultdict, Counter
from array import array
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

TEST_QUERIES = [
    "men's cotton checkered shirt",
    "women casual shirt linen",
    "black sports shoes",
    "soft jacket for women",
    "grey hoodie with pockets",
]

def load_index_info(filename):
    """
    Load preprocessed product fields from a CSV file for indexing.

    :param filename: Path to the CSV file accepted by pandas.read_csv.
    :returns: A tuple (index_info, fields). index_info is a DataFrame with
        parsed fields, sorted by 'pid' in ascending order and with its row
        index reset. fields lists all columns except 'pid' in CSV order.
    """

    index_info = pd.read_csv(filename)

    # List of fields, excluding 'pid': 'title', 'description', etc.
    fields = index_info.columns.to_list()
    fields.remove('pid')

    # Parse fields as lists of terms instead of plain strings
    for f in fields:
        index_info[f] = index_info[f].apply(ast.literal_eval)

    # Sort by PID
    index_info = index_info.sort_values('pid').reset_index(drop=True)

    return index_info, fields

def build_terms(text):
    """
    Preprocess text removing stop words, stemming,
    transforming in lowercase and return the tokens of the text.

    :param text: to be preprocessed
    :returns: a list of tokens corresponding to the input text after the preprocessing
    """

    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')             # keep only words & numbers, drop punctuation
    stop_words = set(stopwords.words("english"))
    for w in ['not', 'no']:
        stop_words.remove(w)                        # keep 'no' and 'not' to preserve meaning of sentences like "shirt with no sleeves"
    text= text.lower()                              # transform in lowercase
    text=  tokenizer.tokenize(text)                 # tokenize the text to get a list of terms
    text=[t for t in text if t not in stop_words]   # eliminate the stopwords
    text=[stemmer.stem(t) for t in text]            # perform stemming
    return text

def create_index_tfidf(index_df: pd.DataFrame, num_documents: int, fields: list):  
    """
    Implement the inverted index and compute tf, df and idf
    
    :param index_df: DataFrame storing pre-processed product information 
    :returns index: the inverted index (implemented through a nested Python dictionary) containing terms and fields as keys and the corresponding
    list of documents where these keys appears in (and the positions) as values.
    :returns tf: normalized term frequency for each term in each document
    :returns df: number of documents each term appear in
    :returns idf: inverse document frequency of each term
    """
    index = defaultdict(dict)
    # tf is also a dictionary of dictionaries e.g. tf["cotton"]["title"][pid] = TF value
    tf = defaultdict(lambda: defaultdict(dict)) 
    # tf = defaultdict(list)  # term frequencies of terms in documents (documents in the same order as in the main index)
    df = defaultdict(int)  # document frequencies of terms in the corpus
    idf = defaultdict(float)

    # Iterate over each row (products)
    for i, row in index_df.iterrows():
        pid = row['pid'] # Product's ID

        # Build index only for this product and field
        # E.g. ['solid', 'women'] --> 'solid': {'title': [pid, [0]]}, 'women': {'title': [pid, [1]]}
        current_product_index = {}

        # Iterate over fields in this row: title, description, brand, ...
        for field in fields:

            terms = row[field] # List of preprocessed terms

            for position, term in enumerate(terms):
                if term in current_product_index.keys():

                    if field in current_product_index[term].keys():
                        # if the term is already in the current_product_index for the current page (current_page_current_product_index)
                        # append the position to the corresponding list
                        current_product_index[term][field][1].append(position)
                    else:
                        current_product_index[term][field] = [pid, array('I',[position])]              
                
                else:
                    # term is not in the dictionary yet
                    current_product_index[term] = {field: [pid, array('I',[position])]} #'I' indicates unsigned int (int in Python)}
                        
        # Normalize term frequencies
        # Compute the denominator to normalize term frequencies.
        # norm is the same for all terms of a document.
        norm = 0
        for term in current_product_index.keys():
            # sum over all the fields for the current term to get the total frequency of the term in the current document
            norm += sum(
                len(posting[1])
                for posting in current_product_index[term].values()
            ) ** 2
        norm = math.sqrt(norm)

        for term in current_product_index.keys():
            for field, posting in current_product_index[term].items():
                # append the tf for current term (tf = term frequency in current doc/norm)
                tf[term][field][pid] = np.round(len(posting[1])/norm, 4) 
        
            df[term] += 1 # increment DF for current term
            
        # Merge the current product index with the main index
        for term, field_dict in current_product_index.items():
            for field, posting_page in field_dict.items():
                index[term].setdefault(field, []).append(posting_page)

    # Compute IDF using the df we calculated before
    for term in df:
        idf[term] = np.round(np.log(float(num_documents/df[term])), 4)

    return index, tf, df, idf

# Rank matching products using field-weighted TF-IDF.
# (if all weights are equal the ranking result should be the same as regular TF-IDF)
def rank_documents(terms, docs, index, idf, tf, fields, field_weights):
    """
    Perform the ranking of the results of a search based on the tf-idf weights

    :param terms: list of query terms
    :param docs: list of documents, to rank, matching the query
    :param index: inverted index data structure
    :param idf: inverted document frequencies
    :param tf: term frequencies
    :param fields: list of fields where terms can be found (e.g ['title', 'description', ...])
    :param field_weights: list of weights for each field

    Returns:
    Print the list of ranked documents
    """
    
    # Empty query or no doc case
    if not terms or not docs:
        return []
    
    # Create dict with fields weights
    if len(field_weights) != len(fields):
        raise ValueError(f"Number of fields weights ({len(field_weights) }) must be the same as number of fields ({len(fields)})")
    weights = dict(zip(fields, field_weights))
    
    # I'm interested only on the element of the docVector corresponding to the query terms
    # The remaining elements would became 0 when multiplied to the query_vector
    doc_vectors = defaultdict(lambda: [0] * len(terms)) # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = Counter(terms)  # get the frequency of each term in the query.
    # Example: collections.Counter(["hello","hello","world"]) --> Counter({'hello': 2, 'world': 1})

    query_norm = la.norm(list(query_terms_count.values()))

    # Avoid duplicates in the query vector
    for termIndex, term in enumerate(query_terms_count):  #termIndex is the index of the term in the query
        if term not in index:
            continue

        ## Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex]=query_terms_count[term]/query_norm * idf[term] # query_vector[0] corresponds to the first term in the query

        # Generate doc_vectors for matching docs
        # index[term][field] contains [doc, positions] records.
        for field, postings in index[term].items():
            for doc, positions in postings:
                if doc in docs:

                    # Sum TF-IDF contributions from all fields
                    doc_vectors[doc][termIndex] += (
                        tf[term][field][doc] * idf[term] * weights[field]
                    )
        
    # Calculate the score of each doc

    # compute the cosine similarity between queyVector and each docVector:
    # HINT: you can use the dot product because in case of normalized vectors it corresponds to the cosine similarity
    # see np.dot
    doc_scores=[[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items() ]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]
    return result_docs

# Conjunctive queries
def search(query, index):
    """
    The output is the list of documents that contain ALL of the query terms.
    So, we will get the list of documents for each query term, and take the intersection of them.

    :param query: Search query
    :param index: Index to use for the search
    :returns docs: List of document IDs that contain ALL of the query terms. 
    """
    query = build_terms(query)

    # Empty string case
    if not query:
        return []
    
    for i, term in enumerate(query):

        # term is not in index, there is no product with this query term
        if term not in index.keys():
            return []

        # store in term_pids the ids of the products that contain "term" in any field
        term_pids = set()
        for field in index[term].keys():
            for posting in index[term][field]:
                pid = posting[0]
                term_pids.add(pid)
        
        # docs = intersection of term_pids
        if i == 0:
            docs = term_pids
        else:
            docs = docs.intersection(term_pids)
            
    docs = list(docs)
    return docs

def search_tf_idf(query, index, tf, idf, fields, field_weights):
    """
    output is the list of documents that contain any of the query terms.
    So, we will get the list of documents for each query term, and take the union of them.
    """
    docs = search(query, index)
    query = build_terms(query)

    ranked_docs = rank_documents(query, docs, index, idf, tf, fields, field_weights)
    return ranked_docs


def display_results(query, ranked_docs, df, top=10):
    print(f"🔎 You searched for '{query}'")
    print(f"\n====================== Top {top} results out of {len(ranked_docs)} ====================== \n")
    
    df_idx = df.set_index("pid")
    
    for rank, d_id in enumerate(ranked_docs[:top], 1):
        if d_id not in df_idx.index:
            continue
            
        row = df_idx.loc[d_id]
        
        # Extract and clean data
        title = row['title']
        brand = str(row.get('brand', 'N/A')) if pd.notna(row.get('brand')) else 'N/A'
        category = str(row.get('category', 'N/A')) if pd.notna(row.get('category')) else 'N/A'
        sub_cat = str(row.get('sub_category', '')) if pd.notna(row.get('sub_category')) else ''
        
        raw_desc = str(row['description']).replace('\n', ' ').replace('\r', ' ') if pd.notna(row['description']) else ""
        desc = raw_desc[:70] + "..." if len(raw_desc) > 70 else raw_desc
        
        # 1. Rank, Title, Brand, Cat > Subcat, PID
        cat_string = f"{category} > {sub_cat}" if sub_cat else category
        print(f"\033[1m\033[96m{rank} {title}\033[0m | \033[95m{brand}\033[0m | \033[94m{cat_string}\033[0m\033[2m | {d_id}\033[0m")

        # 2. Description in grey
        print(f"   \033[90m{desc}\033[0m\n")    


# Evaluation of the ranking methods with manually labeled test queries
def get_rankings(queries, search_functions, top_n=20):
    """
    Run every search function on every query and keep the top_n results.

    :param queries: dict {query_id: query text}
    :param search_functions: dict {method name: function(query) -> list of ranked pids}
    :param top_n: number of results kept per query and method
    :returns: dict {method name: {query_id: [pid_1, ..., pid_top_n]}}
    """
    return {
        method: {qid: list(fn(q))[:top_n] for qid, q in queries.items()}
        for method, fn in search_functions.items()
    }


def export_pooled_rankings(out_path, queries, rankings, products_df, previous_labels_path=None):
    """
    Export the union (pool) of the top results of all methods to a CSV to be labeled by hand.

    Each (query, product) pair appears only once, even if several methods retrieved it,
    so every product is labeled just one time. Rows are sorted by pid inside each query
    so the labeler cannot tell which method ranked a product higher (blind labeling).

    The file is NOT overwritten if it already exists, so manual labels are never lost.

    :param out_path: path of the CSV to create
    :param queries: dict {query_id: query text}
    :param rankings: output of get_rankings
    :param products_df: original products DataFrame (with 'pid', 'title', 'description')
    :param previous_labels_path: optional CSV with columns 'query_id', 'pid', 'labels'
        (e.g. the Part 2 labels) used to prefill labels that we already know
    :returns: the exported (or existing) DataFrame
    """
    if os.path.exists(out_path):
        print(f"{out_path} already exists, nothing exported. Delete it manually to generate it again.")
        return pd.read_csv(out_path)

    # Union of the top results of every method, per query
    pool = []
    for qid, query in queries.items():
        pids = set()
        for method_rankings in rankings.values():
            pids.update(method_rankings[qid])
        pool.extend({"query_id": qid, "query": query, "pid": pid} for pid in sorted(pids))
    export_df = pd.DataFrame(pool)

    # Add original title and description to help the labeler
    info = products_df.drop_duplicates("pid").set_index("pid")[["title", "description"]]
    export_df = export_df.join(info, on="pid")

    # Prefill labels we already have
    export_df["labels"] = np.nan
    if previous_labels_path is not None and os.path.exists(previous_labels_path):
        prev = pd.read_csv(previous_labels_path)[["query_id", "pid", "labels"]].dropna()
        prev = prev.drop_duplicates(["query_id", "pid"]).set_index(["query_id", "pid"])["labels"]
        keys = pd.MultiIndex.from_frame(export_df[["query_id", "pid"]])
        export_df["labels"] = prev.reindex(keys).to_numpy()

    export_df = export_df[["query_id", "query", "pid", "title", "description", "labels"]]
    export_df.to_csv(out_path, index=False)

    n_missing = export_df["labels"].isna().sum()
    print(f"Exported {len(export_df)} rows to {out_path} ({n_missing} still need a label)")
    print(export_df.groupby("query_id").size().rename("pooled products per query").to_string())
    return export_df


def load_labels(path):
    """
    Read the manually labeled CSV and check that every label is a non-negative integer
    (0 = non-relevant; higher = more relevant, e.g. 0/1 or graded 0/1/2).

    :param path: CSV created by export_pooled_rankings, with the 'labels' column filled in
    :returns: DataFrame with columns 'query_id', 'pid', 'labels' (int)
    """
    labels_df = pd.read_csv(path)
    labels = pd.to_numeric(labels_df["labels"], errors="coerce")
    invalid = labels.isna() | (labels < 0) | (labels % 1 != 0)
    if invalid.any():
        raise ValueError(
            f"{invalid.sum()} rows in {path} have a missing or invalid label "
            f"(expected a non-negative integer). First ones:\n"
            f"{labels_df.loc[invalid, ['query_id', 'pid', 'title']].head().to_string()}"
        )
    labels_df["labels"] = labels.astype(int)
    return labels_df[["query_id", "pid", "labels"]]


def dcg_at_k(relevances, k):
    """
    Discounted Cumulative Gain of a list of relevance labels given in ranked order.

    DCG@k = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)
    """
    rel = np.asarray(relevances, dtype=float)[:k]
    discounts = np.log2(np.arange(len(rel)) + 2)
    return np.sum((2 ** rel - 1) / discounts)


def ndcg_at_k(ranked_relevances, all_relevances, k):
    """
    Normalized DCG@k.

    The ideal DCG is computed from ALL the labeled products of the query (the whole pool),
    not only from the ones this method retrieved. This way a method that misses relevant
    products found by the other methods is penalized, and scores are comparable across methods.

    :param ranked_relevances: labels of the method's results, in ranked order
    :param all_relevances: labels of every judged product for this query
    :param k: cutoff
    """
    ideal = dcg_at_k(sorted(all_relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked_relevances, k) / ideal


def evaluate_rankings_ndcg(queries, rankings, labels_df, k=20):
    """
    Compute NDCG@k of every method on every query.

    Products in a ranking that have no label are treated as non-relevant (0) and reported,
    which can happen if a method changed after the labels were exported.

    :param queries: dict {query_id: query text}
    :param rankings: output of get_rankings
    :param labels_df: output of load_labels
    :param k: cutoff
    :returns: DataFrame with one row per query (+ a 'mean' row) and one column per method
    """
    judged = {
        qid: dict(zip(group["pid"], group["labels"]))
        for qid, group in labels_df.groupby("query_id")
    }

    results = {}
    for method, method_rankings in rankings.items():
        results[method] = {}
        for qid in queries:
            qrels = judged.get(qid, {})
            ranked = method_rankings[qid][:k]
            unjudged = [pid for pid in ranked if pid not in qrels]
            if unjudged:
                print(f"[{method}] query {qid}: {len(unjudged)} unlabeled products treated as non-relevant")
            ranked_rel = [qrels.get(pid, 0) for pid in ranked]
            results[method][qid] = ndcg_at_k(ranked_rel, list(qrels.values()), k)

    per_query = pd.DataFrame(results)
    per_query.index = [f"{qid}: {queries[qid]}" for qid in per_query.index]
    per_query.loc["mean"] = per_query.mean()
    return per_query


def rankings_overlap(rankings, k=20):
    """
    Average Jaccard overlap between the top-k results of each pair of methods.

    :param rankings: output of get_rankings
    :returns: square DataFrame (methods x methods)
    """
    methods = list(rankings)
    out = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            jaccards = [
                len(set(rankings[a][q][:k]) & set(rankings[b][q][:k]))
                / max(len(set(rankings[a][q][:k]) | set(rankings[b][q][:k])), 1)
                for q in rankings[a]
            ]
            out.loc[a, b] = np.mean(jaccards)
    return out
