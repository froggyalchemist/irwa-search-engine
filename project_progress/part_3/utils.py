"""utils.py
This module contains functions we defined in Part 2 
related to filtering and ranking, as well as useful

"""
import ast
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

