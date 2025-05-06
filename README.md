# Veridion Challenge - Company Classifier Project

## Overview

This project implements a **multi-stage semantic labeling pipeline** to classify companies based on their descriptions, business tags, and categories.  
It combines **contextual and token-level embeddings**, **ConceptNet relationship graphs**, and **multi-phase intelligent filtering** to assign precise, meaningful labels even in noisy, real-world data.



## Project Structure

| Phase | File | Purpose |
|:---|:---|:---|
| Semantic Relationship Filtering | `classifier_filter.ipynb` | Filters ConceptNet relationships and prepares semantic embeddings |
| Main Preprocessing  | `classifier.ipynb` | Cleans company descriptions, loads embeddings, extracts TF-IDF and token features |
| Matching, Post-Boost Filtering & Final Label Selection | `classifier_continuation.ipynb` | Performs initial FAISS matching, contextual boosting, multi-step label filtering, and fallback |
| Shared Utilities | `common_functions_f.py` | Contains shared helper functions used across all notebooks |

>All input files are located in the `input/` directory, while all processed outputs are saved to the `output/` directory.  
>Project code and notebooks are organized under the `notebooks/` directory.

## Input Data

The full input directory (`input/`) is too large to include in this repository.

It was generated using:
- [ConceptNet Numberbatch embeddings](https://github.com/commonsense/conceptnet-numberbatch)
- [ConceptNet assertions](https://github.com/commonsense/conceptnet5/wiki/Downloads)

We worked with filtered and reduced files inside `classifier_filter.ipynb` to create manageable inputs for our pipeline.

## 1. Semantic Relationship Filtering (`classifier_filter.ipynb`)

- **Extract important relations** from ConceptNet:
  -  `/r/IsA`, `/r/HasContext`, `/r/Antonym`, `/r/HasA`, `/r/PartOf`, `/r/InstanceOf`, `/r/Entails`, `/r/SimilarTo`, `/r/FormOf`, `/r/CapableOf`
  - We used mainly `/r/IsA`, `/r/Antonym`, `/r/FormOf`, `/r/HasContext`
- Keep only **English** entries.
- Save clean relations into `filtered_assertions.txt`.

- **Optimize embeddings**:
  - From the ~20GB Numberbatch embeddings file, select only **relevant words** (appearing in company titles).
  - Remove **named entities** (locations, persons) to avoid noise.
  - Save optimized embeddings as `filtered_numberbatch.txt`.

- **Multilingual Term Filtering**:
  - Extract both **foreign** and **English** terms based on company titles.
  - Build a **lightweight filtered embedding file** for faster downstream processing.



## 2. Contextual Semantic Feature Extraction (`classifier.ipynb`)

### Data Preprocessing

- **Text Cleaning**:
  - Split CamelCase terms (e.g., `SeniorService → Senior Service`).
  - Expand acronyms and remove foreign words when necessary.
  - Merge fields (`description`, `tags`, `niche`, `category`) for richer inputs.

- **Business Tags, Niche, Category, and Title Refinement**:
  - Combine **TF-IDF scores** with **embedding similarity** to identify and remove weak or irrelevant tags.
  - Validate terms using **ConceptNet** relationships (`/r/IsA`, `/r/HasContext`, `correlated_terms`, `/r/Antonym`).
  - For short or noisy descriptions:
    - Compare full tag lists directly.
    - Fallback to matching `niche` or `category` fields.
  - Apply **compound-word splitting**, **spelling normalization**, and **plural-to-singular conversion** to improve match quality.
  - Dynamically adjust filtering thresholds to preserve strong business-specific terms, even in sparse descriptions.

- **Special Handling for Short Descriptions**:
  - Compare full phrases instead of fragmenting into individual tokens.

### Embedding Construction

- **Token-level and contextual embeddings** (`SentenceTransformer`).
- **Dynamic similarity graphs**:
  - RelatedWords
  - CorrelatedWords (based on ConceptNet and embeddings)
  - Antonym graphs to penalize false positives.

- **Label Embeddings**:
  - Core/modifier/domain split for fine-grained matching.



## Matching, Boosting, Filtering, and Final Label Selection (`classifier_continuation.ipynb`)


## 1. Matching and Boosting

- **Cosine Similarity Matching** using FAISS to compare company tokens with label embeddings.

- **Specificity-Based Matching Adjustment** (applied *before boosting*):
  - Dynamic cosine thresholds:
    - **Specific terms** (e.g. `"veterinary"`) require **higher similarity**.
    - **Generic terms** (`__`) and **super generic terms** (`::`) tolerate **lower similarity**.
  - Scoring is adjusted:
    - Generic terms → moderately downweighted.
    - Super generic terms → heavily downweighted.
    - Specific, high-importance terms dominate label scores.
- **Antonym Handling and `non_*` Labels**:
  - If the correct label is a negated term (e.g., `non_alcoholic`), but its opposite scores higher (`alcoholic`), we penalize the match using antonym graphs.
  - Labels that are semantically **closer to an antonym than to the original label** are explicitly filtered out.

- **Antonym-Aware Label Filtering**: After cosine-based and contextual matching, we apply a final semantic filter to catch **labels that are opposite in meaning** to the company’s context.

- For each top label:
  - Skip if already weak (cosine difference > 0.2) or contextually strong (> 0.4).
  - Check if the label’s **domain** is semantically **related to the antonym** of a key company term (via `related_words_to_a_word_for_antonyms`).
  - If such a conflict is found, **invalidate the label** by setting its score to -1.
- **Boosting Correlated Terms**:
  - Tokens with low cosine similarity are boosted if conceptually related.
  - Includes ConceptNet, WordNet, and compound-splitting logic.


- **Context and First-Sentence Boosting**:
  - Context embeddings (`context_matrix`) and first-sentence emphasis are used to adjust label scores.
- **Special Boosts**:
  - If tokens match:
    - Company **category/niche** (`niche_plus_cat`)
    - **Strong value words** (`strong_values_all`)
    - **Important first-sentence tokens** (`most_important_words`)
  - To identify `strong_values_all` and `weak_values_all` terms, we computed per-token z-scores across frequency and TF-IDF distributions.  

- **Weak Term Handling**:
  - Tokens from `weak_values_all` are deboosted unless they show similarity > 0.525.

## 2. Post-Boost Filtering and Final Re-Selection

### Step-by-Step Filtering

#### 1. Save Backup of Scores

- `old_list_matrix` is saved to enable recovery if later filters discard good matches.

#### 2. Strong-Term Validation

- Penalize labels with no strong supporting tokens.

#### 3. Specificity-Aware Thresholding

- Labels with:
  - **Specific** parts require high cosine match
  - **Generic** parts (`__`, `::`) tolerate lower match but are downweighted

#### 4. Label Component Validation

- For each label: validate `domain`, `core`, `modifier` using:
  - Cosine similarity
  - ConceptNet links
  - Compound/merged token alignment

- Using `HasContext` relationships from ConceptNet, we handle cases where terms like `e_commerce` and `retail` may appear semantically close (high cosine similarity).
- To verify if they're **truly related** and not just superficially similar, we:
  - Traverse the `/r/HasContext` edge from the ConceptNet graph.
  - Compute the **minimum cosine similarity** between each candidate term and its `HasContext`-related concepts.
  - If the **minimum similarity < 0.2**, we consider the terms **contextually unrelated** and **filter the match**.

#### 5. Contextual Filtering

- From top 30 candidates, keep labels supported by:
  - `first_sentence_matrix > 0.3`
  - `context_matrix > 0.3`
  - `context_matrix_niche > 0.5`
  - `context_matrix_category > 0.5`



## 3. Anchor-Based and Text-Based Label Filtering

### Category-Based Anchor Filtering

This pass filters labels using **company's important business tokens and metadata (niche + category)**.

- Skips filtering if only one label exists or if top label is clearly dominant (score gap ≥ 0.1).
- For each candidate:
  - Compute weighted cosine similarity from its domain/core/modifier to company tokens
  - Penalize generic patterns (`__`, `::`)
- Select **anchor** label with:
  - Most high-similarity matches (≥ 0.475)
  - High mean similarity (≥ 0.085)
  - Contextual support (from 4 matrices)

**Filter out**:
- Any label with cosine < 0.75 to anchor unless structurally related (e.g., matching core/domain)

### Text-Based Anchor Filtering

Runs only when top 2 labels are close (score diff < 0.1).

- Compares all label components to **all company tokens** (not just strong ones).
- Accept label if:
  - At least one component has high similarity (≥ 0.4)
  - Label supported by context/niche/category matrices or first sentence

**Discard** labels with:
- Cosine < 0.85 to anchor
- Mean similarity < 0.1
- No matching core/modifier/domain


## 4. Layered Fallback Strategy ("Onion Model")

If top score ≤ 0.30 or all labels fail filtering, apply multi-layer recovery:

1. **`old_list_matrix`** — early-stage backup
2. **`list_matrix_original_2`** — less aggressively filtered matrix
3. **`list_matrix_original`** — base version
4. **Final Resort** — first valid label based on semantic match

**Validation Includes**:
- `filter_by_text()` and `find_best_indices_cat()`
- Check domain/core/modifier alignment and contextual scores
- Re-rank using fallback matrices + niche/category validation



## Final Outputs
- `challenge_ml_insurance_completed_full_final.csv`:
  - Original company data
  - Adds `insurance_label`: final label selected via semantic matching, boosting, filtering, and recovery logic

## Key Techniques Used

- **Sentence Transformers + FAISS** for fast cosine similarity search.
- **Dynamic Specificity Thresholds** (based on domain criticality).
- **ConceptNet and Semantic Graph Expansion** (correlations and anti-correlations).
- **TF-IDF and Term Filtering** for quality control.
- **Compound Word Decomposition** (splitting `_` into meaningful parts).
- **Multi-Stage Boosting**: combining first sentence, context, and niche strength.
- **Fallback Strategies** when description data is weak.
- **Efficient Memory Handling** with prefiltered embedding spaces.
- **Weighted Component Embedding Matching**: Decomposes labels into **domain**, **core**, and **modifier**, with adaptive weighting strategies depending on label generality and match strength.
- **Parent–Child Concept Heuristics**: Retains moderately similar parent terms when child terms show strong contextual alignment (e.g., `medical_laboratory` vs `laboratory`).
- **Fallback Matrix Strategy (Onion Layer Model)**: Multi-layered recovery using previously saved label matrices when aggressive filtering eliminates valuable candidates.
- **Semantic Antonym Filtering**: Penalizes labels that conflict with strongly correlated antonyms present in the input, reducing false positives.
- **Contextual Weighting Based on First Sentence, Niche, Category, or Description**:  
  Prioritizes labels supported by semantic similarity in the company's **first sentence**, **niche**, **category**, or full **description**, improving alignment with true business meaning.
- **Semantic Antonym Filtering**: Detects and ignores labels with opposing meaning (e.g. "non-profit") when input context implies a contradiction (e.g. commercial activity), using ConceptNet and embedding similarity to avoid misleading matches.
- **Hierarchical Metadata Cleaning**: Cascades metadata refinement—starting from the description to tags, then niche, and finally category.
- **Title-Based Context Boosting**: Leverages cleaned first sentence and title to enhance contextual relevance of label matching.

## Critical Analysis

### Strengths


- **Accurate Top-1 Labeling**: Selects highly meaningful and semantically strong labels, prioritizing business relevance.
- **Context-Aware Adaptability**: Enhances noisy, short, or incomplete descriptions by incorporating cleaned input, **title**, **first sentence**, and metadata signals (tags, niche, category) to improve classification confidence.
- **Smart Multi-Source Filtering**: Applies a **hierarchical cleaning pipeline** for metadata: starting from the description, it progressively filters business tags, niche, and category to ensure consistency and remove weak or misleading inputs.
- **Smart Semantic Filtering**: Integrates embeddings, ConceptNet relationship graphs, TF-IDF scoring, and multi-stage boosting to refine label selection with precision.
- **Multilingual Handling**: Processes foreign terms effectively, mapping them to English concepts and handling multilingual company names.
- **Modular Structure for Easy Scaling**: The system is built in clearly separated steps (like embedding matching, boosting, and filtering), which makes it easier to understand, update, and scale. 
- **Semantic Recovery for Related Concepts**: Recovers correct labels even when cosine similarity is low, by leveraging indirect semantic links (e.g., linking "optician" to "health services").
- **Evidence Combination**: Cross-validates labels by combining signals from token-level, sentence-level, and graph-based matching to minimize false positives.
- **Layered Fallback Resilience**: Uses a multi-stage fallback strategy (onion-layered model) that restores meaningful labels when aggressive filtering fails, improving robustness in edge cases.
- **Structured Label Decomposition and Weighting**: Splits labels into `domain`, `modifier`, and `core` components, dynamically adjusting their weights depending on specificity and confidence.
- **Semantic Antonym Filtering**: Prevents contradictory labels (e.g., "non-profit") in commercial contexts by downranking antonymic candidates using ConceptNet and embedding-based reasoning.
- **Reliable Fallback Mechanism**: When no high-confidence label exists (score ≤ 0.30 or weak token support), the system intelligently falls back through multiple scoring stages (old_list_matrix, original, pre-filtered) to recover valid candidates, ensuring **no company is left unclassified**.
### Weaknesses
- **Heuristic Dependence**: Some thresholds (e.g., cosine similarity, fallback thresholds) are manually tuned and may need re-calibration for different taxonomies.
- **Limited Full Multi-Labeling**: Focused on selecting the most confident single label for most companies
- **Moderate Computational Load**: The full run takes approximately **18 minutes**. While the implementation includes **parallelization** to improve throughput, this may lead to **increased overhead** or **slower performance** on systems with fewer resources or limited parallel processing capability.
- **Minor Hardcoding**: Some components hardcoded
- **Parallelization Caveats**: While the solution leverages parallel processing to speed up computations, this can lead to **resource saturation** on less powerful machines. In practice, the process may require avoiding heavy multitasking during execution to prevent crashes or slowdowns.
- **Contextual Over-association with Related Terms**: The system may overemphasize semantically related tokens without distinguishing structural roles. For instance, a **glassware retailer** may be mislabeled as a **beverage manufacturer** due to contextual proximity between *glass*, *mugs*, and *drinks*. This points to a limitation of relying solely on term proximity without modeling **intent or function**.

- **Contextual Precision Tradeoff**: While the anchor selection aims to identify the most semantically aligned label, limitations in `context_matrix` and `embeddings_index` may occasionally result in selecting the 2nd or 3rd best option.

# Summary

This system combines **contextual embeddings**, **semantic graphs**, and **domain-aware heuristics** into a robust pipeline for **high-precision company labeling**.  
It handles **noisy, multilingual business data** through layered filtering, label decomposition, and semantic recovery — making it scalable, interpretable, and production-ready.  
Throughout the process, we consistently chose the **lesser of two semantic evils**: always selecting a label, but favoring safer, broadly accurate matches over overly specific or potentially misleading ones.  
Ambiguity was addressed through multi-layered reranking, fallback matrices, and heuristics that penalized overconfident false positives — ensuring both full label coverage and high trust 

# Final Reflection
Initially, I was not so familiar with machine learning, but after doing this task, I can confidently say that I learned a lot. After a week, I thought that I finished the task, but I was wrong.

I found myself returning to the problem again and again, rethinking the approach and adjusting my methods as I discovered edge cases or overlooked essential details.
This project helped me grow in many ways. I began to structure the code in a manner that is easier to understand and take into consideration ways in which I can optimize my code. I discovered the importance of each line of code and how a single line could make a difference in terms of time execution. In addition, I grasped semantic modeling by using embeddings such as Sentence Transformers and ConceptNet, and learned how to handle noisy and multilingual data that keeps the system robust. I relied on a reasoning-based approach, where fallback logic, disambiguation, and custom heuristics played a role.

I’m personally proud of this; it’s the most complex and thorough project I’ve developed so far. I invested so much into it because I genuinely enjoyed the task, and I wanted to submit something that I could truly be proud of.
