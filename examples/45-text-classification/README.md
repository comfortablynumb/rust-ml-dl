# Text Classification

Automatically categorizing text documents into predefined classes using machine learning.

## Overview

Text classification is the most fundamental and widely-used Natural Language Processing (NLP) task. Every company that deals with text data uses it in some form. From spam detection to sentiment analysis to content moderation, text classification powers countless real-world applications.

## Running

```bash
cargo run --package text-classification
```

## Why Text Classification is the Most Common NLP Task

### 1. Universal Applicability

Text classification applies to virtually any domain:
- **E-commerce**: Product categorization, review sentiment
- **Customer Support**: Ticket routing, urgency detection
- **Finance**: Fraud detection, news sentiment
- **Healthcare**: Medical coding, clinical notes classification
- **Media**: Content moderation, topic tagging
- **Legal**: Document classification, contract analysis

### 2. Clear Business Value

Unlike many ML tasks, text classification delivers immediate, measurable ROI:
- Automate manual categorization (saves time)
- Process data at scale (millions of documents)
- Consistent classification (no human variability)
- 24/7 operation (no downtime)

### 3. Supervised Learning Framework

Text classification fits the classic supervised ML paradigm:
```
Input: Text document
Output: Category label
Training: Learn from labeled examples
```

This makes it accessible and interpretable compared to more complex NLP tasks like language generation.

### 4. Foundation for Other Tasks

Many advanced NLP applications build on text classification:
- **Named Entity Recognition**: Classify each word as person/org/location
- **Question Answering**: Classify answer candidates
- **Information Retrieval**: Classify document relevance
- **Chatbots**: Classify user intent

## Approaches to Text Classification

### 1. Bag-of-Words (BoW)

**Concept**: Represent text as word frequency counts, ignoring order.

```
Text: "The cat sat on the mat"
BoW: {"the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1}
```

**Vector representation**:
```
Vocabulary: [the, cat, sat, on, mat, dog, ran]
Text vector: [2, 1, 1, 1, 1, 0, 0]
```

**Advantages**:
✅ Simple and fast
✅ Interpretable (can see which words matter)
✅ Works well for short texts
✅ Requires minimal data

**Disadvantages**:
❌ Ignores word order ("dog bites man" = "man bites dog")
❌ No semantic understanding ("good" ≠ "excellent")
❌ High-dimensional sparse vectors
❌ No context (same word, different meanings)

**When to use**:
- Spam detection
- Topic classification
- Simple sentiment analysis
- Limited training data

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)

**Improvement over BoW**: Weight words by importance.

```
TF-IDF(word, doc) = TF(word, doc) × IDF(word)

TF (Term Frequency):
  How often word appears in this document
  TF = count(word in doc) / total words in doc

IDF (Inverse Document Frequency):
  How rare is the word across all documents
  IDF = log(total docs / docs containing word)
```

**Intuition**:
- Common words ("the", "is") get low weight (appear everywhere)
- Rare but present words get high weight (distinctive)

**Example**:
```
Doc1: "Machine learning is great"
Doc2: "Deep learning is powerful"

Word "is": Appears in both → Low IDF → Low weight
Word "machine": Only in Doc1 → High IDF → High weight
```

**Advantages**:
✅ Better than raw counts
✅ Reduces importance of common words
✅ Still interpretable
✅ Works well in practice

**Disadvantages**:
❌ Still ignores word order
❌ No semantic similarity
❌ Fixed vocabulary (can't handle new words)

**When to use**:
- Document similarity
- Search/information retrieval
- Topic classification
- Baseline for comparison

### 3. Word Embeddings (Word2Vec, GloVe)

**Concept**: Represent words as dense vectors in continuous space where similar words are close together.

```
Traditional (one-hot):
  "king": [1, 0, 0, ..., 0]  (vocab_size dimensions)
  "queen": [0, 1, 0, ..., 0]

Word embeddings:
  "king": [0.2, -0.5, 0.8, ...]  (e.g., 300 dimensions)
  "queen": [0.3, -0.4, 0.7, ...]  (similar to king!)
```

**Famous examples**:
```
king - man + woman ≈ queen
Paris - France + Italy ≈ Rome
```

**Methods**:

**Word2Vec (2013)**:
- Skip-gram: Predict context from word
- CBOW: Predict word from context
- Trained on large text corpora

**GloVe (2014)**:
- Global vectors for word representation
- Based on word co-occurrence statistics
- Pre-trained models available

**Advantages**:
✅ Captures semantic similarity
✅ Dense vectors (typically 100-300 dims)
✅ Pre-trained embeddings available
✅ Transfer learning (trained on Wikipedia, etc.)

**Disadvantages**:
❌ Single vector per word (no context)
❌ Can't handle out-of-vocabulary words
❌ Requires aggregation for document-level

**Document representation**:
```
Average word vectors:
  doc_vector = mean(word_vectors)
```

**When to use**:
- Medium-sized datasets
- Need semantic understanding
- Document similarity
- Classification baseline

### 4. RNN (Recurrent Neural Networks)

**Concept**: Process text sequentially, maintaining hidden state.

```
Input: [word1, word2, word3, word4]
  ↓      ↓      ↓      ↓      ↓
 RNN → RNN → RNN → RNN → Output
  ↑      ↑      ↑      ↑
 h0    h1     h2     h3    (hidden states)
```

**Each step**:
```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
```

**Variants**:

**LSTM (Long Short-Term Memory)**:
- Solves vanishing gradient problem
- Gates control information flow
- Better for long sequences

**GRU (Gated Recurrent Unit)**:
- Simplified LSTM
- Fewer parameters
- Often similar performance

**Advantages**:
✅ Captures word order
✅ Handles variable-length text
✅ Learns hierarchical features
✅ Strong performance

**Disadvantages**:
❌ Sequential processing (slow)
❌ Difficulty with long sequences
❌ Needs more training data
❌ Less interpretable

**When to use**:
- Sentiment analysis (order matters: "not good")
- Long documents
- Temporal patterns important
- Sufficient training data (10K+ examples)

### 5. CNN for Text

**Concept**: Apply convolutional filters to capture local patterns (n-grams).

```
Input: Word embeddings [seq_len, embedding_dim]
  ↓
Conv filters (different sizes: 2, 3, 4 words)
  ↓
Max pooling (extract most important features)
  ↓
Fully connected layer → Output
```

**Example filters**:
```
Filter size 2: Captures bigrams ("not good", "very bad")
Filter size 3: Captures trigrams ("not very good")
Filter size 4: Captures 4-grams
```

**Advantages**:
✅ Parallel processing (fast!)
✅ Captures local patterns (n-grams)
✅ Good for short to medium texts
✅ Fewer parameters than RNN

**Disadvantages**:
❌ Limited global context
❌ Fixed filter sizes
❌ Less intuitive for sequences

**When to use**:
- Short texts (tweets, reviews)
- Speed is important
- Local patterns matter (phrases)
- Sentiment analysis

### 6. Transformers (BERT, RoBERTa)

**Concept**: Attention mechanism to model relationships between all words simultaneously.

```
Input: [word1, word2, word3, word4]
  ↓
Self-Attention (each word attends to all words)
  ↓
Feed-forward layers
  ↓
Output: Contextual embeddings
```

**BERT (Bidirectional Encoder Representations from Transformers)**:
```
Pre-training:
  • Masked Language Modeling: Predict [MASK] tokens
  • Next Sentence Prediction

Fine-tuning:
  • Add classification layer on top
  • Train on specific task
```

**Key innovation**: Contextual embeddings
```
"bank" in "river bank" → one vector
"bank" in "money bank" → different vector!
```

**Advantages**:
✅ State-of-the-art performance
✅ Contextual understanding
✅ Pre-trained models (transfer learning)
✅ Bidirectional context
✅ Handles long-range dependencies

**Disadvantages**:
❌ Computationally expensive
❌ Large model size (100M+ parameters)
❌ Needs GPU for training
❌ Overkill for simple tasks

**When to use**:
- State-of-the-art needed
- Complex understanding required
- Sufficient compute available
- Large datasets or transfer learning

## Sentiment Analysis

**Special case of text classification**: Classify text by emotional tone.

### Classes

**Binary**:
- Positive vs Negative
- Example: Movie reviews, product reviews

**Multi-class**:
- Positive, Negative, Neutral
- More granular: Very Negative, Negative, Neutral, Positive, Very Positive

**Aspect-based**:
```
Review: "The food was great but service was slow"
→ Food: Positive
→ Service: Negative
```

### Challenges

**1. Negation**:
```
"good" → Positive
"not good" → Negative

Needs word order understanding!
```

**2. Sarcasm**:
```
"Oh great, another delay" → Actually negative
```

**3. Context**:
```
"This product is sick!" → Positive (slang)
"I am sick" → Negative
```

**4. Mixed sentiment**:
```
"The plot was interesting but acting was terrible"
→ Overall: Negative? Mixed?
```

### Applications

**E-commerce**:
- Review analysis
- Product ranking
- Feature extraction

**Customer Support**:
- Prioritize negative feedback
- Track satisfaction trends
- Identify escalation cases

**Brand Monitoring**:
- Social media sentiment
- PR crisis detection
- Competitor analysis

**Finance**:
- News sentiment for trading
- Earnings call analysis
- Market sentiment tracking

## Real-World Applications

### 1. Email Spam Detection
```
Input: Email text
Output: Spam / Not Spam
Features: Keywords, sender, links
Method: Naive Bayes, SVM, or simple BoW
```

### 2. News Categorization
```
Input: Article
Output: Politics / Sports / Technology / ...
Method: TF-IDF + Logistic Regression (simple, effective)
```

### 3. Customer Support Routing
```
Input: Support ticket
Output: Billing / Technical / Sales
Impact: 80% reduction in routing time
Method: BERT fine-tuned on historical tickets
```

### 4. Content Moderation
```
Input: User comment
Output: Toxic / Safe
Challenges: Context, sarcasm, evolving language
Method: Ensemble of models + human review
```

### 5. Medical Coding
```
Input: Clinical notes
Output: ICD-10 codes
Value: Automate billing, reduce errors
Method: Clinical BERT (domain-specific)
```

### 6. Legal Document Classification
```
Input: Legal document
Output: Contract type, jurisdiction, practice area
Method: BERT + domain vocabulary
```

## Industry Usage

**Reality**: If a company has text data, they use text classification.

**Google**: Email categorization (Primary/Social/Promotions)
**Facebook**: Content moderation, hate speech detection
**Amazon**: Product categorization, review analysis
**Netflix**: Content tagging, recommendation
**Airbnb**: Review sentiment, listing categorization
**Uber**: Support ticket routing, driver/rider issues
**Banks**: Fraud detection, transaction categorization
**Healthcare**: Medical coding, clinical decision support

## Best Practices

### 1. Start Simple

```
Progression:
1. Baseline: BoW + Logistic Regression
2. Better: TF-IDF + SVM
3. Advanced: Word embeddings + Deep learning
4. State-of-art: BERT fine-tuning
```

**Why**: Simple models often work well, fast to iterate.

### 2. Data Quality Over Quantity

```
1000 high-quality labels > 10,000 noisy labels

Quality checklist:
✓ Consistent labeling guidelines
✓ Multiple annotators (measure agreement)
✓ Representative of real-world distribution
✓ Balanced classes (or handle imbalance)
```

### 3. Proper Preprocessing

```
Standard pipeline:
1. Lowercase: "Hello" → "hello"
2. Remove punctuation: "hello!" → "hello"
3. Remove stopwords: "the cat sat" → "cat sat"
4. Stemming/Lemmatization: "running" → "run"
5. Remove URLs, emails, special chars
```

**Warning**: Don't over-preprocess! For sentiment, "!!!" might be important.

### 4. Handle Class Imbalance

```
Problem: 95% negative, 5% positive
Model learns to predict "negative" always (95% accuracy!)

Solutions:
• Oversample minority class
• Undersample majority class
• Class weights in loss function
• SMOTE (synthetic examples)
• Collect more minority examples
```

### 5. Use Cross-Validation

```
k-fold cross-validation:
- Split data into k parts
- Train on k-1, test on 1
- Repeat k times
- Average performance

Better than single train/test split!
```

### 6. Monitor in Production

```
Track:
• Prediction distribution (drift over time?)
• Confidence scores (low confidence → review)
• User feedback (correct predictions?)
• Edge cases (new patterns)

Retrain periodically!
```

## Common Pitfalls

### ❌ Data Leakage

```
Bad: Include test data in TF-IDF vocabulary
Good: Fit TF-IDF on training data only

Bad: Preprocess all data, then split
Good: Split first, preprocess separately
```

### ❌ Overfitting

```
Symptoms:
• High training accuracy, low test accuracy
• Model memorizes training examples

Solutions:
• More data
• Regularization (L1/L2)
• Simpler model
• Dropout (neural networks)
• Early stopping
```

### ❌ Ignoring Baseline

```
Always compare against:
• Random classifier (1/num_classes accuracy)
• Majority class classifier
• Simple BoW + Logistic Regression

If your complex model isn't better, why use it?
```

### ❌ Wrong Metric

```
For imbalanced classes, accuracy is misleading!

Better metrics:
• Precision: Of predicted positive, how many correct?
• Recall: Of actual positive, how many found?
• F1-score: Harmonic mean of precision/recall
• AUC-ROC: Area under ROC curve
```

### ❌ Not Handling Out-of-Vocabulary

```
Problem: Test data has words not in training vocabulary

Solutions:
• Use character-level models
• Subword tokenization (BPE)
• Pre-trained embeddings (Word2Vec, GloVe)
• Transformers (BERT) with WordPiece
```

### ❌ Ignoring Domain Adaptation

```
Model trained on movie reviews won't work well on tweets!

Different:
• Vocabulary
• Sentence structure
• Sentiment expressions

Solution: Fine-tune on target domain data
```

## Papers & Resources

- [Bag-of-Words Model](https://en.wikipedia.org/wiki/Bag-of-words_model) - Classic approach
- [TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting) - Term weighting
- [Word2Vec](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013) - Word embeddings
- [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) (Pennington et al., 2014) - Global vectors
- [CNN for Text](https://arxiv.org/abs/1408.5882) (Kim, 2014) - Convolutional networks
- [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf) (Hochreiter & Schmidhuber, 1997)
- [Attention](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) - Transformers
- [BERT](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018) - Pre-trained transformers

## Key Takeaways

✓ Text classification: Most common NLP task, used everywhere
✓ Approaches: BoW → TF-IDF → Embeddings → RNN → CNN → Transformers
✓ Start simple (BoW/TF-IDF), add complexity as needed
✓ Sentiment analysis: Classify emotional tone (positive/negative/neutral)
✓ Applications: Spam, categorization, sentiment, moderation, routing
✓ Industry: Every company with text data uses this
✓ Best practices: Quality data, proper preprocessing, handle imbalance
✓ Avoid: Data leakage, overfitting, wrong metrics, ignoring baselines
✓ Modern: BERT achieves best results but simple methods often sufficient
✓ Production: Monitor, collect feedback, retrain periodically
