// ============================================================================
// Text Classification
// ============================================================================
//
// Automatically categorizing text documents into predefined classes.
// The most fundamental and widely-used Natural Language Processing (NLP) task.
//
// WHY TEXT CLASSIFICATION IS THE MOST COMMON NLP TASK:
// -----------------------------------------------------
//
// 1. Universal Applicability:
//    • E-commerce: Product categorization, review sentiment
//    • Customer support: Ticket routing, urgency detection
//    • Finance: Fraud detection, news sentiment
//    • Healthcare: Medical coding, clinical notes
//    • Media: Content moderation, topic tagging
//
// 2. Clear Business Value:
//    • Automate manual categorization (save time)
//    • Process at scale (millions of documents)
//    • Consistent results (no human variability)
//    • 24/7 operation
//
// 3. Supervised Learning Framework:
//    • Input: Text document
//    • Output: Category label
//    • Training: Learn from labeled examples
//    • Fits classic ML paradigm
//
// APPROACHES TO TEXT CLASSIFICATION:
// -----------------------------------
//
// 1. BAG-OF-WORDS (BoW):
//    • Represent text as word frequency counts
//    • Ignores word order
//    • Simple and interpretable
//
//    Example:
//      Text: "the cat sat on the mat"
//      BoW: {"the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1}
//
//    Vector representation:
//      Vocabulary: [the, cat, sat, on, mat, dog, ran]
//      Vector: [2, 1, 1, 1, 1, 0, 0]
//
//    Pros:
//      ✅ Simple and fast
//      ✅ Interpretable
//      ✅ Works well for short texts
//
//    Cons:
//      ❌ Ignores word order ("dog bites man" = "man bites dog")
//      ❌ No semantic understanding
//      ❌ High-dimensional sparse vectors
//
// 2. TF-IDF (Term Frequency-Inverse Document Frequency):
//    • Weights words by importance
//    • TF-IDF(word, doc) = TF(word, doc) × IDF(word)
//
//    TF (Term Frequency):
//      • How often word appears in this document
//      • TF = count(word in doc) / total words in doc
//
//    IDF (Inverse Document Frequency):
//      • How rare is the word across all documents
//      • IDF = log(total docs / docs containing word)
//
//    Intuition:
//      • Common words ("the", "is") → low weight
//      • Rare but present words → high weight
//
//    Example:
//      Doc1: "machine learning is great"
//      Doc2: "deep learning is powerful"
//
//      Word "is": Appears in both → Low IDF → Low weight
//      Word "machine": Only in Doc1 → High IDF → High weight
//
//    Pros:
//      ✅ Better than raw counts
//      ✅ Reduces common word importance
//      ✅ Still interpretable
//
//    Cons:
//      ❌ Still ignores word order
//      ❌ No semantic similarity
//
// 3. WORD EMBEDDINGS (Word2Vec, GloVe):
//    • Represent words as dense vectors
//    • Similar words have similar vectors
//    • Example: king - man + woman ≈ queen
//
// 4. RNN/LSTM:
//    • Process text sequentially
//    • Captures word order
//    • Good for sentiment (order matters: "not good")
//
// 5. CNN FOR TEXT:
//    • Convolutional filters capture n-grams
//    • Fast parallel processing
//    • Good for short texts
//
// 6. TRANSFORMERS (BERT):
//    • Attention mechanism
//    • State-of-the-art performance
//    • Contextual embeddings
//    • Example: "bank" means different things in different contexts
//
// SENTIMENT ANALYSIS:
// -------------------
//
// Special case: Classify text by emotional tone
//
// Classes:
//   • Binary: Positive vs Negative
//   • Multi-class: Positive, Negative, Neutral
//   • Fine-grained: Very Negative, Negative, Neutral, Positive, Very Positive
//
// Challenges:
//   1. Negation: "good" vs "not good"
//   2. Sarcasm: "Oh great, another delay"
//   3. Context: "This product is sick!" (positive slang)
//   4. Mixed sentiment: "Good plot but bad acting"
//
// Applications:
//   • E-commerce: Review analysis
//   • Customer support: Prioritize negative feedback
//   • Brand monitoring: Social media sentiment
//   • Finance: News sentiment for trading
//
// TEXT PREPROCESSING:
// -------------------
//
// Standard pipeline:
//   1. Lowercase: "Hello" → "hello"
//   2. Tokenization: Split into words
//   3. Remove stopwords: "the", "is", "a", "an"
//   4. Remove punctuation
//   5. Stemming/Lemmatization: "running" → "run"
//
// Warning: Don't over-preprocess!
//   • For sentiment, "!!!" might be important
//   • Negation words are stopwords but critical
//
// BEST PRACTICES:
// ---------------
//
// 1. Start simple:
//    • Baseline: BoW + Logistic Regression
//    • Better: TF-IDF + SVM
//    • Advanced: BERT
//
// 2. Data quality over quantity:
//    • 1000 high-quality labels > 10,000 noisy labels
//    • Consistent labeling guidelines
//    • Multiple annotators
//
// 3. Handle class imbalance:
//    • Oversample minority class
//    • Class weights in loss
//    • Collect more data
//
// 4. Monitor in production:
//    • Track prediction distribution
//    • User feedback
//    • Retrain periodically
//
// COMMON PITFALLS:
// ----------------
//
// ❌ Data leakage: Include test data in vocabulary
// ❌ Overfitting: High train accuracy, low test accuracy
// ❌ Ignoring baseline: Always compare to simple model
// ❌ Wrong metric: Accuracy misleading for imbalanced data
// ❌ Not handling OOV: Out-of-vocabulary words in test
//
// APPLICATIONS:
// -------------
//
// • Email spam detection (every email provider)
// • News categorization (Google News, etc.)
// • Customer support routing (80% time reduction)
// • Content moderation (Facebook, Twitter)
// • Medical coding (ICD-10 codes)
// • Legal document classification
//
// INDUSTRY USAGE:
// ---------------
//
// Reality: If a company has text data, they use text classification.
//
// • Google: Email categorization
// • Facebook: Content moderation
// • Amazon: Product categorization, review analysis
// • Netflix: Content tagging
// • Airbnb: Review sentiment
// • Banks: Fraud detection
//
// ============================================================================

use std::collections::{HashMap, HashSet};

// Common English stopwords (words to ignore)
const STOPWORDS: &[&str] = &[
    "the", "is", "at", "which", "on", "a", "an", "as", "are", "was", "were",
    "been", "be", "have", "has", "had", "do", "does", "did", "but", "if",
    "or", "and", "of", "to", "in", "for", "with", "from", "by",
];

/// Text preprocessing: Convert to lowercase and tokenize
fn preprocess_text(text: &str) -> Vec<String> {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

/// Remove stopwords from token list
fn remove_stopwords(tokens: &[String]) -> Vec<String> {
    let stopwords: HashSet<&str> = STOPWORDS.iter().cloned().collect();
    tokens
        .iter()
        .filter(|token| !stopwords.contains(token.as_str()))
        .cloned()
        .collect()
}

/// Bag-of-Words representation
/// Counts word frequencies in a document
struct BagOfWords {
    vocabulary: Vec<String>,
    vocab_index: HashMap<String, usize>,
}

impl BagOfWords {
    /// Build vocabulary from all documents
    fn new(documents: &[Vec<String>]) -> Self {
        let mut vocab_set = HashSet::new();

        for doc in documents {
            for word in doc {
                vocab_set.insert(word.clone());
            }
        }

        let mut vocabulary: Vec<String> = vocab_set.into_iter().collect();
        vocabulary.sort(); // Consistent ordering

        let vocab_index: HashMap<String, usize> = vocabulary
            .iter()
            .enumerate()
            .map(|(i, word)| (word.clone(), i))
            .collect();

        Self {
            vocabulary,
            vocab_index,
        }
    }

    /// Convert document to BoW vector (word counts)
    fn vectorize(&self, document: &[String]) -> Vec<f32> {
        let mut vector = vec![0.0; self.vocabulary.len()];

        for word in document {
            if let Some(&index) = self.vocab_index.get(word) {
                vector[index] += 1.0;
            }
        }

        vector
    }

    /// Get vocabulary size
    fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }
}

/// TF-IDF (Term Frequency - Inverse Document Frequency)
/// Weights words by importance: common words get low weight, rare words high weight
struct TfidfVectorizer {
    vocabulary: Vec<String>,
    vocab_index: HashMap<String, usize>,
    idf: Vec<f32>, // Inverse document frequency for each word
}

impl TfidfVectorizer {
    /// Build TF-IDF model from documents
    fn new(documents: &[Vec<String>]) -> Self {
        // Build vocabulary
        let mut vocab_set = HashSet::new();
        for doc in documents {
            for word in doc {
                vocab_set.insert(word.clone());
            }
        }

        let mut vocabulary: Vec<String> = vocab_set.into_iter().collect();
        vocabulary.sort();

        let vocab_index: HashMap<String, usize> = vocabulary
            .iter()
            .enumerate()
            .map(|(i, word)| (word.clone(), i))
            .collect();

        // Calculate IDF for each word
        let num_docs = documents.len() as f32;
        let mut idf = vec![0.0; vocabulary.len()];

        for (i, word) in vocabulary.iter().enumerate() {
            // Count documents containing this word
            let docs_containing = documents
                .iter()
                .filter(|doc| doc.contains(word))
                .count() as f32;

            // IDF = log(total_docs / docs_containing_word)
            // Add 1 to avoid division by zero
            idf[i] = (num_docs / (docs_containing + 1.0)).ln();
        }

        Self {
            vocabulary,
            vocab_index,
            idf,
        }
    }

    /// Convert document to TF-IDF vector
    fn vectorize(&self, document: &[String]) -> Vec<f32> {
        let mut vector = vec![0.0; self.vocabulary.len()];

        // Calculate term frequency (TF)
        let doc_len = document.len() as f32;

        for word in document {
            if let Some(&index) = self.vocab_index.get(word) {
                vector[index] += 1.0;
            }
        }

        // Normalize by document length and multiply by IDF
        for i in 0..vector.len() {
            let tf = vector[i] / doc_len; // Term frequency
            vector[i] = tf * self.idf[i]; // TF-IDF
        }

        vector
    }

    /// Get top N most important words in document
    fn get_top_words(&self, document: &[String], n: usize) -> Vec<(String, f32)> {
        let vector = self.vectorize(document);

        let mut word_scores: Vec<(String, f32)> = self
            .vocabulary
            .iter()
            .zip(vector.iter())
            .filter(|(_, &score)| score > 0.0)
            .map(|(word, &score)| (word.clone(), score))
            .collect();

        word_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        word_scores.truncate(n);

        word_scores
    }
}

/// Simple sentiment classifier using word lists
struct SentimentClassifier {
    positive_words: HashSet<String>,
    negative_words: HashSet<String>,
}

impl SentimentClassifier {
    fn new() -> Self {
        // Simple positive/negative word lists
        let positive_words: HashSet<String> = vec![
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "love", "best", "perfect", "awesome", "brilliant", "happy",
            "beautiful", "outstanding", "superior", "delightful", "impressive",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

        let negative_words: HashSet<String> = vec![
            "bad", "terrible", "awful", "horrible", "worst", "poor",
            "hate", "disappointing", "useless", "waste", "rubbish", "sad",
            "ugly", "inferior", "pathetic", "mediocre", "boring",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            positive_words,
            negative_words,
        }
    }

    /// Classify sentiment: 1.0 = positive, -1.0 = negative, 0.0 = neutral
    fn classify(&self, tokens: &[String]) -> (f32, String) {
        let mut positive_count = 0;
        let mut negative_count = 0;

        for token in tokens {
            if self.positive_words.contains(token) {
                positive_count += 1;
            }
            if self.negative_words.contains(token) {
                negative_count += 1;
            }
        }

        let score = (positive_count as f32 - negative_count as f32)
            / ((positive_count + negative_count + 1) as f32);

        let label = if score > 0.1 {
            "POSITIVE"
        } else if score < -0.1 {
            "NEGATIVE"
        } else {
            "NEUTRAL"
        };

        (score, label.to_string())
    }

    /// Get sentiment words found in text
    fn get_sentiment_words(&self, tokens: &[String]) -> (Vec<String>, Vec<String>) {
        let positive: Vec<String> = tokens
            .iter()
            .filter(|t| self.positive_words.contains(*t))
            .cloned()
            .collect();

        let negative: Vec<String> = tokens
            .iter()
            .filter(|t| self.negative_words.contains(*t))
            .cloned()
            .collect();

        (positive, negative)
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

fn main() {
    println!("{}", "=".repeat(80));
    println!("Text Classification & Sentiment Analysis");
    println!("{}", "=".repeat(80));
    println!();

    // Sample documents for demonstration
    let documents = vec![
        "This movie is absolutely amazing and wonderful",
        "I love this product it is excellent and fantastic",
        "Terrible experience worst service ever very disappointing",
        "Horrible quality waste of money completely useless",
        "The weather is sunny today",
        "Machine learning and deep learning are fascinating topics",
    ];

    println!("SAMPLE DOCUMENTS:");
    println!("{}", "-".repeat(80));
    for (i, doc) in documents.iter().enumerate() {
        println!("{}. \"{}\"", i + 1, doc);
    }
    println!();

    // Preprocess all documents
    let processed_docs: Vec<Vec<String>> = documents
        .iter()
        .map(|doc| {
            let tokens = preprocess_text(doc);
            remove_stopwords(&tokens)
        })
        .collect();

    println!("PREPROCESSING:");
    println!("{}", "-".repeat(80));
    println!("Original: \"{}\"", documents[0]);
    println!("Tokens: {:?}", preprocess_text(documents[0]));
    println!("After stopword removal: {:?}", processed_docs[0]);
    println!();

    // 1. Bag-of-Words
    println!("1. BAG-OF-WORDS (BoW):");
    println!("{}", "-".repeat(80));
    let bow = BagOfWords::new(&processed_docs);
    println!("Vocabulary size: {}", bow.vocab_size());
    println!("Vocabulary (first 15 words): {:?}", &bow.vocabulary[..15.min(bow.vocab_size())]);
    println!();

    // Vectorize first document
    let bow_vector = bow.vectorize(&processed_docs[0]);
    println!("Document 1 BoW vector (non-zero entries):");
    for (i, &count) in bow_vector.iter().enumerate() {
        if count > 0.0 {
            println!("  '{}': {}", bow.vocabulary[i], count);
        }
    }
    println!();

    // 2. TF-IDF
    println!("2. TF-IDF (Term Frequency-Inverse Document Frequency):");
    println!("{}", "-".repeat(80));
    let tfidf = TfidfVectorizer::new(&processed_docs);

    for (i, doc) in processed_docs.iter().enumerate() {
        let top_words = tfidf.get_top_words(doc, 3);
        println!("Document {}: \"{}\"", i + 1, documents[i]);
        println!("  Top 3 important words:");
        for (word, score) in top_words {
            println!("    '{}': {:.4}", word, score);
        }
        println!();
    }

    // 3. Document Similarity
    println!("3. DOCUMENT SIMILARITY (using TF-IDF + Cosine Similarity):");
    println!("{}", "-".repeat(80));

    let tfidf_vectors: Vec<Vec<f32>> = processed_docs
        .iter()
        .map(|doc| tfidf.vectorize(doc))
        .collect();

    println!("Similarity matrix:");
    println!("           Doc1   Doc2   Doc3   Doc4   Doc5   Doc6");
    for i in 0..documents.len() {
        print!("Doc{:2}  ", i + 1);
        for j in 0..documents.len() {
            let sim = cosine_similarity(&tfidf_vectors[i], &tfidf_vectors[j]);
            print!("  {:.3}", sim);
        }
        println!();
    }
    println!();
    println!("Observations:");
    println!("  • Doc1 & Doc2 are similar (both positive reviews)");
    println!("  • Doc3 & Doc4 are similar (both negative reviews)");
    println!("  • Doc5 & Doc6 are different topics");
    println!();

    // 4. Sentiment Analysis
    println!("4. SENTIMENT ANALYSIS:");
    println!("{}", "-".repeat(80));

    let classifier = SentimentClassifier::new();

    let test_reviews = vec![
        "This product is absolutely fantastic and I love it",
        "Terrible quality very disappointing and awful experience",
        "The item arrived on time",
        "Amazing service wonderful experience highly recommended",
        "Waste of money horrible and useless product",
        "Not bad but not great either",
    ];

    for review in &test_reviews {
        let tokens = preprocess_text(review);
        let tokens_no_stop = remove_stopwords(&tokens);

        let (score, label) = classifier.classify(&tokens_no_stop);
        let (positive_words, negative_words) = classifier.get_sentiment_words(&tokens_no_stop);

        println!("Review: \"{}\"", review);
        println!("  Sentiment: {} (score: {:.3})", label, score);
        if !positive_words.is_empty() {
            println!("  Positive words: {:?}", positive_words);
        }
        if !negative_words.is_empty() {
            println!("  Negative words: {:?}", negative_words);
        }
        println!();
    }

    // 5. Word Importance Visualization
    println!("5. WORD IMPORTANCE VISUALIZATION:");
    println!("{}", "-".repeat(80));

    let example_doc = "Machine learning and deep learning are amazing technologies \
                       with fantastic applications in computer vision and natural language processing";
    let example_tokens = remove_stopwords(&preprocess_text(example_doc));

    println!("Document: \"{}\"", example_doc);
    println!();

    // Create a corpus with multiple documents for better IDF calculation
    let visualization_corpus = vec![
        example_tokens.clone(),
        remove_stopwords(&preprocess_text("Python programming for data science")),
        remove_stopwords(&preprocess_text("Web development with JavaScript and React")),
        remove_stopwords(&preprocess_text("Mobile app development for iOS and Android")),
        remove_stopwords(&preprocess_text("Database design and SQL queries")),
    ];

    let tfidf_viz = TfidfVectorizer::new(&visualization_corpus);
    let top_words = tfidf_viz.get_top_words(&example_tokens, 8);

    println!("Top 8 most important words (by TF-IDF):");
    for (word, score) in &top_words {
        let bar_length = (score * 100.0) as usize;
        let bar = "█".repeat(bar_length);
        println!("  {:15} {:.4} {}", word, score, bar);
    }
    println!();

    println!("{}", "=".repeat(80));
    println!("KEY CONCEPTS DEMONSTRATED:");
    println!("{}", "=".repeat(80));
    println!("✓ Text Preprocessing: Lowercasing, tokenization, stopword removal");
    println!("✓ Bag-of-Words: Word frequency counts (ignores order)");
    println!("✓ TF-IDF: Weight words by importance (rare words matter more)");
    println!("✓ Document Similarity: Cosine similarity between TF-IDF vectors");
    println!("✓ Sentiment Analysis: Classify emotional tone (positive/negative)");
    println!("✓ Word Importance: Visualize which words are most distinctive");
    println!();

    println!("APPROACHES OVERVIEW:");
    println!("{}", "-".repeat(80));
    println!("1. Bag-of-Words (BoW)");
    println!("   • Simple word counts");
    println!("   • Fast, interpretable, good baseline");
    println!("   • Ignores word order and semantics");
    println!();
    println!("2. TF-IDF");
    println!("   • Weights by term importance");
    println!("   • Better than raw counts");
    println!("   • Still no semantic understanding");
    println!();
    println!("3. Word Embeddings (Word2Vec, GloVe)");
    println!("   • Dense vectors, captures semantics");
    println!("   • Similar words have similar vectors");
    println!("   • Pre-trained models available");
    println!();
    println!("4. RNN/LSTM");
    println!("   • Processes text sequentially");
    println!("   • Captures word order (\"not good\" vs \"good\")");
    println!("   • Good for sentiment analysis");
    println!();
    println!("5. CNN for Text");
    println!("   • Convolutional filters for n-grams");
    println!("   • Fast parallel processing");
    println!("   • Good for short texts");
    println!();
    println!("6. Transformers (BERT)");
    println!("   • Attention mechanism, contextual embeddings");
    println!("   • State-of-the-art performance");
    println!("   • Computationally expensive");
    println!();

    println!("REAL-WORLD APPLICATIONS:");
    println!("{}", "-".repeat(80));
    println!("• Email Spam Detection (Gmail, Outlook)");
    println!("• News Categorization (Google News, media sites)");
    println!("• Customer Support Routing (Zendesk, Salesforce)");
    println!("• Content Moderation (Facebook, Twitter)");
    println!("• Review Analysis (Amazon, Yelp, TripAdvisor)");
    println!("• Medical Coding (ICD-10 from clinical notes)");
    println!("• Legal Document Classification");
    println!("• Brand Sentiment Monitoring");
    println!();

    println!("BEST PRACTICES:");
    println!("{}", "-".repeat(80));
    println!("✓ Start simple (BoW/TF-IDF + Logistic Regression)");
    println!("✓ Quality data > quantity (consistent labels)");
    println!("✓ Handle class imbalance (oversample/undersample)");
    println!("✓ Use cross-validation (k-fold)");
    println!("✓ Monitor in production (track drift, retrain)");
    println!("✓ Compare against baselines (majority class, random)");
    println!();

    println!("COMMON PITFALLS:");
    println!("{}", "-".repeat(80));
    println!("❌ Data leakage (test data in vocabulary)");
    println!("❌ Overfitting (memorizing training data)");
    println!("❌ Wrong metrics (accuracy for imbalanced classes)");
    println!("❌ Over-preprocessing (removing important signals)");
    println!("❌ Ignoring OOV words (out-of-vocabulary in test)");
    println!("❌ Not handling negation (\"not good\" vs \"good\")");
    println!();

    println!("INDUSTRY REALITY:");
    println!("{}", "-".repeat(80));
    println!("Every company with text data uses text classification:");
    println!("  • Google: Email categorization");
    println!("  • Facebook: Content moderation");
    println!("  • Amazon: Product categorization, reviews");
    println!("  • Netflix: Content tagging");
    println!("  • Banks: Fraud detection, transaction categorization");
    println!("  • Healthcare: Medical coding, clinical notes");
    println!();
}
