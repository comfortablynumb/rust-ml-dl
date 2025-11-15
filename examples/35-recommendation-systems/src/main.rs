// ============================================================================
// Recommendation Systems
// ============================================================================
//
// Predict user preferences and suggest relevant items.
// Powers YouTube, Netflix, Amazon, Spotify, and most modern platforms.
//
// PROBLEM:
// --------
// • Too many items for users to explore (millions of movies, products, songs)
// • Need to personalize: Different users have different tastes
// • Goal: Predict which items a user will like
//
// TYPES OF RECOMMENDATION SYSTEMS:
// --------------------------------
//
// 1. COLLABORATIVE FILTERING:
//    • "Users who liked X also liked Y"
//    • Uses ratings from similar users/items
//    • No content information needed
//    • Most popular approach
//
// 2. CONTENT-BASED:
//    • "Recommend items similar to what user liked before"
//    • Uses item features (genre, director, keywords)
//    • No need for other users' data
//    • Works for new users
//
// 3. HYBRID:
//    • Combine collaborative + content-based
//    • Best of both worlds
//    • Used by Netflix, YouTube
//
// ============================================================================
// COLLABORATIVE FILTERING
// ============================================================================
//
// Two main approaches:
//
// 1. USER-BASED:
//    • Find users similar to you
//    • Recommend what they liked
//
// 2. ITEM-BASED:
//    • Find items similar to what you liked
//    • Recommend those items
//
// SIMILARITY METRICS:
// -------------------
//
// 1. Cosine Similarity:
//    sim(u, v) = (u · v) / (||u|| × ||v||)
//    • Measures angle between rating vectors
//    • Range: -1 (opposite) to 1 (identical)
//
// 2. Pearson Correlation:
//    • Like cosine, but centers ratings first
//    • Accounts for rating scale differences (some rate 1-3, others 3-5)
//
// 3. Jaccard Similarity:
//    • For binary data (liked/not liked)
//    • |A ∩ B| / |A ∪ B|
//
// USER-BASED COLLABORATIVE FILTERING:
// ------------------------------------
//
// Algorithm:
// ```
// 1. For target user u:
//    Find k most similar users (neighbors)
//
// 2. For each item i not rated by u:
//    Predict rating as weighted average of neighbors' ratings
//    r̂_{u,i} = Σ sim(u,v) × r_{v,i} / Σ sim(u,v)
//
// 3. Recommend items with highest predicted ratings
// ```
//
// Example:
// ```
// User Alice wants movie recommendations
// Similar users: Bob (sim=0.8), Carol (sim=0.6)
//
// Movie "Inception":
//   Bob rated: 5
//   Carol rated: 4
//   Predicted for Alice: (0.8×5 + 0.6×4) / (0.8+0.6) = 4.57
// ```
//
// Pros:
//   ✅ Simple, intuitive
//   ✅ No item features needed
//   ✅ Explains recommendations ("users like you enjoyed...")
//
// Cons:
//   ❌ Sparsity: Most user-item pairs missing
//   ❌ Scalability: O(users²) to find neighbors
//   ❌ Cold start: New users have no ratings
//
// ITEM-BASED COLLABORATIVE FILTERING:
// ------------------------------------
//
// More popular than user-based (used by Amazon)
//
// Algorithm:
// ```
// 1. Pre-compute item-item similarities
//    sim(i, j) = cosine(ratings_i, ratings_j)
//
// 2. For target user u:
//    For each rated item i by u:
//      Find k similar items
//      Predict ratings for those items
//
// 3. Recommend highest predicted ratings
// ```
//
// Why better than user-based?
//   • Items more stable than users (fewer items than users)
//   • Can pre-compute similarities (offline)
//   • Better for explaining ("because you liked X")
//
// ============================================================================
// MATRIX FACTORIZATION
// ============================================================================
//
// Modern collaborative filtering approach
// Key idea: Latent factors represent users and items
//
// RATINGS MATRIX:
// ```
//          Item1  Item2  Item3  Item4
// User1      5      3      ?      1
// User2      4      ?      ?      1
// User3      1      1      5      4
// User4      ?      1      4      4
//
// Sparse! Most entries missing (users rate <1% of items)
// ```
//
// MATRIX FACTORIZATION IDEA:
// ```
// R ≈ U × I^T
//
// R: User-item rating matrix (n_users × n_items)
// U: User factors (n_users × k)
// I: Item factors (n_items × k)
// k: Number of latent factors (50-200 typical)
// ```
//
// Latent factors might represent:
//   • Movies: Action, Romance, Sci-fi, Comedy, ...
//   • Music: Rock, Pop, Classical, Electronic, ...
//   • But learned automatically, not predefined!
//
// Example with k=2:
// ```
// User factors U:
//           Factor1  Factor2
// Alice       0.8      0.1     (likes action, dislikes romance)
// Bob         0.2      0.9     (dislikes action, likes romance)
//
// Item factors I:
//             Factor1  Factor2
// Die Hard      0.9      0.1     (action movie)
// Titanic       0.1      0.9     (romance movie)
//
// Predicted rating Alice for Titanic:
//   r̂ = [0.8, 0.1] · [0.1, 0.9] = 0.8×0.1 + 0.1×0.9 = 0.17 (low!)
//
// Predicted rating Bob for Titanic:
//   r̂ = [0.2, 0.9] · [0.1, 0.9] = 0.2×0.1 + 0.9×0.9 = 0.83 (high!)
// ```
//
// TRAINING (SVD / ALS):
// ---------------------
//
// Loss function:
//   L = Σ (r_{u,i} - û_u · î_i)² + λ(||û_u||² + ||î_i||²)
//
//   Minimize difference between actual and predicted ratings
//   λ: Regularization to prevent overfitting
//
// Optimization:
//
// 1. SVD (Singular Value Decomposition):
//    • Closed-form solution
//    • Works for complete matrices
//    • Modified for sparse data (Funk SVD)
//
// 2. ALS (Alternating Least Squares):
//    • Fix user factors, optimize item factors
//    • Fix item factors, optimize user factors
//    • Repeat until convergence
//    • Parallelizable, scales to billions of ratings
//
// 3. SGD (Stochastic Gradient Descent):
//    • For each rating (u, i, r):
//      e = r - û_u · î_i
//      û_u += α(e·î_i - λ·û_u)
//      î_i += α(e·û_u - λ·î_i)
//
// ADDING BIASES:
// --------------
//
// Account for user and item biases:
//
// r̂_{u,i} = μ + b_u + b_i + û_u · î_i
//
//   μ: Global average rating
//   b_u: User bias (some rate higher/lower than average)
//   b_i: Item bias (some items rated higher/lower)
//   û_u · î_i: User-item interaction
//
// Example:
//   μ = 3.5 (global average)
//   b_Alice = +0.5 (Alice rates 0.5 higher than average)
//   b_Inception = +0.8 (Inception rated 0.8 higher)
//   û_Alice · î_Inception = 0.3 (weak preference match)
//
//   r̂ = 3.5 + 0.5 + 0.8 + 0.3 = 5.1
//
// ============================================================================
// NEURAL COLLABORATIVE FILTERING (NCF)
// ============================================================================
//
// Use deep learning instead of dot product
//
// Architecture:
// ```
// User ID → Embedding → ┐
//                       ├─→ Concatenate → MLP → Prediction
// Item ID → Embedding → ┘
// ```
//
// Why neural?
//   • Dot product is linear: û_u · î_i
//   • Neural network can learn non-linear interactions
//   • Can incorporate features (user age, item category)
//
// GMF (Generalized Matrix Factorization):
// ```
// User embedding: e_u
// Item embedding: e_i
// Prediction: σ(h^T (e_u ⊙ e_i))
//   ⊙: Element-wise product
//   h: Learnable weights
//   σ: Sigmoid activation
// ```
//
// MLP (Multi-Layer Perceptron):
// ```
// Input: [e_u, e_i] (concatenation)
//   ↓
// Dense(128, ReLU)
//   ↓
// Dense(64, ReLU)
//   ↓
// Dense(32, ReLU)
//   ↓
// Dense(1, Sigmoid) → Prediction
// ```
//
// NeuMF (Neural Matrix Factorization):
// ```
// Combines GMF + MLP:
//   GMF path: e_u ⊙ e_i
//   MLP path: MLP([e_u, e_i])
//   Final: Concatenate → Dense → Prediction
// ```
//
// Training:
//   • Positive samples: Actual user-item interactions
//   • Negative samples: Random unobserved interactions (assumed negative)
//   • Loss: Binary cross-entropy or pairwise ranking loss
//
// ============================================================================
// EVALUATION METRICS
// ============================================================================
//
// 1. RMSE (Root Mean Squared Error):
//    RMSE = √[Σ (r - r̂)² / n]
//    • Measures prediction accuracy
//    • Lower is better
//
// 2. MAE (Mean Absolute Error):
//    MAE = Σ |r - r̂| / n
//    • Less sensitive to outliers than RMSE
//
// 3. Precision@K:
//    • Of top K recommended items, how many user actually liked?
//    • Precision@10 = (relevant in top 10) / 10
//
// 4. Recall@K:
//    • Of all items user liked, how many in top K?
//    • Recall@10 = (relevant in top 10) / (total relevant)
//
// 5. NDCG (Normalized Discounted Cumulative Gain):
//    • Accounts for ranking order
//    • Higher-ranked relevant items contribute more
//    • Industry standard
//
// 6. Hit Rate@K:
//    • Did any relevant item appear in top K?
//    • Binary: Yes (1) or No (0)
//
// COLD START PROBLEM:
// -------------------
//
// New users or new items with no ratings
//
// Solutions:
//
// 1. Content-based:
//    • Use item features for new items
//    • Use user demographics for new users
//
// 2. Hybrid approach:
//    • Collaborative for existing users
//    • Content-based for new users/items
//
// 3. Popularity:
//    • Recommend popular items to new users
//    • Non-personalized but better than nothing
//
// 4. Ask questions:
//    • New users rate a few items upfront
//    • Rapid personalization
//
// REAL-WORLD SYSTEMS:
// -------------------
//
// YOUTUBE:
//   • Two-stage:
//     1. Candidate generation: Retrieve 100s from millions
//     2. Ranking: Rank top 100s, return top 10
//   • Features: User history, video features, context (time, device)
//   • Model: Deep neural networks
//   • Objective: Watch time (not just clicks)
//
// NETFLIX:
//   • Hybrid: Collaborative + content + context
//   • Matrix factorization baseline
//   • Neural networks for ranking
//   • Personalized thumbnails, artwork
//   • A/B testing everything
//
// AMAZON:
//   • Item-based collaborative filtering
//   • "Customers who bought X also bought Y"
//   • Real-time updates
//   • Complementary items (batteries with toys)
//
// SPOTIFY:
//   • Collaborative filtering for known songs
//   • Content-based (audio features) for new songs
//   • NLP on playlists
//   • Bandits for exploration
//
// BEYOND ACCURACY:
// ----------------
//
// 1. Diversity:
//    • Don't recommend only similar items
//    • Expose users to variety
//
// 2. Serendipity:
//    • Recommend unexpected items user might like
//    • Balance safe bets with surprises
//
// 3. Novelty:
//    • Recommend items user doesn't know about
//    • Not just popular items
//
// 4. Coverage:
//    • Recommend from full catalog
//    • Avoid filter bubbles
//
// 5. Fairness:
//    • Don't over-recommend popular items
//    • Give niche items chance
//    • Avoid bias (gender, age, race)
//
// 6. Explainability:
//    • "Because you watched X"
//    • "Users who liked X also liked Y"
//    • Trust and transparency
//
// KEY TAKEAWAYS:
// --------------
//
// ✓ Recommendation systems personalize content for users
// ✓ Collaborative filtering: Use ratings from similar users/items
// ✓ Matrix factorization: Learn latent factors for users and items
// ✓ Neural CF: Deep learning for non-linear interactions
// ✓ Evaluation: RMSE, Precision@K, Recall@K, NDCG
// ✓ Cold start: Combine with content-based, ask questions
// ✓ Real-world: YouTube, Netflix, Amazon, Spotify
// ✓ Beyond accuracy: Diversity, serendipity, fairness, explainability
//
// ============================================================================

use ndarray::Array2;
use rand::Rng;
use std::collections::HashMap;

/// User-Item ratings matrix (sparse)
struct RatingsMatrix {
    ratings: HashMap<(usize, usize), f32>, // (user_id, item_id) -> rating
    n_users: usize,
    n_items: usize,
}

impl RatingsMatrix {
    fn new(n_users: usize, n_items: usize) -> Self {
        Self {
            ratings: HashMap::new(),
            n_users,
            n_items,
        }
    }

    fn add_rating(&mut self, user: usize, item: usize, rating: f32) {
        self.ratings.insert((user, item), rating);
    }

    fn get_rating(&self, user: usize, item: usize) -> Option<f32> {
        self.ratings.get(&(user, item)).copied()
    }

    /// Get items rated by user
    fn get_user_ratings(&self, user: usize) -> Vec<(usize, f32)> {
        self.ratings.iter()
            .filter(|((u, _), _)| *u == user)
            .map(|((_, i), &r)| (*i, r))
            .collect()
    }
}

/// Cosine similarity between two rating vectors
fn cosine_similarity(ratings_a: &[(usize, f32)], ratings_b: &[(usize, f32)]) -> f32 {
    let mut common_items: HashMap<usize, (f32, f32)> = HashMap::new();

    for &(item, rating) in ratings_a {
        common_items.entry(item).or_insert((0.0, 0.0)).0 = rating;
    }

    for &(item, rating) in ratings_b {
        common_items.entry(item).or_insert((0.0, 0.0)).1 = rating;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (r_a, r_b) in common_items.values() {
        if *r_a > 0.0 && *r_b > 0.0 {
            dot_product += r_a * r_b;
            norm_a += r_a * r_a;
            norm_b += r_b * r_b;
        }
    }

    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a.sqrt() * norm_b.sqrt())
    } else {
        0.0
    }
}

/// User-based collaborative filtering
struct UserBasedCF<'a> {
    matrix: &'a RatingsMatrix,
}

impl<'a> UserBasedCF<'a> {
    fn new(matrix: &'a RatingsMatrix) -> Self {
        Self { matrix }
    }

    fn predict(&self, user: usize, item: usize, k_neighbors: usize) -> Option<f32> {
        let user_ratings = self.matrix.get_user_ratings(user);

        // Find similar users who rated this item
        let mut similarities = Vec::new();

        for other_user in 0..self.matrix.n_users {
            if other_user == user {
                continue;
            }

            if let Some(rating) = self.matrix.get_rating(other_user, item) {
                let other_ratings = self.matrix.get_user_ratings(other_user);
                let sim = cosine_similarity(&user_ratings, &other_ratings);

                if sim > 0.0 {
                    similarities.push((sim, rating));
                }
            }
        }

        if similarities.is_empty() {
            return None;
        }

        // Sort by similarity and take top k
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        similarities.truncate(k_neighbors);

        // Weighted average
        let mut weighted_sum = 0.0;
        let mut sim_sum = 0.0;

        for (sim, rating) in similarities {
            weighted_sum += sim * rating;
            sim_sum += sim;
        }

        if sim_sum > 0.0 {
            Some(weighted_sum / sim_sum)
        } else {
            None
        }
    }
}

/// Simple matrix factorization using SGD
struct MatrixFactorization {
    user_factors: Array2<f32>,
    item_factors: Array2<f32>,
    n_factors: usize,
}

impl MatrixFactorization {
    fn new(n_users: usize, n_items: usize, n_factors: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize with small random values
        let user_factors = Array2::from_shape_fn((n_users, n_factors), |_| rng.gen_range(-0.1..0.1));
        let item_factors = Array2::from_shape_fn((n_items, n_factors), |_| rng.gen_range(-0.1..0.1));

        Self {
            user_factors,
            item_factors,
            n_factors,
        }
    }

    fn predict(&self, user: usize, item: usize) -> f32 {
        let user_vec = self.user_factors.row(user);
        let item_vec = self.item_factors.row(item);

        user_vec.iter().zip(item_vec.iter()).map(|(u, i)| u * i).sum()
    }

    fn train(
        &mut self,
        matrix: &RatingsMatrix,
        epochs: usize,
        learning_rate: f32,
        lambda: f32,
    ) {
        for epoch in 0..epochs {
            let mut total_error = 0.0;
            let mut count = 0;

            for (&(user, item), &rating) in &matrix.ratings {
                // Predict
                let prediction = self.predict(user, item);
                let error = rating - prediction;
                total_error += error * error;
                count += 1;

                // Update factors (SGD)
                for f in 0..self.n_factors {
                    let u_f = self.user_factors[[user, f]];
                    let i_f = self.item_factors[[item, f]];

                    self.user_factors[[user, f]] += learning_rate * (error * i_f - lambda * u_f);
                    self.item_factors[[item, f]] += learning_rate * (error * u_f - lambda * i_f);
                }
            }

            if epoch % 10 == 0 {
                let rmse = (total_error / count as f32).sqrt();
                println!("  Epoch {}: RMSE = {:.3}", epoch, rmse);
            }
        }
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("Recommendation Systems");
    println!("{}", "=".repeat(70));
    println!();

    // Create sample ratings matrix (MovieLens-style)
    let n_users = 5;
    let n_items = 6;
    let mut matrix = RatingsMatrix::new(n_users, n_items);

    // Add some ratings (user, item, rating 1-5)
    matrix.add_rating(0, 0, 5.0);
    matrix.add_rating(0, 1, 3.0);
    matrix.add_rating(0, 2, 4.0);
    matrix.add_rating(1, 0, 3.0);
    matrix.add_rating(1, 1, 1.0);
    matrix.add_rating(1, 3, 5.0);
    matrix.add_rating(2, 2, 5.0);
    matrix.add_rating(2, 3, 4.0);
    matrix.add_rating(2, 4, 1.0);
    matrix.add_rating(3, 1, 1.0);
    matrix.add_rating(3, 4, 5.0);
    matrix.add_rating(3, 5, 4.0);
    matrix.add_rating(4, 0, 2.0);
    matrix.add_rating(4, 2, 3.0);
    matrix.add_rating(4, 5, 5.0);

    println!("DATASET:");
    println!("--------");
    println!("Users: {}, Items: {}", n_users, n_items);
    println!("Ratings: {} (sparsity: {:.1}%)",
             matrix.ratings.len(),
             100.0 * (1.0 - matrix.ratings.len() as f32 / (n_users * n_items) as f32));
    println!();

    // 1. User-based collaborative filtering
    println!("1. USER-BASED COLLABORATIVE FILTERING:");
    println!("---------------------------------------");
    let cf = UserBasedCF::new(&matrix);

    let test_user = 0;
    let test_item = 3;
    let k = 2;

    if let Some(pred) = cf.predict(test_user, test_item, k) {
        println!("Predicting User {} for Item {} (k={})", test_user, test_item, k);
        println!("Predicted rating: {:.2}", pred);
    } else {
        println!("Not enough data for prediction");
    }
    println!();

    // 2. Matrix factorization
    println!("2. MATRIX FACTORIZATION (SGD):");
    println!("-------------------------------");
    let n_factors = 3;
    let epochs = 50;
    let learning_rate = 0.01;
    let lambda = 0.02;

    println!("Factors: {}, Epochs: {}, LR: {}, Lambda: {}",
             n_factors, epochs, learning_rate, lambda);
    println!();

    let mut mf = MatrixFactorization::new(n_users, n_items, n_factors);
    mf.train(&matrix, epochs, learning_rate, lambda);

    println!();
    println!("PREDICTIONS (Matrix Factorization):");
    println!("User 0, Item 3: {:.2}", mf.predict(0, 3));
    println!("User 1, Item 2: {:.2}", mf.predict(1, 2));
    println!("User 2, Item 0: {:.2}", mf.predict(2, 0));
    println!();

    println!("{}", "=".repeat(70));
    println!("KEY CONCEPTS DEMONSTRATED:");
    println!("{}", "=".repeat(70));
    println!("✓ User-based collaborative filtering (k-nearest neighbors)");
    println!("✓ Cosine similarity for finding similar users");
    println!("✓ Matrix factorization (latent factor model)");
    println!("✓ SGD training with regularization");
    println!();
    println!("RECOMMENDATION APPROACHES:");
    println!("  • Collaborative filtering: User-based, Item-based");
    println!("  • Matrix factorization: SVD, ALS, SGD");
    println!("  • Neural collaborative filtering: Deep learning");
    println!("  • Content-based: Use item features");
    println!("  • Hybrid: Combine multiple approaches");
    println!();
    println!("REAL-WORLD SYSTEMS:");
    println!("  • YouTube: Two-stage (candidate + ranking)");
    println!("  • Netflix: Hybrid (collaborative + content + context)");
    println!("  • Amazon: Item-based CF");
    println!("  • Spotify: Collaborative + content + NLP");
    println!();
    println!("EVALUATION METRICS:");
    println!("  • RMSE/MAE: Prediction accuracy");
    println!("  • Precision@K, Recall@K: Ranking quality");
    println!("  • NDCG: Ranking with position importance");
    println!();
    println!("CHALLENGES:");
    println!("  • Cold start: New users/items");
    println!("  • Sparsity: Most ratings missing");
    println!("  • Scalability: Millions of users/items");
    println!("  • Beyond accuracy: Diversity, serendipity, fairness");
    println!();
}
