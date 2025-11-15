"""
Hotel Booking Cancellation Prediction - XGBoost
---------------------------------------------------------------------------------
This script performs the following steps:
1.  Load training ('train.csv') and test ('test.csv') data.
2.  Perform initial data exploration, checking label distribution and calculating
    class weights ('scale_pos_weight') for imbalance.
3.  Engineer a comprehensive set of new features (e.g., interaction, ratio,
    and binary flag features).
4.  Define a ColumnTransformer preprocessing pipeline to apply StandardScaler
    to numeric features and OneHotEncoder to categorical features.
5.  Split the training data into training (80%) and validation (20%) sets
    for model tuning and evaluation.
6.  Apply the preprocessing pipeline (fit on training, transform on validation).
7.  Perform Hyperparameter Tuning for XGBClassifier using RandomizedSearchCV
    with 'f1_macro' scoring, using the validation set.
8.  Store the best parameters found during tuning.
9.  Evaluate the tuned model on the validation set to get a 'baseline' F1 score.
10. Perform a computationally intensive wrapper feature selection using
    Strict Backward Elimination (BE), starting from all features.
11. Identify the 'optimal_feature_set' that yields the highest F1 score on the
    validation set during the BE process.
12. Define a new, final preprocessing pipeline using only this optimal feature set.
13. Train a final XGBClassifier (using the best parameters) on the *entire*
    training dataset, preprocessed with the *final* optimal pipeline.
14. Apply feature engineering to the test data.
15. Transform the test data using the *final* (optimal) preprocessor.
16. Predict labels for the transformed test data using the final trained model.
17. Generate a Kaggle submission file ('sample_submission.csv').
"""

import pandas as pd
import numpy as np
import warnings
import time
import tracemalloc  # Import for memory tracking

# -----------------------------------------------------------------------------
# Script Setup: Start Timers and Memory Tracking
# -----------------------------------------------------------------------------

tracemalloc.start()  # Start tracking memory allocations
total_start_time = time.time()
print(f"Script started at {time.ctime(total_start_time)}")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold  # For data splitting, hyperparameter search, cross-validation
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # For data standardization and one-hot encoding
from sklearn.compose import ColumnTransformer  # For building feature processing pipelines
from sklearn.metrics import f1_score, classification_report, confusion_matrix  # For model evaluation
from scipy.stats import uniform, randint  # For defining hyperparameter search space
from copy import deepcopy  # For deep copying parameters, preventing modification of original dictionary

# -----------------------------------------------------------------------------
# XGBoost Import Check
# -----------------------------------------------------------------------------

try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
    print("XGBoost imported successfully")
except ImportError:
    print("Error: XGBoost is not installed, the program cannot run. Please install XGBoost.")
    XGB_AVAILABLE = False
    exit()

# -----------------------------------------------------------------------------

warnings.filterwarnings('ignore')

# =============================================================================
# 1. Data Loading and Initial Exploration
# =============================================================================

print("=== 1. Data Loading and Initial Exploration ===")


def load_and_explore_data():
    """Load data and perform initial exploration"""
    try:
        # Assuming file paths are train.csv and test.csv
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')

        print(f"Training set shape: {train_df.shape}")  # Training data rows and columns
        # Check label distribution and calculate scale_pos_weight
        label_counts = train_df['label'].value_counts()  # Count data for label 0 and 1
        neg_count = label_counts.get(0, 1)  # Not Canceled (Negative Sample)
        pos_count = label_counts.get(1, 1)  # Canceled (Positive Sample)
        # Calculate class weight for XGBoost's scale_pos_weight parameter to address class imbalance
        scale_pos_weight = neg_count / pos_count  # Calculate class weight

        print(f"Label distribution (0: {neg_count}, 1: {pos_count})")
        print(f"Calculated scale_pos_weight (for XGBoost): {scale_pos_weight:.2f}")

        return train_df, test_df, scale_pos_weight
    except Exception as e:
        print(f"Data loading error: {e}")
        return None, None, 1.0


train_df, test_df, scale_pos_weight = load_and_explore_data()  # Output externally

# --- Critical Error Handling ---
# Check if data loaded successfully, if train_df or test_df is None, terminate
if train_df is None or test_df is None:
    print("\nError: Data loading failed, cannot continue. Please check if 'train.csv' and 'test.csv' exist.")
    exit()
# -----------------------------

# --- Section 1 Metrics ---
# (Intermediate metrics removed)
# -------------------------


# =============================================================================
# 2. Feature Engineering
# =============================================================================

print("\n=== 2. Feature Engineering ===")


def safe_feature_engineering(df):
    """Safe feature engineering function, avoids NaN values"""
    df = df.copy()

    # 1. Basic Transformations
    # Apply log1p to 'lead_time' (includes 0), reduces skewness, makes distribution more normal
    df['log_lead_time'] = np.log1p(df['lead_time'])
    # Apply square root to price, also to mitigate data skewness
    df['sqrt_avg_price'] = np.sqrt(df['avg_price_per_room'] + 1)

    # 2. Interaction Features
    # Total guests
    df['total_guests'] = df['no_of_adults'] + df['no_of_children']
    # Total nights
    df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']

    # 3. Ratio Features (Safe handling of division by zero)
    # Guest/Room ratio? (Here using Total Guests / Adults, may imply presence of children)
    df['guest_to_room_ratio'] = np.where(
        df['no_of_adults'] > 0,
        df['total_guests'] / df['no_of_adults'],
        1  # If no adults, set to 1
    )
    # Average price per night
    df['price_per_night'] = np.where(
        df['total_nights'] > 0,
        df['avg_price_per_room'] / df['total_nights'],
        df['avg_price_per_room']  # If 0 nights stay (e.g., bad data or same-day booking no-show), keep original price
    )
    # Historical cancellation rate
    total_previous = df['no_of_previous_cancellations'] + df['no_of_previous_bookings_not_canceled']
    df['cancellation_ratio'] = np.where(
        total_previous > 0,
        df['no_of_previous_cancellations'] / total_previous,
        0  # If no prior history, cancellation rate is 0
    )
    # Special requests per guest
    df['special_requests_per_guest'] = np.where(
        df['total_guests'] > 0,
        df['no_of_special_requests'] / df['total_guests'],
        0  # If no guests, set to 0
    )

    # 4. Binning and Categorical Features
    # Price Binning
    def get_price_category(price):
        if price <= 50:
            return 'P1_Low'
        elif price <= 100:
            return 'P2_Medium'
        elif price <= 150:
            return 'P3_High'
        elif price <= 200:
            return 'P4_VeryHigh'
        else:
            return 'P5_Luxury'

    df['price_category'] = df['avg_price_per_room'].apply(get_price_category)

    # Stay Length Binning
    def get_stay_category(nights):
        if nights <= 2:
            return 'S1_Short'
        elif nights <= 5:
            return 'S2_Medium'
        elif nights <= 10:
            return 'S3_Long'
        else:
            return 'S4_Extended'

    df['stay_length_category'] = df['total_nights'].apply(get_stay_category)

    # Binning Season by Month
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['season'] = df['arrival_month'].apply(get_season)

    # 5. Binary Features (Creating flags)
    # Is weekend check-in (assuming 5=Fri, 6=Sat)
    df['is_weekend_checkin'] = ((df['arrival_date'] % 7).isin([5, 6])).astype(int)
    # Is long lead booking
    df['long_lead_booking'] = (df['lead_time'] > 60).astype(int)
    # Is short stay
    df['short_stay'] = (df['total_nights'] <= 2).astype(int)
    # Is family booking
    df['family_booking'] = ((df['no_of_children'] > 0) & (df['no_of_adults'] >= 2)).astype(int)
    # Is business travel (guess: single, short stay, no children)
    df['business_travel'] = ((df['no_of_children'] == 0) & (df['no_of_adults'] == 1) &
                             (df['total_nights'] <= 3)).astype(int)
    # Has children
    df['has_children'] = (df['no_of_children'] > 0).astype(int)
    # Has special requests
    df['has_special_requests'] = (df['no_of_special_requests'] > 0).astype(int)
    # Is repeated guest
    df['is_repeated_guest'] = (df['repeated_guest'] == 1).astype(int)
    # Requires parking
    df['requires_parking'] = (df['required_car_parking_space'] > 0).astype(int)

    # 6. Final NaN Check and Handling
    # For all numeric columns, fill NaN with median
    print("\nCheck loss value:")
    print(df.isnull().sum())

    if not df.isnull().any().any():
        print("There is no loss value")
    else:
        print("There are some loss values")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
               df[col] = df[col].fillna(df[col].median())
    # For all categorical (object) columns, fill NaN with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
               mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
               df[col] = df[col].fillna(mode_val)

    print("\nCheck repeating events:")
    columns_to_check = df.columns.difference(['id'])
    is_duplicated = df.duplicated(subset=columns_to_check).sum()
    if is_duplicated>0:
         df_cleaned = df.drop_duplicates(subset=columns_to_check, keep='first')
         print(f"There are {is_duplicated} repeating events having been removed")
    else:
        print("There is no repeating events")

    print(f"Feature engineering complete, new features added: {len(df.columns) - 19}")
    return df


# Apply feature engineering
train_df = safe_feature_engineering(train_df)
test_df = safe_feature_engineering(test_df)

# --- Section 2 Metrics ---
# (Intermediate metrics removed)
# -------------------------


# =============================================================================
# 3. Feature Definition and Preprocessing
# =============================================================================

print("\n=== 3. Preprocessing Pipeline Setup ===")

# --- "Complete" Feature List ---
# Note: These features are a combination of original and engineered features
# Numeric Features: Need scaling
numeric_features = [
    'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
    'no_of_week_nights', 'log_lead_time',
    'arrival_month', 'arrival_date', 'avg_price_per_room',
    'no_of_special_requests', 'no_of_previous_cancellations',
    'no_of_previous_bookings_not_canceled', 'total_guests',
    'total_nights', 'price_per_night', 'cancellation_ratio',
    'special_requests_per_guest', 'sqrt_avg_price', 'guest_to_room_ratio'
]

# Binary Features: Already 0 or 1, no processing needed
binary_features = [
    'is_weekend_checkin', 'long_lead_booking', 'short_stay',
    'family_booking', 'business_travel', 'has_children',
    'has_special_requests', 'is_repeated_guest', 'requires_parking'
]

# Categorical Features: Need one-hot encoding
categorical_features = [
    'type_of_meal_plan', 'room_type_reserved', 'market_segment_type',
    'price_category', 'stay_length_category', 'season',
    'arrival_year'  # Treat as a categorical feature
]

# Aggregate all features used for modeling
all_features = numeric_features + binary_features + categorical_features

print(f"Numeric features: {len(numeric_features)}")
print(f"Binary features: {len(binary_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Total features: {len(all_features)}")

# Create Preprocessing Pipeline (ColumnTransformer)
# This is a powerful tool that can apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        # Transformer 1: 'num'
        # Apply StandardScaler (standardization) to all columns in 'numeric_features' list
        ('num', StandardScaler(), numeric_features),

        # Transformer 2: 'bin'
        # Apply 'passthrough' (i.e., no processing) to all columns in 'binary_features' list
        ('bin', 'passthrough', binary_features),

        # Transformer 3: 'cat'
        # Apply OneHotEncoder (one-hot encoding) to all columns in 'categorical_features' list
        # handle_unknown='ignore': If an unknown category is encountered during test set transformation, ignore it (set all to 0) to avoid errors
        # sparse_output=False: Output a dense matrix (NumPy array) instead of a sparse matrix
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# =============================================================================
# 4. Model Definition and Hyperparameter Tuning
# =============================================================================

print("\n=== 4. Model Configuration and Tuning ===")

# Prepare Data (Relying on model's built-in weight handling for imbalance)
X = train_df[all_features]
y = train_df['label']

# Split Training/Validation Set
# test_size=0.2: Split 20% of data as validation set
# random_state=42: Ensures consistent split every time
# stratify=y: Maintains the same label (y) distribution ratio in train and validation sets as the original data, very important for imbalanced datasets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocess Data
print("Preprocessing data...")
# fit_transform: Learn transformation rules (e.g., mean, variance) on X_train and apply transformation
X_train_transformed = preprocessor.fit_transform(X_train)
# transform: Use rules learned from X_train to transform X_val
X_val_transformed = preprocessor.transform(X_val)
# transform: Also use rules from X_train to transform the full dataset X
X_full_transformed = preprocessor.transform(X)  # Preprocess full dataset, for final training

print(f"Training set size after preprocessing: {X_train_transformed.shape}")

# Define XGBoost Tuning Parameter Space
if XGB_AVAILABLE:
    # Define parameter search space for RandomizedSearchCV
    xgb_param_dist = {
        'n_estimators': randint(100, 500),  # Number of trees
        'max_depth': randint(4, 10),  # Max depth of tree
        'learning_rate': uniform(0.01, 0.1),  # Learning rate
        'subsample': uniform(0.6, 0.4),  # Subsample ratio of training instances (0.6 to 1.0)
        'colsample_bytree': uniform(0.6, 0.4),  # Subsample ratio of columns when constructing each tree (0.6 to 1.0)
        'min_child_weight': randint(1, 10),  # Minimum sum of instance weight (hessian) needed in a child
        'gamma': uniform(0, 0.5)  # Minimum loss reduction required to make a further partition on a leaf node
    }

    # ... (tune_model function remains unchanged) ...
    def tune_model(X_train_t, y_train_t, model_class, param_dist, n_iter=30, cv=5, scoring='f1_macro', **kwargs):
        """Use RandomizedSearchCV for model tuning"""
        print(f"Starting {model_class.__name__} model tuning...")

        # Use StratifiedKFold cross-validation to ensure class distribution is consistent in each fold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        model = model_class(random_state=42, **kwargs)

        # n_iter: Number of iterations for random search
        # scoring: Evaluation metric, using 'f1_macro' (macro-averaged F1)
        # cv: Cross-validation strategy
        # n_jobs=-1: Use all available CPU cores
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=skf,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )

        start_time = time.time()
        random_search.fit(X_train_t, y_train_t)
        end_time = time.time()

        print(f"Tuning complete, time taken: {end_time - start_time:.2f} seconds")
        print(f"Best {scoring} score: {random_search.best_score_:.4f}")
        print(f"Best parameters: {random_search.best_params_}")

        return random_search.best_estimator_


    # =============================================================================
    # 5. Model Training and Validation
    # =============================================================================

    # Implement tuned XGBoost
    final_model = tune_model(
        X_train_transformed, y_train,
        XGBClassifier, xgb_param_dist, n_iter=30,  # n_iter=30 means trying 30 parameter combinations
        eval_metric='logloss', use_label_encoder=False,
        scale_pos_weight=scale_pos_weight  # Pass in class weights
    )
    final_model_name = 'xgboost'
    print("XGBoost tuning complete, setting as final model.")

    # New: Store best parameters, for feature selection and final model
    best_params = final_model.get_params()

# --- Section 4 Metrics ---
# (Intermediate metrics removed)
# -------------------------

print("\n=== 5. Model Training and Final Model ===")

# Evaluate final model (i.e., the tuned XGBoost)
y_pred = final_model.predict(X_val_transformed)  # Model predicts on validation set
final_score = f1_score(y_val, y_pred, average='macro')

print(f"✓ Selecting single model {final_model_name} as the final model")
print(f"Baseline {final_model_name} - Validation Set Macro F1: {final_score:.4f}")

# =============================================================================
# 6. Feature Importance Analysis (For ranking in wrapper feature selection)
# =============================================================================

print("\n=== 6.1 Feature Importance Analysis ===")
'''
feat_imp_df = None
if hasattr(final_model, 'feature_importances_'):
    # Get feature names after OneHotEncoder
    # This step is crucial, as 'categorical_features' were converted into multiple new features
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    # Combine all feature names (Numeric + Binary + One-Hot Encoded Categorical)
    base_feature_names = numeric_features + binary_features
    feature_names = list(base_feature_names) + list(ohe_feature_names)
    importances = final_model.feature_importances_

    # As OHE can cause feature count mismatches, we must ensure list lengths are consistent
    if len(feature_names) != len(importances):
        print("Warning: Mismatch between feature names and importances count, potential issue.")
        # Attempt to match minimum length to avoid index errors
        min_len = min(len(feature_names), len(importances))
        feature_names = feature_names[:min_len]
        importances = importances[:min_len]

    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("Top 15 Feature Importances (Text output only):")
    print(feat_imp_df.head(15))
else:
    print("Cannot extract feature importances from the final model.")
'''
# =============================================================================
# 6.2 Wrapper Feature Selection: Strict Backward Elimination (BE)(Feature Ablation)
# =============================================================================

print("\n" + "=" * 50)
print("=== 6.2 Wrapper Method: Strict Backward Elimination (BE) ===")
print("Note: This method is computationally expensive, but can find the optimal subset")
print("=" * 50)

# --- Section 6.2 Timer Start ---
# (Intermediate metrics removed)
# -------------------------------

# Store current best results
best_score = final_score  # Initial best score is the score using all features
optimal_feature_set = all_features  # Initial optimal feature set is all features
current_features = all_features.copy()

# Iteration flag
improved = True  # Flag to mark if performance is still improving
iteration = 0

# Continue loop while feature count > 1 and last round showed improvement
while improved and len(current_features) > 1:

    start_time_iter = time.time()
    iteration += 1
    improved = False

    candidate_scores = {}  # Store scores after removing each feature

    print(f"\n--- Iteration {iteration}: Current features {len(current_features)}, Baseline F1: {best_score:.4f} ---")

    # Inner loop: Evaluate removing each feature one by one
    # This is the core of the wrapper method, computationally expensive
    for feature_to_remove in current_features:
        # 1. Construct candidate subset
        candidate_set = [f for f in current_features if f != feature_to_remove]

        # 2. Classify candidate subset, for creating preprocessor
        # Must redefine in each iteration, as feature list changes
        temp_numeric = [f for f in numeric_features if f in candidate_set]
        temp_binary = [f for f in binary_features if f in candidate_set]
        temp_categorical = [f for f in categorical_features if f in candidate_set]

        # 3. Recreate preprocessing pipeline
        temp_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), temp_numeric),
                ('bin', 'passthrough', temp_binary),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), temp_categorical)
            ],
            remainder='drop'
        )

        # 4. Prepare Data: Slice data using RAW feature set
        # Note: X_train[candidate_set] is selecting columns on the original DataFrame
        X_train_t = temp_preprocessor.fit_transform(X_train[candidate_set])
        X_val_t = temp_preprocessor.transform(X_val[candidate_set])

        # 5. Train model (using best parameters from tuning)
        model_be = XGBClassifier(**best_params)
        model_be.fit(X_train_t, y_train)

        # 6. Evaluate
        y_pred_be = model_be.predict(X_val_t)
        score_be = f1_score(y_val, y_pred_be, average='macro')

        candidate_scores[feature_to_remove] = score_be

        # print(f"  Testing removal of {feature_to_remove:<30}: Macro F1 = {score_be:.4f}")

    # 7. Find the best-scoring candidate subset in this iteration
    # (i.e., removing which feature results in the best model performance)
    best_candidate = max(candidate_scores.items(), key=lambda item: item[1])
    feature_removed_this_iter = best_candidate[0]
    best_score_this_iter = best_candidate[1]

    print(f"--- Iteration {iteration} Summary (Time: {time.time() - start_time_iter:.2f}s) ---")
    print(f"Best score this round: {best_score_this_iter:.4f} (by removing feature '{feature_removed_this_iter}')")

    # 8. Strict Backward Elimination Decision: Remove only if performance improves
    if best_score_this_iter > best_score:
        best_score = best_score_this_iter
        # Update current feature set, remove that "best" removed feature
        current_features = [f for f in current_features if f != feature_removed_this_iter]
        optimal_feature_set = current_features.copy()  # Update optimal feature set
        improved = True
        print(f"**Decision: Remove '{feature_removed_this_iter}'. New optimal feature count: {len(optimal_feature_set)}**")
    else:
        # If removing any feature does not improve performance (or is equal), then stop
        improved = False
        print("**Decision: Performance did not improve or was equal, stopping backward elimination.**")

# --- Section 6.2 Metrics ---
# (Intermediate metrics removed)
# ---------------------------

print("\nStrict Backward Elimination complete.")
print(f"Final optimal F1 score: {best_score:.4f}")
print(f"Optimal feature set count (original): {len(optimal_feature_set)}")

# Reclassify optimal feature set, for the final preprocessor
optimal_numeric_features = [f for f in numeric_features if f in optimal_feature_set]
optimal_binary_features = [f for f in binary_features if f in optimal_feature_set]
optimal_categorical_features = [f for f in categorical_features if f in optimal_feature_set]

# Define final preprocessor and data
final_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), optimal_numeric_features),
        ('bin', 'passthrough', optimal_binary_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), optimal_categorical_features)
    ],
    remainder='drop'
)

# Prepare final training data from original DataFrame using optimal feature set
X_final = train_df[optimal_feature_set]
# Fit the final preprocessor on the full dataset
X_full_transformed_final = final_preprocessor.fit_transform(X_final)

# Prepare final test set data
X_test_final = test_df[optimal_feature_set]

# =============================================================================
# 7. Full Data Training and Test Set Prediction
# =============================================================================

print("\n=== 7. Final Prediction ===")

# Retrain final model using the optimal feature set
print(f"Retraining final model using optimal feature set: {final_model_name}...")
final_model_optimized = XGBClassifier(**best_params)
# Train using the full data transformed with the optimal feature set (X_full_transformed_final) and all labels (y)
final_model_optimized.fit(X_full_transformed_final, y)

# --- Final Train Metrics ---
final_train_end_time = time.time()
# (Intermediate metrics removed)
# ---------------------------

# Test Set Prediction
print("Making test set predictions...")
# Transform test set using the final preprocessor
X_test_transformed = final_preprocessor.transform(X_test_final)

# --- Test Predict Timer Start ---
prediction_start_time = time.time()
test_preds = final_model_optimized.predict(X_test_transformed)
# --- Test Predict Timer End ---
prediction_end_time = time.time()


# Generate submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'label': test_preds
})

submission.to_csv('sample_submission.csv', index=False)

print("✓ Submission file 'sample_submission.csv' generated!")

# -----------------------------------------------------------------------------
# Script End: Final Metrics
# -----------------------------------------------------------------------------
print("\n" + "=" * 50)
print(" Optimized training flow complete!")

# Calculate and print final metrics
total_training_time = final_train_end_time - total_start_time
prediction_time = prediction_end_time - prediction_start_time
print(f" Total Training Time: {total_training_time:.2f} seconds")
print(f" Prediction Time: {prediction_time:.2f} seconds")

# Final memory usage
current_mem, peak_mem = tracemalloc.get_traced_memory()
print(f" Peak Memory Usage: {peak_mem / 1024**2:.2f} MB")
tracemalloc.stop() # Stop memory tracking

print("=" * 50)