# Car Price Prediction API Documentation

## Data Preprocessing Module (`src.utils.data_preprocessing`)

### `load_and_validate_car_data(filepath: str) -> pd.DataFrame`
Loads and validates automotive dataset from CSV file.

**Parameters:**
- `filepath`: Path to the car data CSV file

**Returns:**
- Validated pandas DataFrame with car features

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If validation fails

### `encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame`
Encodes categorical variables for ML models.

**Parameters:**
- `data`: DataFrame with categorical variables

**Returns:**
- DataFrame with encoded categorical features

### `prepare_features_target(data: pd.DataFrame, target_column: str, drop_columns: list) -> Tuple[pd.DataFrame, pd.Series]`
Prepares features and target for model training.

## Model Training Module (`src.utils.model_training`)

### `CarPricePredictor`
Main class for car price prediction with dual algorithm support.

#### Methods:
- `train_models(X_train, y_train)`: Train Linear and Lasso regression models
- `evaluate_models(X_test, y_test)`: Evaluate models on test data
- `save_models(filepath_prefix)`: Save trained models to disk
- `load_models(filepath_prefix)`: Load models from disk

## Visualization Module (`src.utils.visualization`)

### `CarPriceVisualizer`
Comprehensive visualization suite for automotive price analysis.

#### Methods:
- `plot_actual_vs_predicted()`: Scatter plot of predictions vs actual
- `plot_model_comparison()`: Compare multiple model performances
- `plot_feature_importance()`: Display feature importance
- `plot_residuals()`: Residuals analysis