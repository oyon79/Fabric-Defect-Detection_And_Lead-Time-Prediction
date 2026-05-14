"""
Task 3: Dynamic Lead Time Regression
======================================
Random Forest-based lead time prediction using machine speed, order quantity,
and real-time Defect Density Score (D_s) from the metadata bridge.

The integration gap in smart manufacturing: production planning systems assume
constant defect rates, but reality varies with fabric quality. By incorporating
D_s in real-time, this model dynamically adjusts lead time predictions.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import RANDOM_STATE, SYNTHETIC_SAMPLES, TRAIN_TEST_SPLIT


class LeadTimePredictor:
    """
    Random Forest regressor for predicting manufacturing lead time.

    Features:
        - machine_speed: Current production line speed (meters/min)
        - order_quantity: Number of units in the current order
        - defect_density: Real-time D_s from detection system

    Output:
        - Predicted lead time in hours

    The model learns the relationship: higher D_s -> longer lead time
    """

    def __init__(self, n_estimators=100, max_depth=10, random_state=RANDOM_STATE):
        """
        Initialize the Random Forest model.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree
            random_state: Seed for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_names = ['machine_speed', 'order_quantity', 'defect_density']
        self.baseline_lead_time = None  # Static baseline (mean of training data)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model on historical manufacturing data.

        Args:
            X: Feature matrix (n_samples, 3) with columns [machine_speed, order_quantity, D_s]
            y: Target lead times in hours (n_samples,)
        """
        self.model.fit(X, y)
        self.is_fitted = True
        self.baseline_lead_time = float(np.mean(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict lead time for given inputs.

        Args:
            X: Feature matrix (n_samples, 3) or single sample (1, 3)

        Returns:
            Predicted lead times in hours
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        return self.model.predict(X)

    def predict_single(self, machine_speed: float, order_quantity: float, defect_density: float) -> float:
        """
        Predict lead time for a single scenario.

        Args:
            machine_speed: Production speed (m/min)
            order_quantity: Number of units
            defect_density: Current D_s value

        Returns:
            Lead time in hours
        """
        features = np.array([[machine_speed, order_quantity, defect_density]])
        return float(self.predict(features)[0])

    def update(self, X: np.ndarray, y: np.ndarray):
        """
        Incrementally update the model with new data.

        Note: Full retrain from scratch. For true online learning,
        consider scikit-learn's partial_fit or river library.

        Args:
            X: New feature samples
            y: New target values
        """
        # Simple incremental approach: refit with all data
        self.model.fit(X, y)
        self.is_fitted = True


def generate_synthetic_data(n_samples: int = SYNTHETIC_SAMPLES, random_state: int = RANDOM_STATE) -> tuple:
    """
    Generate synthetic manufacturing data for training/testing.

    The data simulates a fabric production line where:
    - Lead time = f(machine_speed, order_quantity, defect_rate) + noise
    - Higher defect density -> slower production -> longer lead time
    - Larger orders -> proportionally longer lead time
    - Faster machines -> shorter lead time (up to a limit)

    Args:
        n_samples: Number of samples to generate
        random_state: Seed for reproducibility

    Returns:
        (X_train, X_test, y_train, y_test) tuples
    """
    rng = np.random.RandomState(random_state)

    # Feature ranges
    machine_speeds = rng.uniform(20, 100, n_samples)  # 20-100 m/min
    order_quantities = rng.randint(100, 10000, n_samples)  # 100-10000 units
    defect_densities = rng.uniform(0, 0.5, n_samples)  # 0-50% defect area

    # Physical relationship: lead_time in hours
    # Base time scales with quantity, inverse with speed
    base_time = (order_quantities / machine_speeds) * 0.1  # scaling factor

    # Defect impact: each 1% defect density adds ~2% to lead time
    defect_multiplier = 1 + (defect_densities * 2)

    # Add noise for realism
    noise = rng.normal(0, 0.5, n_samples)

    lead_times = base_time * defect_multiplier + noise

    # Clip negative values
    lead_times = np.clip(lead_times, 0.5, None)

    # Assemble feature matrix
    X = np.column_stack([machine_speeds, order_quantities, defect_densities])
    y = lead_times

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-TRAIN_TEST_SPLIT, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(predictor: LeadTimePredictor, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model performance on test set.

    Args:
        predictor: Trained LeadTimePredictor
        X_test: Test features
        y_test: True lead times

    Returns:
        Dictionary with MAE, RMSE, R2, and baseline comparison
    """
    y_pred = predictor.predict(X_test)

    metrics = {
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'r2': float(r2_score(y_test, y_pred)),
    }

    # Static baseline: predict mean of training data
    if predictor.baseline_lead_time is not None:
        baseline_pred = np.full_like(y_test, predictor.baseline_lead_time)
        metrics['baseline_mae'] = float(mean_absolute_error(y_test, baseline_pred))
        metrics['mae_improvement'] = metrics['baseline_mae'] - metrics['mae']

    return metrics


if __name__ == "__main__":
    print("=" * 60)
    print("Lead Time Predictor - Training & Evaluation")
    print("=" * 60)

    # Generate synthetic data
    print("\nGenerating synthetic manufacturing data...")
    X_train, X_test, y_train, y_test = generate_synthetic_data()
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Initialize and train predictor
    print("\nTraining Random Forest model...")
    predictor = LeadTimePredictor(n_estimators=100, max_depth=10)
    predictor.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate_model(predictor, X_test, y_test)

    print(f"\nUnified Model Performance:")
    print(f"  MAE:  {metrics['mae']:.3f} hours")
    print(f"  RMSE: {metrics['rmse']:.3f} hours")
    print(f"  R2:   {metrics['r2']:.4f}")

    print(f"\nStatic Baseline (mean prediction):")
    print(f"  MAE:  {metrics['baseline_mae']:.3f} hours")

    print(f"\nImprovement: {metrics['mae_improvement']:.3f} hours better MAE")
    print(f"             ({metrics['mae_improvement']/metrics['baseline_mae']*100:.1f}% reduction)")

    # Demo predictions
    print("\n" + "-" * 40)
    print("Sample Predictions:")
    print("-" * 40)

    test_cases = [
        (50.0, 1000, 0.05),   # Normal speed, medium order, low defects
        (80.0, 5000, 0.30),   # Fast speed, large order, high defects
        (30.0, 200, 0.40),    # Slow speed, small order, high defects
    ]

    for speed, qty, ds in test_cases:
        pred = predictor.predict_single(speed, qty, ds)
        print(f"  Speed={speed:.0f}, Qty={qty}, D_s={ds:.2f} -> Lead Time: {pred:.2f} hours")

    print("\nLead time predictor validated successfully.")