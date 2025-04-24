import tensorflow as tf
from model import TemperatureScaledModel

def tune_temperature(model, X_val, y_val, epochs=100, verbose=True):
    """
    Performs temperature scaling on a trained model to calibrate output probabilities.

    Args:
        model (tf.keras.Model): Trained base model.
        X_val (np.ndarray): Validation input data.
        y_val (np.ndarray): Ground truth labels.
        epochs (int): Number of optimization steps.
        verbose (bool): Whether to print loss during optimization.

    Returns:
        float: Optimized temperature value.
    """
    temperature = tf.Variable(1.0, trainable=True, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    logits = model.predict(X_val)

    for step in range(epochs):
        with tf.GradientTape() as tape:
            scaled_logits = logits / tf.maximum(temperature, 1e-2)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_val, tf.nn.softmax(scaled_logits)))
        grads = tape.gradient(loss, [temperature])
        optimizer.apply_gradients(zip(grads, [temperature]))

        temperature.assign(tf.clip_by_value(temperature, 0.5, 10.0))

        if verbose and step % 10 == 0:
            print(f"Step {step:3d} | T = {temperature.numpy():.4f} | NLL = {loss.numpy():.4f}")

    return temperature.numpy()

def apply_temperature_scaling(models, predictions):
    """
    Applies temperature scaling calibration to a trained MMCNN cascade.

    For each level (L1, L2, L3), this function calibrates the output probabilities
    of the respective models using validation data and returns a new dictionary of
    temperature-scaled models.

    Args:
        models (dict): Dictionary containing trained models at levels L1, L2, and L3.
            - models['L1']: Base level model
            - models['L2']: Dictionary of L2 models, keyed by L1 quadrant (1-4)
            - models['L3']: Dictionary of L3 models, keyed by (L1, L2) quadrant pairs
        predictions (dict): Dictionary containing validation input and labels per model.
            - predictions['L1']: Tuple of (X_val, y_val) for Level 1
            - predictions['L2']: Dict of (X_val, y_val) for each L2 model
            - predictions['L3']: Dict of (X_val, y_val) for each L3 model

    Returns:
        dict: Dictionary of calibrated models using TemperatureScaledModel.
            - Same structure as the input `models` dict
    """
    # Initialize output dictionary for calibrated models
    calibrated_models = {
        'L1': None,
        'L2': {},
        'L3': {}
    }

    # --- Level 1 Calibration ---
    X1_val, y1_val = predictions['L1'][0], predictions['L1'][1]
    T1 = tune_temperature(models['L1'], X1_val, y1_val)  # Optimize temperature for L1
    calibrated_models['L1'] = TemperatureScaledModel(models['L1'], init_temperature=T1)

    # --- Level 2 Calibration ---
    for l1 in range(1, 5):
        if l1 not in models['L2']:
            continue
        X2_val, y2_val = predictions['L2'][l1][0], predictions['L2'][l1][1]
        T2 = tune_temperature(models['L2'][l1], X2_val, y2_val)  # Optimize temperature for L2 quadrant
        calibrated_models['L2'][l1] = TemperatureScaledModel(models['L2'][l1], init_temperature=T2)

    # --- Level 3 Calibration ---
    for (l1, l2) in models['L3']:
        X3_val, y3_val = predictions['L3'][(l1, l2)][0], predictions['L3'][(l1, l2)][1]
        T3 = tune_temperature(models['L3'][(l1, l2)], X3_val, y3_val)  # Optimize temperature for L3 quadrant pair
        calibrated_models['L3'][(l1, l2)] = TemperatureScaledModel(models['L3'][(l1, l2)], init_temperature=T3)

    return calibrated_models

