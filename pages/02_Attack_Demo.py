import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier

st.title("Adversarial Attack Demo")
st.write("### Demonstration of a Simple Adversarial Evasion Attack using ART")

st.markdown("**Overview**: This page showcases a minimal example of an adversarial attack on a simple logistic regression classifier. "
            "We use the Fast Gradient Method (FGM) from the Adversarial Robustness Toolbox (ART) library to generate adversarial samples. "
            "Adjust the perturbation level (epsilon) below to see how even slight changes can have a large impact on the model's predictions.")

# Generate synthetic 2D data
np.random.seed(42)
num_samples = 200
x1 = np.random.normal(loc=0, scale=1, size=num_samples)
x2 = x1 * 1.2 + np.random.normal(loc=0, scale=1, size=num_samples)
X = np.column_stack((x1, x2))
y = (x1 + x2 > 0).astype(int)  # Simple decision boundary

# Train a logistic regression classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Wrap the trained model in an ART SklearnClassifier
classifier = SklearnClassifier(model=model)

# User-specified epsilon for adversarial attack
epsilon = st.slider("FGM Epsilon (Attack Strength)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)

# Instantiate and apply the Fast Gradient Method attack
attack = FastGradientMethod(estimator=classifier, eps=epsilon)
X_test_adv = attack.generate(x=X_test)

# Evaluate model on both clean and adversarial samples
clean_preds = model.predict(X_test)
adv_preds = model.predict(X_test_adv)

accuracy_clean = (clean_preds == y_test).mean()
accuracy_adv = (adv_preds == y_test).mean()

st.write(f"**Accuracy on clean test set**: {accuracy_clean:.2f}")
st.write(f"**Accuracy on adversarial test set**: {accuracy_adv:.2f}")

st.markdown("Below is a 2D visualization of a subset of the clean and adversarial samples. "
            "Points are colored by the predicted class of the logistic regression model.")

subset_size = 50
X_vis = X_test[:subset_size]
X_vis_adv = X_test_adv[:subset_size]
pred_clean_vis = model.predict(X_vis)
pred_adv_vis = model.predict(X_vis_adv)

df_vis = {
    "X1": np.concatenate((X_vis[:, 0], X_vis_adv[:, 0])),
    "X2": np.concatenate((X_vis[:, 1], X_vis_adv[:, 1])),
    "Type": ["Clean"] * subset_size + ["Adversarial"] * subset_size,
    "Predicted Class": np.concatenate((pred_clean_vis, pred_adv_vis)),
}

fig = px.scatter(df_vis,
                 x="X1",
                 y="X2",
                 color="Predicted Class",
                 symbol="Type",
                 title="Clean vs. Adversarial Samples (subset)",
                 labels={"X1": "Feature 1", "X2": "Feature 2"})
st.plotly_chart(fig, use_container_width=True)

st.markdown("**Key Takeaway**: Notice how a small perturbation (controlled by epsilon) can drastically change the model's predictions, "
            "underscoring the vulnerability of machine learning models to adversarial attacks.")
