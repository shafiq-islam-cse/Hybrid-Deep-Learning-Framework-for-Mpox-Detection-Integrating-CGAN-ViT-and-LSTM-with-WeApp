import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from PIL import Image
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Flatten

# Define the Attention Layer
class Attention(tf.keras.layers.Layer):
    def __init__(self, step_dim, W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None, bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint,
                                 name='{}_W'.format(self.name))
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint,
                                     name='{}_b'.format(self.name))
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            'step_dim': self.step_dim,
            'W_regularizer': regularizers.serialize(self.W_regularizer),
            'b_regularizer': regularizers.serialize(self.b_regularizer),
            'W_constraint': constraints.serialize(self.W_constraint),
            'b_constraint': constraints.serialize(self.b_constraint),
            'bias': self.bias
        })
        return config


# Cache LSTM Model
@st.cache_resource
def load_lstm_model():
    with CustomObjectScope({'Attention': Attention}):
        lstm_model = load_model("monkeypox_detection_model.keras")  # Path to LSTM model
    return lstm_model


# Cache ViT Model
@st.cache_resource
def load_vit_model():
    vit_model_path = r"C:\Users\suzon\.cache\kagglehub\models\spsayakpaul\deit\tensorFlow2\tiny-patch16-224-fe\1"

    # Load the ViT model directly as a Keras Layer
    base_model_vit = hub.KerasLayer(vit_model_path, trainable=False)

    return base_model_vit  # Return the model directly


def extract_features(image, feature_extractor_model):
    """Extract features from the image using ViT."""
    image = image.resize((224, 224))  # Resize to ViT input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Convert image to TensorFlow tensor
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

    # Extract features from ViT model
    features = feature_extractor_model(image_tensor)

    # If features is a tuple, extract the first element (features, logits)
    if isinstance(features, tuple):
        features = features[0]

    return features.numpy()  # Convert back to NumPy array


def predict_monkeypox(features, lstm_model):
    """Predict whether the image contains Monkeypox or not using LSTM."""
    features = np.array(features)  # Ensure it's a NumPy array

    # 🔥 Fix: Reshape the features to match LSTM input dimensions
    features = features.reshape(features.shape[0], 1, -1)  # Reshape for LSTM

    # Add a fully connected layer to project 192 to 768 if necessary
    if features.shape[-1] != 768:
        feature_projection = Dense(768, activation="relu")(features)
        features = Flatten()(feature_projection)  # Flatten the output of Dense layer

    # Reshape for LSTM: (batch_size, time_steps, features)
    features = np.expand_dims(features, axis=1)  # Ensure correct LSTM shape

    prediction = lstm_model.predict(features)
    return prediction[0][0]


# Streamlit UI
st.markdown("""
    <style>
    .title {
        font-size: 3em;
        color: #ff5733;
        text-align: center;
        padding: 20px;
        background-color: #f4f4f4;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    <div class="title">Monkeypox Detection</div>
""", unsafe_allow_html=True)

# Load models
lstm_model = load_lstm_model()
feature_extractor_model = load_vit_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image for Monkeypox detection", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Resize the image for the model
        resized_image = image.resize((224, 224))  # Resizing to 224x224

        # Display the resized image
        st.image(resized_image, caption='Resized Image (224x224)', use_container_width=True)

        # Extract features
        features = extract_features(resized_image, feature_extractor_model)

        # Predict using LSTM
        confidence = predict_monkeypox(features, lstm_model)

        # Display result
        if confidence > 0.8:
            st.markdown(
                f"""<h1 style="color: red;">Prediction: Monkeypox detected </h1>""",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"""<h1 style="color: green;">Prediction: No Monkeypox detected </h1>""",
                unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload an image to detect Monkeypox.")

