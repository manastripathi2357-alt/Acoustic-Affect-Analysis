# Acoustic-Affect-Analysis
An end-to-end Machine Learning system that integrates speaker diarization and feature extraction to accurately classify human emotions from raw speech signals. This production-ready architecture was originally engineered and utilized as the gold standard for a comprehensive 7-week technical curriculum.
Key Features & Architecture


Intelligent Data Processing: Processes the 1,440-file RAVDESS dataset while programmatically filtering out OS-level corruption and metadata files to prevent data leakage. Implements Pickle-based caching to heavily optimize extraction speeds.


Acoustic Feature Extraction: Computes time-axis means across 16kHz downsampled audio to extract a static, 193-dimensional feature vector. This array captures the physiological and harmonic shape of the voice using 40 MFCCs, 128 Mel-Spectrogram bands, 12 Chroma STFTs, 7 Spectral Contrast bands, and 6 Tonnetz features.


Speaker Diarization: Isolates individual speakers by segmenting the audio into 20-30ms frames using precise Voice Activity Detection (VAD). It groups distinct voices by adapting a Universal Background Model (GMM-UBM) via MAP estimation and applying Cosine Spectral Clustering.


Neural Network Classification: Trains a Multi-Layer Perceptron (MLP) built with a 400-neuron input layer to act as an expansion funnel for the acoustic features, utilizing Dropout regularization to prevent overfitting. Performance is validated and compared against an optimized, tree-based XGBoost classifier.


Unified Inference: Features a fully autonomous pipeline that digests raw, multi-speaker audio and outputs a human-readable timeline predicting 8 distinct emotional states.
