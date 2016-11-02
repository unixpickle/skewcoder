# Skewed auto-encoders

In a traditional neural auto-encoder, an input vector is compressed to a feature vector, and then the original input is reconstructed from that feature vector. In such a model, every component of the feature vector is equally "important"--there is no priority about which feature stores the most information. The goal of this project is to create a prioritized auto-encoder which generates skewed feature vectors (i.e. different components have different priorities).

# How it works

The idea is that an LSTM (or any RNN for that matter) learns to focus on recent inputs more than inputs in the distant past. As a result, feeding a large vector into an LSTM component-by-component naturally introduces a priority ordering: the inputs closer to the last input are the most "important" when we use the last timestep's output as the reconstruction vector.
