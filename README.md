# Exploring XAI for the Arts: Explaining Latent Space in Generative Music

Augmentation of Ashis Pati and Alexander Lerch's "Latent Space Regularization for Explicit Control of Musical Attributes" (2019) and "Attribute-based Regularization of Latent Spaces for Variational Auto-Encoders" (2020).

Implementation source:
https://github.com/ashispati/AttributeModelling.


# Web Demo Links:

https://xai-lsr-ui.vercel.app/

https://xai-no-lsr-ui.vercel.app/


# Python and Cuda Versions:
python/3.7.7       
cuda/10.2-cudnn8.0.5

# For the python packages:
pip install -r requirements.txt

# Dataset:

Download from: https://github.com/ashispati/AttributeModelling

Unzip the downloaded file and put the `datasets` and `folk_raw_data` folders under `data`.

# Generated Pianoroll and Audio Files with LSR for Demonstration:

They are named with their corresponding musical metrics levels (10 discrete level for each of the 4 metrics). For example, midi_3_4_5_3.mid means, this file has 3/10 rhythmic complexity, 4/10 note range, 5/10 note density and 3/10 average interval jump.
