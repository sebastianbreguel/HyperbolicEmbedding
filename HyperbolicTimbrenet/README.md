# TimbreNet

This project aims to create deep learning tools for musicians to work with the timbre of different sounds and instruments.

```bash

|__📜timbrenet_generate_chord.py
|
|__📜timbrenet_generate_latent_map.py
|
|__📜timbrenet_train.py
|
|
|__📂datasets/450pianoChordDataset
|   |__📂audio
|   |__📜README.md
|
|__📂generated_chords
|
|__📂lib
|  |__📜latent_chord.py
|  |__📜model.py
|  |__📜specgrams_helper.py
|  |__📜spectral_ops.py
|  |__📂hyp_model           # hyperbolic layers
|  |   |__📜__init_.py
|  |   |__📜linear_hyp.py
|  |   |__📜manifold.py
|  |   |__📜util.py
|  |
|  |__📂models          # models
|  |   |__📜__init_.py
|  |   |__📜baseline.py
|  |   |__📜baseline_hyp.py
|  |   |__📜breguel_model.py
|  |   |__📜hyp_vae.py
|  |   |__📜mircea_model.py
|
|__📂logs/gradient_tape/2022_12_09 #cache for training model
|
|__📂results
|  |   |__📜Dowloand_file.ipynb    # dowloand the weights of the models
|  |   |__📂baseline
|  |   |__📂baseline_hyp
|  |   |__📂breguel_model
|  |   |__📂mircea_model   # * all the models have the same structure
|  |   |   |__📂model_weights
|  |   |   |__📜loss.txt
|  |   |__📂analisis
|  |   |   |__📂sounds # audios of the generated chords of the top two models
|  |   |   |__📜architectures.jpeg #  visual architecture of each model
|  |   |   |__📜losses.png # comparation of the losses of the models
|
|
|_📂trained_models/450_piano_chords    #pretrained models weights
```

- To run trained models you need to run the jupyter notebook Dowloand_file.ipynb, to dowloand the weight of each model from dropbox.

## Datasets

- PiancoChordDatastet: Dataset with 450 piano chords audios.

## Models

- TimbreNet_PianoChordVAE: VAE for encoding piano chords in a low-dimension latent space (2D - 3D). This latent space can be used to create new sounds of piano chords or create chord sequences by moving through the latent space.

## How to generate a chord from a trained model

- Clone this repository.
- Open timbrenet_generate_chord.py
- In the "trained_model_path" variable put the file with the weights of a trained model (there are some trained models in the "trained_models" folder.
- In the "latent_dim" variable select the latent dimention of the trained model.
- In the "sample_points" variable put all the poins from where you want to sample chords. You can samplle as many chords as you want at the same time. Each point needs to have the same ammount of dimentions as the "latent_dim" variable
- In the "chord_saving_path" put the path of the folder where you want to save the chords. If the folder does not exist, the code will create it for you.
- Finally, run timbrenet_generate_chord.py

## How to generate a 2D latent map

- Clone this repository.
- Open timbrenet_generate_latent_map.py
- In the "trained_model_path" variable put the file with the weights of a trained model (there are some trained models in the "trained_models" folder.
- In the "latent_dim" variable select the latent dimention of the trained model.
- In the "dataset_path" variable select the path of the dataset you want to plot.
- In the "instruments" select the instruments of the dataset you want to plot.
- In the "chords" select the chords of the dataset you want to plot.
- In the "volumes" select the volumes of the dataset you want to plot.
- In the "examples" select the examples of the dataset you want to plot.
- Finally, run timbrenet_generate_latent_map.py
