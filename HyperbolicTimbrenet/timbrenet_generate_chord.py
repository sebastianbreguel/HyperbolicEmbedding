import os
import tensorflow as tf
from scipy.io.wavfile import write
from lib.models.baseline import CVAE as Model
from lib.models.baseline_hyp import HCVAE as HModel
from lib.models.hyp_vae import EHYP_VAE as EModel
from lib.models.breguel_model import HVAE_BREGUEL as HModel_breguel
from lib.models.mircea_model import M_VAE as HModel_new
from lib.latent_chord import latent_chord
from lib.specgrams_helper import SpecgramsHelper


def generate_chord_from_trained_model(
    latent_dim, sample_points, chord_saving_path, model
):
    if not os.path.exists(chord_saving_path):
        os.makedirs(chord_saving_path)

    spec_helper = SpecgramsHelper(
        audio_length=64000,
        spec_shape=(128, 1024),
        overlap=0.75,
        sample_rate=16000,
        mel_downscale=1,
    )

    model_save = "./results"
    if model == 1:
        model_save += "/baseline_hyp"
        file = "baseline_hyperbolic_latent_2_lr_3e-05_b_1_the_best"
        model = HModel(latent_dim)

    elif model == 2:
        file = "baseline_latent_2_lr_3e-05_b_1_the_best"
        model_save += "/baseline"
        model = Model(latent_dim)

    elif model == 3:
        file = "mircea_model_latent_2_lr_3e-05_b_1_the_best"
        model_save += "/mircea_model"
        model = HModel_new(latent_dim)

    elif model == 4:
        file = "breguel_model_latent_2_lr_3e-05_b_1_the_best"
        model_save += "/breguel_model"
        model = HModel_breguel(latent_dim)

    elif model == 5:
        file = "hyp+vae_latent_2_lr_3e-05_b_1_the_best"
        model_save += "/hyp+vae"
        model = EModel(latent_dim)

    model_save += "/model_weights/" + file
    print("\n\nLoading Trained Model...")
    model.load_weights(model_save)
    print("Success Loading Trained Model!\n")

    n = 1
    for sample_point in sample_points:
        chord = latent_chord(
            tf.constant([sample_point], dtype="float32"), model, spec_helper
        )
        write(
            chord_saving_path + "chord" + str(n) + ".wav", data=chord.audio, rate=16000
        )
        print("Chord " + str(n) + " generated!")
        n += 1
    print(
        "\n\nSUCCESS: ALL CHORDS GENERATED!    (chords are saved at "
        + chord_saving_path
        + ")"
    )


if __name__ == "_main_":
    # Select trained model path
    trained_model_path = (
        "./trained_models/450_piano_chords/latent_2_lr_3e-05_epoch_385_of_501"
    )
    # trained_model_path = './trained_models/450_piano_chords/latent_8_lr_3e-05_epoch_141_of_501'

    # Select latent dimension
    latent_dim = 2
    # latent_dim = 8

    # Select sample points
    # sample_points = [[7, 8], [18, -18], [18, -7], [7, -30], [39, -10], [17, 10]]
    sample_points = [[35, -12]]
    """
    sample_points = [[11.7 , 8.9, 12.8, 16.2,- 2.6,- 4.3,- 9.1, 21.0],
                    [- 8.0 , 9.6,-23.6, 20.0, 13.5,  8.0,-14.6,  3.1],
                    [-11.6 , 5.9,- 9.0,- 0.5,-25.4,-15.3,  3.1,  4.9],
                    [  6.3 , 3.9,  2.1,  9.1,-16.4,-13.8,- 1.8, 10.9]]
                    """

    # Select path for saving chords
    chord_saving_path = "./generated_chords/"

    generate_chord_from_trained_model(latent_dim, sample_points, chord_saving_path, 2)
