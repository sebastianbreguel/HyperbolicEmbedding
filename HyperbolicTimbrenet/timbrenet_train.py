import datetime
import os
import time

import numpy as np
import tensorflow as tf
from lib.models.baseline import CVAE as Model
from lib.models.baseline_hyp import HCVAE as HModel
from lib.models.breguel_model import HVAE_BREGUEL as HModel_breguel
from lib.models.hyp_vae import EHYP_VAE as EModel
from lib.models.mircea_model import M_VAE as HModel_new
from lib.specgrams_helper import SpecgramsHelper
from scipy.io.wavfile import read as read_wav


def import_audio(filename):
    audio = np.array([read_wav(filename)[1]], dtype=float)
    return audio


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi + 1e-10)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


def compute_loss(model, x, beta):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    MSE = mse(x, x_logit)
    logpx_z = -tf.reduce_sum(MSE, axis=[1, 2])
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + beta * (logpz - logqz_x))


def compute_apply_gradients(model, x, optimizer, beta):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, beta)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train_model(
    latent_dim,
    dataset_path,
    instruments,
    chords,
    volumes,
    examples,
    epochs,
    beta,
    learning_rate,
    optimizer,
    model=1,
    baseline=True,
    hyperbolic=False,
    mircea_model=False,
    gpu=True,
):
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print(tf.config.list_physical_devices("GPU"))

    spec_helper = SpecgramsHelper(
        audio_length=64000,
        spec_shape=(128, 1024),
        overlap=0.75,
        sample_rate=16000,
        mel_downscale=1,
    )

    # IMPORT AND CONVERT AUDIO TO MEL_SPECTROGRAMS
    print("\n\nImporting Dataset...")

    num_examples = len(instruments) * len(chords) * len(volumes) * len(examples)
    print("Number of examples: " + str(num_examples))

    audio_matrix = np.zeros([num_examples, 64000, 1])
    n = 0
    for instrument in instruments:
        for chord in chords:
            for volume in volumes:
                for example in examples:
                    a = import_audio(
                        dataset_path + instrument + chord + volume + example + ".wav"
                    )[0, :]
                    try:
                        audio_matrix[n, :, 0] = a
                    except:
                        print(
                            "\nError en:  "
                            + str(n)
                            + "  "
                            + str(a.shape)
                            + "  "
                            + dataset_path
                            + instrument
                            + chord
                            + volume
                            + example
                            + ".wav"
                        )
                    n = n + 1

    print("Success Importing Dataset!\n")

    print("\n\nConverting to mel spectrograms...")
    mel = spec_helper.waves_to_melspecgrams(audio_matrix)
    mel = tf.random.shuffle(mel, seed=21)
    melA = mel[0:450, :, :, 0] / 13.82  # /13.815511
    melB = mel[0:450, :, :, 1] / 1.00001
    mel = tf.stack([melA, melB], axis=-1)
    print(mel.shape)
    print("Success converting to mel spectrograms!\n")

    print("\n\nPreparing train and test dataset...")
    train_melgrams = mel[0 : num_examples - 50, :, :, :]
    test_melgrams = mel[num_examples - 50 : num_examples, :, :, :]
    TRAIN_BUF = num_examples - 50
    BATCH_SIZE = 5
    TEST_BUF = 50
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_melgrams)
        .shuffle(TRAIN_BUF, seed=21)
        .batch(BATCH_SIZE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices(test_melgrams)
        .shuffle(TEST_BUF, seed=21)
        .batch(BATCH_SIZE)
    )
    print("Success preparing train and test dataset!")

    model_save = "./results"
    if model == 1:
        file = "baseline_latent_2_lr_3e-05_b_1_the_best"
        model_save += "/baseline"
        model = Model(latent_dim)

    elif model == 2:
        model_save += "/baseline_hyp"
        file = "baseline_hyperbolic_latent_2_lr_3e-05_b_1_the_best"
        model = HModel(latent_dim)

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

    model.inference_net.summary()
    model.generative_net.summary()

    print("New Model")
    description = (
        "_mel_p0_latent_"
        + str(latent_dim)
        + "_lr_"
        + str(learning_rate)
        + "_b_"
        + str(beta)
    )
    day = datetime.datetime.now().strftime("%Y_%m_%d")
    time_clock = datetime.datetime.now().strftime("%H_%M_%S")

    # Create saving variables    x
    train_log_dir = (
        "logs/gradient_tape/" + day + "/" + time_clock + description + "/train"
    )
    test_log_dir = (
        "logs/gradient_tape/" + day + "/" + time_clock + description + "/test"
    )
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    model_save += "/model_weights/" + file

    # model.load_weights(model_save)

    # Train
    best_elbo = -1e20
    model_save = "./model_weights/" + day + "/" + time_clock + description
    start_epoch = 1

    for epoch in range(start_epoch, start_epoch + epochs):
        start_time = time.time()
        train_loss = tf.keras.metrics.Mean()
        id = 1
        intermedio = start_time
        for train_x in train_dataset:
            train_loss(compute_apply_gradients(model, train_x, optimizer, beta))
            print(
                f"  - {id} -> {id}/{len(train_dataset)}  {id/len(train_dataset)}% total: {time.time()- start_time} segundos, epoca: {time.time()-intermedio} segundos",
                end="\r",
            )
            id = id + 1
            intermedio = time.time()
        end_time = time.time()

        test_loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            test_loss(compute_loss(model, test_x, beta))
        elbo = -test_loss.result()
        # test acurracy of reconstruction of test data

        train_elbo = -train_loss.result()

        with test_summary_writer.as_default():
            tf.summary.scalar("Test ELBO", -elbo, step=epoch)
        with train_summary_writer.as_default():
            tf.summary.scalar("Train ELBO", -train_elbo, step=epoch)

        print(
            "Epoch: {}, Test set ELBO: {}, Train set ELBO: {}, time elapse for current epoch {}".format(
                epoch, elbo, train_elbo, end_time - start_time
            )
        )

        # if elbo > best_elbo:
        #     print("Model saved:")
        #     best_elbo = elbo
        #     model.save_weights(
        #         model_save
        #         + "_se_"
        #         + str(1)
        #         + "_ee_"
        #         + str(epochs + start_epoch)
        #         + "_ep_"
        #         + str(epoch)
        #     )
        #     model.save_weights(model_save + "_the_best")


if __name__ == "__main__":
    # Select latent dimension
    latent_dim = 2
    # latent_dim = 8

    # Select datasetr path
    dataset_path = "./datasets/450pianoChordDataset/audio/"

    # Select elements of dataset to plot
    instruments = ["piano_"]
    chords = [
        "C2_",
        "Dm2_",
        "Em2_",
        "F2_",
        "G2_",
        "Am2_",
        "Bdim2_",
        "C3_",
        "Dm3_",
        "Em3_",
        "F3_",
        "G3_",
        "Am3_",
        "Bdim3_",
        "C4_",
    ]
    volumes = ["f_", "m_", "p_"]
    examples = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # Select training params
    epochs = 500
    beta = 1
    learning_rate = 3e-5
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_model(
        latent_dim,
        dataset_path,
        instruments,
        chords,
        volumes,
        examples,
        epochs,
        beta,
        learning_rate,
        optimizer,
        model=2,
        gpu=True,
    )


# model 1 is baseline
# model 2 is baseline hyperbolic
# model 3 is mircea
# model 4 is breguel
# model 5 is timbreNet + embedding hyperbolico
