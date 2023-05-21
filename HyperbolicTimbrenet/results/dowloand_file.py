import urllib.request

# Downloading baseline model weights
url_index = "https://www.dropbox.com/s/x6n4c8k0eq9t68h/baseline_latent_2_lr_3e-05_b_1_the_best.index"
url_data = "https://www.dropbox.com/s/mgldawvoqv6tfmb/baseline_latent_2_lr_3e-05_b_1_the_best.data-00000-of-00001"
file_index = "baseline/model_weights/baseline_latent_2_lr_3e-05_b_1_the_best.index"
file_data = "baseline/model_weights/baseline_latent_2_lr_3e-05_b_1_the_best.data-00000-of-00001"
urllib.request.urlretrieve(url_index, file_index)
urllib.request.urlretrieve(url_data, file_data)

# Downloading baseline_hyp model weights
url_index = "https://www.dropbox.com/s/jncdm9b58uwqo93/baseline_hyperbolic_latent_2_lr_3e-05_b_1_the_best.index"
url_data = "https://www.dropbox.com/s/yjg1l77ymndkwpg/baseline_hyperbolic_latent_2_lr_3e-05_b_1_the_best.data-00000-of-00001"
file_index = "baseline_hyp/model_weights/baseline_hyperbolic_latent_2_lr_3e-05_b_1_the_best.index"
file_data = "baseline_hyp/model_weights/baseline_hyperbolic_latent_2_lr_3e-05_b_1_the_best.data-00000-of-00001"
urllib.request.urlretrieve(url_index, file_index)
urllib.request.urlretrieve(url_data, file_data)

# Downloading breguel_model model weights
url_index = "https://www.dropbox.com/s/vdxggzjyl76niiv/breguel_model_latent_2_lr_3e-05_b_1_the_best.index"
url_data = "https://www.dropbox.com/s/wpnumej64mc290d/breguel_model_latent_2_lr_3e-05_b_1_the_best.data-00000-of-00001"
file_index = "breguel_model/model_weights/breguel_model_latent_2_lr_3e-05_b_1_the_best.index"
file_data = "breguel_model/model_weights/breguel_model_latent_2_lr_3e-05_b_1_the_best.data-00000-of-00001"
urllib.request.urlretrieve(url_index, file_index)
urllib.request.urlretrieve(url_data, file_data)

# Downloading mircea_model model weights
url_index = "https://www.dropbox.com/s/83fuwtawrcc3br6/mircea_model_latent_2_lr_3e-05_b_1_the_best.index"
url_data = "https://www.dropbox.com/s/p2kaizm4pwyqkr6/mircea_model_latent_2_lr_3e-05_b_1_the_best.data-00000-of-00001"
file_index = "mircea_model/model_weights/mircea_model_latent_2_lr_3e-05_b_1_the_best.index"
file_data = "mircea_model/model_weights/mircea_model_latent_2_lr_3e-05_b_1_the_best.data-00000-of-00001"
urllib.request.urlretrieve(url_index, file_index)
urllib.request.urlretrieve(url_data, file_data)

# Downloading hyp+vae model weights
url_index = "https://www.dropbox.com/s/3gg0h57boq5uhf9/Hyp%2BCNN_latent_2_lr_3e-05_b_1_the_best.index"
url_data = "https://www.dropbox.com/s/6xkn0af5a006d1x/Hyp%2BCNN_latent_2_lr_3e-05_b_1_the_best.data-00000-of-00001"
file_index = "hyp+vae/model_weights/hyp+vae_latent_2_lr_3e-05_b_1_the_best.index"
file_data = "hyp+vae/model_weights/hyp+vae_latent_2_lr_3e-05_b_1_the_best.data-00000-of-00001"
urllib.request.urlretrieve(url_index, file_index)
urllib.request.urlretrieve(url_data, file_data)
