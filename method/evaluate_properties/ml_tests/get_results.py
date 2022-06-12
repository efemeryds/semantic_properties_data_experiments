from logistic_regression import main_lr
from method.evaluate_properties.helper_functions import load_word2vec_model

experiment_name = 'logistic_regression'

# experiments = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

experiments = [10]

for exp in experiments:
    file_name = "black"
    path_to_model = f'../../add/learnt_embeddings/{file_name}_{exp}.bin'
    model_name = f'{file_name}_{exp}'
    features = ['is_black']

    # loads w2v model
    w2v_model = load_word2vec_model(path_to_model)
    main_lr(w2v_model, features, model_name, experiment_name)




