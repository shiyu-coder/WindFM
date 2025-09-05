from .windfm import WindFM, WindFMTokenizer, WindFMPredictor

model_dict = {
    'windfm_tokenizer': WindFMTokenizer,
    'windfm': WindFM,
    'windfm_predictor': WindFMPredictor
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        print(f"Model {model_name} not found in model_dict")
        raise NotImplementedError


