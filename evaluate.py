"""TODO"""


def chunk_predict(chunk, model):
    return grad_model.predict_proba(chunk).sum(axis=0).argmax()
