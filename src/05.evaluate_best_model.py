import os
import numpy as np
import pickle as pk
from tqdm import tqdm
import struct
import yaml
from time import sleep
from keras.models import Model
from keras import backend as K
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from model_RawNet import get_model


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compose_spkFeat_dic(lines, model, f_desc_dic, base_dir):
    dic_spkFeat = {}
    for line in tqdm(lines, desc="extracting spk feats"):
        k, f, p = line.strip().split(" ")
        p = int(p)
        if f not in f_desc_dic:
            f_tmp = "/".join([base_dir, f])
            f_desc_dic[f] = open(f_tmp, "rb")

        f_desc_dic[f].seek(p)
        l = struct.unpack("i", f_desc_dic[f].read(4))[0]
        utt = np.asarray(
            struct.unpack("%df" % l, f_desc_dic[f].read(l * 4)), dtype=np.float32
        )
        spkFeat = model.predict(utt.reshape(1, -1, 1))[0]
        dic_spkFeat[k] = spkFeat

    return dic_spkFeat


# Load the saved model weights and create the model
def load_model_weights(model, weights_path):
    model.load_weights(weights_path)


# Function to evaluate and get EER
def evaluate_model(eval_lines, model_pred, base_dir, trials):
    dic_eval = compose_spkFeat_dic(
        lines=eval_lines, model=model_pred, f_desc_dic={}, base_dir=base_dir
    )

    y = []
    y_score = []
    for smpl in trials:
        target, spkMd, utt = smpl.strip().split(" ")
        target = int(target)
        cos_score = cos_sim(dic_eval[spkMd], dic_eval[utt])
        y.append(target)
        y_score.append(cos_score)

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer


def make_spkdic(lines):
    """
    Returns a dictionary where
            key: (str) speaker name
            value: (int) unique integer for each speaker
    """
    idx = 0
    dic_spk = {}
    list_spk = []
    for line in lines:
        k, f, p = line.strip().split(" ")
        spk = k.split("/")[0]
        if spk not in dic_spk:
            dic_spk[spk] = idx
            list_spk.append(spk)
            idx += 1
    return (dic_spk, list_spk)


# Example usage
if __name__ == "__main__":
    _abspath = os.path.abspath(__file__)
    dir_yaml = os.path.splitext(_abspath)[0] + ".yaml"
    with open(dir_yaml, "r") as f_yaml:
        parser = yaml.load(f_yaml)

    dir_dev_scp = parser["dev_scp"]
    with open(dir_dev_scp, "r") as f_dev_scp:
        dev_lines = f_dev_scp.readlines()
    dic_spk, list_spk = make_spkdic(dev_lines)
    parser["model"]["nb_spk"] = len(list_spk)
    print("# spk: ", len(list_spk))

    # Load model and weights
    model, m_name = get_model(argDic=parser["model"])
    model_pred = Model(
        inputs=model.get_layer("input_RawNet").input,
        outputs=model.get_layer("code_RawNet").output,
    )
    load_model_weights(model, parser["save_dir"] + "RawNet_weights.h5")

    # Evaluate the model
    eval_lines = open(parser["eval_scp"], "r").readlines()
    trials = open(parser["trials"], "r").readlines()
    eer = evaluate_model(eval_lines, model_pred, parser["base_dir"], trials)

    print("EER: %f" % eer)

#     dev_dic_embeddings = compose_spkFeat_dic(lines = dev_lines,
#         model = model_pred,
#         f_desc_dic = {},
#         base_dir = parser['base_dir'])

#     print('Extracting Embeddings from GRU model: eval set')
#     eval_dic_embeddings = compose_spkFeat_dic(lines = eval_lines,
#         model = model_pred,
#         f_desc_dic = {},
#         base_dir = parser['base_dir'])

#     f_embeddings = open(parser['gru_embeddings'] + 'speaker_embeddings_RawNet', 'wb')
#     pk.dump({'dev_dic_embeddings': dev_dic_embeddings, 'eval_dic_embeddings': eval_dic_embeddings},
#         f_embeddings,
#         protocol = pk.HIGHEST_PROTOCOL)
