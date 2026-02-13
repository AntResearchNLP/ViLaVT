import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

# ==================================== vilavt training data ====================================
SR_91K_TEXT = {
    "annotation_path": "path/to/SR_91K_TEXT.json",     # "/path/to/annotations.json",
    "data_path": "/path/to/image/data",                                               # Can be empty if paths are in annotations
}

SPAR7M_TEXT = {
    "annotation_path": "path/to/SPAR7M_TEXT .json",         # "/path/to/annotations.json",
    "data_path": "/path/to/image/data",                                             # Can be empty if paths are in annotations
}

SR_91K_COT = {
    "annotation_path": "path/to/SR_91K_COT.json",           # "/path/to/annotations.json",
    "data_path": "/path/to/image/data",                                             # Can be empty if paths are in annotations
}

SPAR7M_COT = {
    "annotation_path": "path/to/SPAR7M_COT.json",                 # "/path/to/annotations.json",
    "data_path": "",                                                                # Can be empty if paths are in annotations
}

VGR_COT = {
    "annotation_path": "path/to/VGR_COT.json",         
    "data_path": "/path/to/image/data",                                             # Can be empty if paths are in annotations
}

THYME_COT = {
    "annotation_path": "path/to/THYME_COT.json",         
    "data_path": "/path/to/image/data",                                             # Can be empty if paths are in annotations
}

THYME_TEXT = {
    "annotation_path": "path/to/THYME_TEXT.json",         
    "data_path": "/path/to/image/data",                                             # Can be empty if paths are in annotations
}


VICA_COT = {
    "annotation_path": "path/to/VICA_COT.json",
    "data_path": "/path/to/image/data",                                             # Can be empty if paths are in annotations
}

VICA_TEXT = {
    "annotation_path": "path/to/VICA_TEXT.json",
    "data_path": "/path/to/image/data",                                             # Can be empty if paths are in annotations
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    # ViLaVT data
    "sr_91k_text": SR_91K_TEXT,
    "spar7m_text": SPAR7M_TEXT,
    "sr_91k_v2": SR_91K_COT,
    "spar7m_v2": SPAR7M_COT,
    "vgr_v2": VGR_COT,
    "thyme_2turn_v2": THYME_COT,
    "thyme_text_v2": THYME_TEXT,
    "vica_cot": VICA_COT,
    "vica_text": VICA_TEXT,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
