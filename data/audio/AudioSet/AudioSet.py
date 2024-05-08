# Copyright (C) 2024 Aaron Keesing
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# “Software”), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from itertools import chain
import json
import os
import tarfile

import pandas as pd
import datasets


_CITATION = """\
@inproceedings{45857,
    title = {Audio Set: An ontology and human-labeled dataset for audio events},
    author = {Jort F. Gemmeke and Daniel P. W. Ellis and Dylan Freedman and Aren Jansen and Wade Lawrence and R. Channing Moore and Manoj Plakal and Marvin Ritter},
    year = {2017},
    booktitle = {Proc. IEEE ICASSP 2017},
    address	= {New Orleans, LA}
}
"""

_DESCRIPTION = """\
This repository contains the balanced training set and evaluation set of the AudioSet
data, described here: https://research.google.com/audioset/dataset/index.html. The
YouTube videos were downloaded in March 2023, and so not all of the original audios are
available.
"""

_HOMEPAGE = "https://research.google.com/audioset/dataset/index.html"

_LICENSE = "cc-by-4.0"

_URL_PREFIX = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main"

_N_BAL_TRAIN_TARS = 10
_N_UNBAL_TRAIN_TARS = 870
_N_EVAL_TARS = 9


def _iter_tar(path):
    """Iterate through the tar archive, but without skipping some files, which the HF
    DL does.
    """
    with open(path, "rb") as fid:
        stream = tarfile.open(fileobj=fid, mode="r|*")
        for tarinfo in stream:
            file_obj = stream.extractfile(tarinfo)
            yield tarinfo.name, file_obj
            stream.members = []
        del stream


class AudioSetDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="balanced",
            version=VERSION,
            description="Balanced training and balanced evaluation set.",
        ),
        datasets.BuilderConfig(
            name="unbalanced",
            version=VERSION,
            description="Full unbalanced training set and balanced evaluation set.",
        ),
    ]
    DEFAULT_CONFIG_NAME = "balanced"

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=datasets.Features(
                {
                    "video_id": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=None, mono=True, decode=True),
                    "labels": datasets.Sequence(datasets.Value("string")),
                    "human_labels": datasets.Sequence(datasets.Value("string")),
                }
            ),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        if self.config.data_dir:
            prefix = self.config.data_dir
        else:
            prefix = _URL_PREFIX
        prefix = prefix + "/data"

        _LABEL_URLS = {
            "bal_train": (
                f"{prefix}/balanced_train_segments.csv"
                if self.config.name == "balanced"
                else f"{prefix}/unbalanced_train_segments.csv"
            ),
            "eval": f"{prefix}/eval_segments.csv",
            "ontology": f"{prefix}/ontology.json",
        }
        _DATA_URLS = {
            "bal_train": (
                [f"{prefix}/bal_train0{i}.tar" for i in range(_N_BAL_TRAIN_TARS)]
                if self.config.name == "balanced"
                else [
                    f"{prefix}/unbal_train{i:03d}.tar"
                    for i in range(_N_UNBAL_TRAIN_TARS)
                ]
            ),
            "eval": [f"{prefix}/eval0{i}.tar" for i in range(_N_EVAL_TARS)],
        }

        tar_files = dl_manager.download(_DATA_URLS)
        label_files = dl_manager.download(_LABEL_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "labels": label_files["bal_train"],
                    "ontology": label_files["ontology"],
                    "audio_files": chain.from_iterable(
                        _iter_tar(x) for x in tar_files["bal_train"]
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "labels": label_files["eval"],
                    "ontology": label_files["ontology"],
                    "audio_files": chain.from_iterable(
                        _iter_tar(x) for x in tar_files["eval"]
                    ),
                },
            ),
        ]

    def _generate_examples(self, labels, ontology, audio_files):
        with open(ontology) as fid:
            ontology_data = json.load(fid)
        id_to_name = {x["id"]: x["name"] for x in ontology_data}

        labels_df = pd.read_csv(
            labels,
            skiprows=3,
            header=None,
            skipinitialspace=True,
            names=["vid_id", "start", "end", "labels"],
            index_col="vid_id",
        )

        for path, fid in audio_files:
            vid_id = os.path.splitext(os.path.basename(path))[0]
            label_ids = labels_df.loc[vid_id, "labels"].split(",")
            human_labels = [id_to_name[x] for x in label_ids]
            example = {
                "video_id": vid_id,
                "labels": label_ids,
                "human_labels": human_labels,
                "audio": {"path": path, "bytes": fid.read()},
            }
            yield vid_id, example
