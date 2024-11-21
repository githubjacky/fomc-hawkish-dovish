####################################################################################################
# There are three classes in this file, one base class and two subclasses. The only difference
# between subclasses is that one provides sentence-level embeddings and the other destined to 
# word-level embeddings. Moreover, what I mean by "provide" is how we call `generate_embeds` in
# different scenarios. If we aim to train a sequence model, such as RNN we need word-level 
# embeddings. Nevertheless, if we focus on training a MLP or multinominal regression, we can rely 
# on sentence-level embeddings.
####################################################################################################


from datasets import load_dataset
from flair.data import Sentence
from flair.embeddings import (
    TransformerDocumentEmbeddings,
    TransformerWordEmbeddings
)
import lightning as L
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence
from transformers import AutoConfig
from typing import List, Dict, Literal


class BaseDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int, embed_model_name: str):
        """
        If one only wants to use the last layers, `flair_layers` should be
        specified as "-1". Moreover, `flar_layers` = "-1,-2" means the last
        second layers are used so "all" means all layers.

        If `flair_layers` != "-1", then `flair_layer_mean` should be True to
        average all embeddings across layers.
        """

        super().__init__()
        self.batch_size = batch_size

        # it's identified in subclass
        self.embed_model_name = embed_model_name
        self.embed_model = None

    @property
    def embed_dimension(self):
        return AutoConfig.from_pretrained(self.embed_model_name).hidden_size


    # download dataset
    def prepare_data(self) -> None:
        dataset = load_dataset("gtfintechlab/fomc_communication")
        self.train_dataset = dataset["train"]
        self.test_dataset = dataset["test"]

        self.len_train_dataset = len(self.train_dataset)
        self.len_test_dataset = len(self.test_dataset)


    # split the train dataset
    def setup(self, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        """
        `train_idx` and `val_idx` are the results of `sklearn.model_selection.train_test_split`
        """
        self.train_data = Subset(self.train_dataset, train_idx)
        self.val_data = Subset(self.train_dataset, val_idx)

        self.len_train_data = len(train_idx)
        self.len_val_data = len(val_idx)


    @property
    def sklearn_class_weight(self) -> torch.Tensor:
        y = []
        for i in range(len(self.train_data)):
            item = self.train_data[[i]]
            y.append(item["label"][0])
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)

        return torch.tensor(weights, dtype=torch.float).cuda()


    # it's defined in subclass
    def generate_embeds(self, batch: List[Dict]) -> None:
        pass


    # generate embedding for `sentence` in collate_fn
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Batch lookes like this:
        [
            {"index":..., "sentence":..., "year": ..., "label":..., "orig_index":...},
            ...
        ]
        The length of batch is self.batch_size.

        If a key is a numeric value, then the output of this function is
        converted to torch.Tensor.
        For example:
        {
            "year": tensor([2011, 2004, ...])
        }
        """
        return {
            "sentence": [elem["sentence"] for elem in batch],
            "label": torch.tensor([elem["label"] for elem in batch]),
            "embed": self.generate_embeds(batch)
        }


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_fn)


    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.collate_fn)


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)


class MLPDataModule(BaseDataModule):
    def __init__(self,
                 batch_size: int = 256,
                 framework: Literal["flair", "sbert"] = "flair",
                 embed_model_name: str = "bert-base-uncased",
                 flair_layers: str = "all",
                 flair_layer_mean = True):

        super().__init__(batch_size, embed_model_name)

        self.embed_model = (
            TransformerDocumentEmbeddings(
                embed_model_name,
                layers = flair_layers,
                layer_mean = flair_layer_mean
            )
            if framework == "flair"
            else
            SentenceTransformer(embed_model_name, device="cuda")
        )
        self.framework = framework


    def generate_embeds(self, batch: List[Dict]) -> torch.Tensor:
        if self.framework == "flair":
            sentences = [Sentence(elem["sentence"]) for elem in batch]
            self.embed_model.embed(sentences)

            # denote embedding's dimension(bert: 768) as D
            # denote `self.batch_size` as N
            # the dimension of embed: N * D
            embeds = torch.stack([sentence.embedding for sentence in sentences])
        else:
            embeds = torch.tensor(self.embed_model.encode([elem["sentence"] for elem in batch]))

        return embeds


class RNNFamilyDataModule(BaseDataModule):
    """
    We we decide to use RNNFamily model, we not only need to generate embeddings, we also need to
    pad the sequce as setneces have different length. Normally, we rely on the zero padding, but
    these paddings shouldn't be responsibile for the model training, i.e., both forwawrd pass and 
    back propagation should eglect these inputs of padding. And that's where `PackedSequence` comes
    into play. It tell RNNFamily model when to stop the forward pass before the start of paddings.
    """
    def __init__(self,
                 batch_size: int = 256,
                 embed_model_name: str = "bert-base-uncased",
                 flair_layers: str = "all",
                 flair_layer_mean = True):

        super().__init__(batch_size, embed_model_name)

        self.embed_model = TransformerWordEmbeddings(
            embed_model_name,
            layers = flair_layers,
            layer_mean = flair_layer_mean
        )


    def generate_embeds(self, batch: List[Dict]) -> PackedSequence:
        sentences = [Sentence(elem["sentence"]) for elem in batch]
        self.embed_model.embed(sentences)

        # denote embedding's dimension(bert: 768) as D
        # denote `self.batch_size` as N

        # the dimension of embed: N * L * D
        # where L is the maximm sequence length of each batch
        seq = pad_sequence(
            [
                torch.stack([item.embedding for item in sentence])
                for sentence in sentences
            ],
            batch_first=True
        )

        packed = pack_padded_sequence(
            seq,
            lengths=[len(sentence) for sentence in sentences],
            batch_first=True,
            enforce_sorted=False
        )

        return packed