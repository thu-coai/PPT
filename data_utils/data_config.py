from .downstream.SST5Datasets import SST5Dataset, SST5DatasetUni
from .downstream.SST2Datasets import SST2Dataset, SST2DatasetUni
from .downstream.RACEDatasets import RACEDataset, RACEDatasetUni
from .downstream.CBDatasets import CBDataset, CBDatasetUni
from .downstream.RTEDatasets import RTEDataset, RTEDatasetUni
from .downstream.BoolQDatasets import BoolQDataset, BoolQDatasetUni

from .pretrain.CLSPretrainDatasets import CLSPretrainDataset
from .pretrain.NSSPretrainDatasets import NSSPretrainDataset
from .pretrain.NSPPretrainDatasets import NSPPretrainDataset
from .pretrain.LMPretrainDatasets import LMPretrainDataset

DATA_CONFIG = {
    "boolq": {
        "dataset": BoolQDataset,
    },
    "boolq_uni": {
        "dataset": BoolQDatasetUni,
    },
    "rte": {
        "dataset": RTEDataset,
    },
    "rte_uni": {
        "dataset": RTEDatasetUni,
    },
    "cb": {
        "dataset": CBDataset,
    },
    "cb_uni": {
        "dataset": CBDatasetUni,
    },
    "race": {
        "dataset": RACEDataset,
    },
    "race_uni": {
        "dataset": RACEDatasetUni,
    },
    "sst2": {
        "dataset": SST2Dataset,
    },
    "sst2_uni": {
        "dataset": SST2DatasetUni,
    },
    "sst5": {
        "dataset": SST5Dataset,
    },
    "sst5_uni": {
        "dataset": SST5DatasetUni,
    },
    "lm": {
        "dataset": LMPretrainDataset
    },
    "cls": {
        "dataset": CLSPretrainDataset,
    },
    "nsp": {
        "dataset": NSPPretrainDataset
    },
    "nss": {
        "dataset": NSSPretrainDataset
    }
}