from enum import Enum

ASSISTIVE_EVENT_IDS = [4, 8, 17, 24, 27, 34] # Open Calculator, Eliminate Choice, Scratchwork Mode On, Highlight, TextToSpeech, Increase Zoom

class Mode(Enum):
    PRE_TRAIN = 1
    PREDICT = 2
    CLUSTER = 3
    CKT_ENCODE = 4
    IRT = 5

class Direction(Enum):
    FWD = 1
    BACK = 2
    BI = 3

class PredictionState(Enum):
    LAST = 1
    FIRST = 2
    BOTH_SUM = 3
    BOTH_CONCAT = 4
    AVG = 5
    ATTN = 6

class TrainOptions:
    def __init__(self, options: dict):
        self.task: str = options.get("task", "comp")

        self.lr: float = options.get("lr", 1e-4)
        self.weight_decay: bool = options.get("weight_decay", 1e-6)
        self.epochs: int = options.get("epochs", 100)
        self.mixed_time: bool = options.get("mixed_time", False)
        self.random_trim: bool = options.get("random_trim", False)
        self.concat_visits: bool = options.get("concat_visits", True)

        self.lstm_dir: Direction = options.get("lstm_dir", Direction.BI)
        self.use_pretrained_weights: bool = options.get("pretrained_model", True)
        self.use_pretrained_embeddings: bool = options.get("pretrained_emb", True)
        self.use_pretrained_head: bool = options.get("pretrained_head", False)
        self.freeze_model: bool = options.get("freeze_model", True)
        self.freeze_embeddings: bool = options.get("freeze_emb", True)
        self.prediction_state: PredictionState = options.get("pred_state", PredictionState.BOTH_CONCAT)
        self.dropout: float = options.get("dropout", 0.25)
        self.hidden_ff_layer: bool = options.get("hidden_ff_layer", False)
        self.engineered_features: bool = options.get("eng_feat", False)
        self.multi_head: bool = options.get("multi_head", False)
        self.use_correctness: bool = options.get("use_correctness", True)
        self.use_visit_pt_objs: bool = options.get("use_visit_pt_objs", True)

NUM_CORRECTNESS_STATES = 3

class Correctness(Enum):
    INCOMPLETE = 0
    INCORRECT = 1
    CORRECT = 2
