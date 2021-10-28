'''
Data flow:
The RNN input will be a single question, which is a list of event vectors.
Note: we might actually want each clickstream to be the full clickstream for a student, rather than for a student's question - should experiment with both.
We can pad the question lists so the RNN can have consistent lengths.
We will have a list of these question clickstreams, taken across students.
So the shape of the data is num_questions x longest_question_stream x event_vector_size.
Then we'll feed the data into the RNN in batches, and then RNN will have to feed into a classifier.
The classifier will be a linear layer with softmax that predicts an event type.

Things to try now:
 - Try adding extra linear layer to frozen model

Pretraining Enhancements:
 - Split into per-question streams and capture switching between questions
 - Expand ExtendedInfo into new events
 - Restrict event predictions to those possible for associated question types
 - Feed question-level info into model
 - Get ideas from NAEP test https://nces.ed.gov/nationsreportcard/nqt/

 - Stretches:
    - Masked language modeling alternative to LSTMs
     -Use per-question sequences for pretraining

Training Enhancements:
 - Adjust learning rate
 - First train final layer on top of frozen pretrained model, and then fine-tune whole thing
 - Train for more epochs and plot learning curve
'''

import argparse
import torch
import numpy as np
import random
from dataset import save_type_mappings, convert_raw_data_to_json, convert_raw_labels_to_json
from training import pretrain, train, test_pretrain, test_predictor
from model import TrainOptions, PredictionState, Direction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LSTM_DIRS = {
    "fwd": Direction.FWD,
    "back": Direction.BACK,
    "bi": Direction.BI,
}

PRED_STATES = {
    "last": PredictionState.LAST,
    "first": PredictionState.FIRST,
    "both_concat": PredictionState.BOTH_CONCAT,
    "both_sum": PredictionState.BOTH_SUM,
    "avg": PredictionState.AVG,
}

def initialize_seeds(seedNum):
    # TODO: analyze this and make sure it's all necessary
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    random.seed(seedNum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seedNum)
        torch.cuda.manual_seed_all(seedNum)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Clickstream Assessments")
    parser.add_argument("--data", nargs="+", help="Process raw data into json format")
    parser.add_argument("--labels", help="Process raw labels into json format")
    parser.add_argument("--types", help="Calculate event types")
    parser.add_argument("--out", help="File to output processed data to")
    parser.add_argument("--trim_after", help="Stop processing streams after they exceed this many seconds", type=float)
    parser.add_argument("--name", help="Filename to save model to", default="models/model")
    parser.add_argument("--pretrained_name", help="File that contains pretrained model", default="models/model")
    parser.add_argument("--pretrain", help="Pre-train the LSTM model", action="store_true")
    parser.add_argument("--test_pretrain", help="Validate pretrained model", action="store_true")
    parser.add_argument("--train", help="Train the LSTM predictor", action="store_true")
    parser.add_argument("--test_predictor", help="Validate predictive model", action="store_true")
    parser.add_argument("--lstm_dir", help="LSTM direction", choices=list(LSTM_DIRS.values()), type=lambda lstm_dir: LSTM_DIRS.get(lstm_dir), default=Direction.BI)
    parser.add_argument("--pretrained_model", action="store_true")
    parser.add_argument("--pretrained_emb", action="store_true")
    parser.add_argument("--freeze_model", action="store_true")
    parser.add_argument("--freeze_emb", action="store_true")
    parser.add_argument("--pred_state", choices=list(PRED_STATES.values()), type=lambda pred_state: PRED_STATES.get(pred_state), default=PredictionState.BOTH_CONCAT)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--dropout", type=float)
    args = parser.parse_args()
    arg_dict = {arg: val for arg, val in vars(args).items() if val is not None}

    initialize_seeds(221)

    if device.type == "cuda":
        print("Running on GPU")

    if args.types:
        save_type_mappings(args.types)
    if args.data:
        convert_raw_data_to_json(args.data, args.out, args.trim_after)
    if args.labels:
        convert_raw_labels_to_json(args.labels, args.out)
    if args.pretrain:
        pretrain(args.name, TrainOptions(arg_dict))
    if args.test_pretrain:
        test_pretrain(args.name, TrainOptions(arg_dict))
    if args.train:
        train(args.pretrained_name, args.name, TrainOptions(arg_dict))
    if args.test_predictor:
        test_predictor(args.name, TrainOptions(arg_dict))
