'''
Data flow:
The RNN input will be a single question, which is a list of event vectors.
Note: we might actually want each clickstream to be the full clickstream for a student, rather than for a student's question - should experiment with both.
We can pad the question lists so the RNN can have consistent lengths.
We will have a list of these question clickstreams, taken across students.
So the shape of the data is num_questions x longest_question_stream x event_vector_size.
Then we'll feed the data into the RNN in batches, and then RNN will have to feed into a classifier.
The classifier will be a linear layer with softmax that predicts an event type.

Pretraining Enhancements:
 - New objectives:
    - Predict timestep (as a proportion between previous and post timesteps)
    - Predict next question (only at events where question is actually changing)
 - Split into per-question streams and capture switching between questions
 - Expand ExtendedInfo into new events
 - Restrict event predictions to those possible for associated question types
 - Feed question-level info into model
 - Get ideas from NAEP test https://nces.ed.gov/nationsreportcard/nqt/

 - Stretches:
    - Masked language modeling alternative to LSTMs
     -Use per-question sequences for pretraining

Training Enhancements:
  - Mix in engineered features:
    - Per question:
        - Correctness
        - Amount of time taken?
 - Visualize attention and analyze
 - First train final layer on top of frozen pretrained model, and then fine-tune whole thing
'''

import argparse
import json
import os
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
    "attn": PredictionState.ATTN,
}

def bool_type(arg):
    return False if arg == "0" else True

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
    parser.add_argument("--process_data", nargs="+", help="Process raw data files into json format")
    parser.add_argument("--labels", help="Process raw labels into json format")
    parser.add_argument("--types", help="Calculate event types")
    parser.add_argument("--out", help="File to output processed data to")
    parser.add_argument("--trim_after", help="Stop processing streams after they exceed this many seconds", type=float)
    parser.add_argument("--name", help="Filename to save model to", default="models/model")
    parser.add_argument("--pretrained_name", help="File that contains pretrained model", default="models/model")
    parser.add_argument("--data_src", help="File to use as data source")
    parser.add_argument("--pretrain", help="Pre-train the LSTM model", action="store_true")
    parser.add_argument("--test_pretrain", help="Validate pretrained model", action="store_true")
    parser.add_argument("--train", help="Train the LSTM predictor", action="store_true")
    parser.add_argument("--test_predictor", help="Validate predictive model", action="store_true")
    parser.add_argument("--config", help="Config file for multiple runs")
    parser.add_argument("--lstm_dir", help="LSTM direction", choices=list(LSTM_DIRS.keys()))
    parser.add_argument("--pretrained_model", type=bool_type)
    parser.add_argument("--pretrained_emb", type=bool_type)
    parser.add_argument("--freeze_model", type=bool_type)
    parser.add_argument("--freeze_emb", type=bool_type)
    parser.add_argument("--pred_state", choices=list(PRED_STATES.keys()))
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--hidden_ff_layer", type=bool_type)
    args = parser.parse_args()

    if os.path.isfile("default.json"):
        with open("default.json") as default_param_file:
            arg_dict: dict = json.load(default_param_file)
    else:
        arg_dict = {}

    arg_dict.update({arg: val for arg, val in vars(args).items() if val is not None})
    arg_dict["lstm_dir"] = LSTM_DIRS.get(arg_dict.get("lstm_dir"), Direction.BI)
    arg_dict["pred_state"] = PRED_STATES.get(arg_dict.get("pred_state"), PredictionState.BOTH_CONCAT)
    print("Settings:", arg_dict)

    initialize_seeds(221)

    if device.type == "cuda":
        print("Running on GPU")

    if args.types:
        save_type_mappings(args.types)
    if args.process_data:
        convert_raw_data_to_json(args.process_data, args.out, args.trim_after)
    if args.labels:
        convert_raw_labels_to_json(args.labels, args.out)
    if args.pretrain:
        pretrain(args.name, args.data_src, TrainOptions(arg_dict))
    if args.test_pretrain:
        test_pretrain(args.name, args.data_src, TrainOptions(arg_dict))
    if args.train:
        train(args.pretrained_name, args.name, args.data_src, TrainOptions(arg_dict))
    if args.test_predictor:
        test_predictor(args.name, args.data_src, TrainOptions(arg_dict))
