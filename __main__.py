import argparse
import json
import os
from data_processing import save_type_mappings, convert_raw_data_to_json, convert_raw_labels_to_json, gen_score_label, gen_per_q_stat_label, analyze_processed_data
from training import (
    pretrain_and_split_data, train_predictor_and_split_data, train_ckt_encoder_and_split_data, train_ckt_predictor_and_split_data,
    test_pretrain, test_predictor, test_ckt_encoder, test_ckt_predictor, cluster
)
from experiments import full_pipeline, get_ppl_performance
from utils import initialize_seeds, device
from constants import TrainOptions, PredictionState, Direction
from irt.irt_training import irt, test_irt

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Clickstream Assessments")
    parser.add_argument("--process_data", nargs="+", help="Process raw data files into json format")
    parser.add_argument("--block", choices=["A", "B"], help="If given, only use events from this block while processing data")
    parser.add_argument("--analyze_data", help="Analyze processed data file")
    parser.add_argument("--data_classes", nargs="+", help="Assign data_class of each sequence in corresponding data file")
    parser.add_argument("--labels", help="Process raw labels into json format")
    parser.add_argument("--types", help="Calculate event types")
    parser.add_argument("--out", help="File to output processed data to")
    parser.add_argument("--trim_after", nargs="+", type=float, help="Stop processing streams after they exceed this many seconds for each data file")
    parser.add_argument("--name", help="Filename to save model to", default="models/model")
    parser.add_argument("--pretrained_name", help="File that contains pretrained model", default="models/model")
    parser.add_argument("--data_src", help="File to use as data source")
    parser.add_argument("--pretrain", help="Pre-train the LSTM model", action="store_true")
    parser.add_argument("--test_pretrain", help="Validate pretrained model", action="store_true")
    parser.add_argument("--train", help="Train the LSTM predictor", action="store_true")
    parser.add_argument("--task", choices=["comp", "score", "q_stats"])
    parser.add_argument("--test_predictor", help="Validate predictive model", action="store_true")
    parser.add_argument("--ckt", help="Use CKT models for operations", action="store_true")
    parser.add_argument("--concat_visits", help="Concatenate visits by QID for per-question encoding", type=bool_type)
    parser.add_argument("--full_pipeline", help="Perform cross-validation on pretraining/fine-tuning pipeline", action="store_true")
    parser.add_argument("--ppl")
    parser.add_argument("--irt", help="Train IRT model", action="store_true")
    parser.add_argument("--test_irt", action="store_true")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--mixed_time", action="store_true")
    parser.add_argument("--random_trim", action="store_true")
    parser.add_argument("--lstm_dir", help="LSTM direction", choices=list(LSTM_DIRS.keys()))
    parser.add_argument("--pretrained_model", type=bool_type)
    parser.add_argument("--pretrained_emb", type=bool_type)
    parser.add_argument("--pretrained_head", type=bool_type)
    parser.add_argument("--freeze_model", type=bool_type)
    parser.add_argument("--freeze_emb", type=bool_type)
    parser.add_argument("--pred_state", choices=list(PRED_STATES.keys()))
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--hidden_ff_layer", type=bool_type)
    parser.add_argument("--eng_feat", type=bool_type)
    parser.add_argument("--multi_head", type=bool_type)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--use_correctness", type=bool_type)
    parser.add_argument("--use_visit_pt_objs", type=bool_type)
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
        convert_raw_data_to_json(args.process_data, args.out, args.block, args.trim_after, args.data_classes)
    if args.analyze_data:
        analyze_processed_data(args.analyze_data)
    if args.labels:
        if args.task == "comp":
            convert_raw_labels_to_json(args.labels, args.out)
        elif args.task == "score":
            gen_score_label(args.labels, args.out)
        elif args.task == "q_stats":
            gen_per_q_stat_label(args.labels, args.out)
    if args.pretrain:
        if args.ckt:
            train_ckt_encoder_and_split_data(args.name, args.data_src, TrainOptions(arg_dict))
        else:
            pretrain_and_split_data(args.name, args.data_src, TrainOptions(arg_dict))
    if args.test_pretrain:
        if args.ckt:
            test_ckt_encoder(args.name, args.data_src, TrainOptions(arg_dict))
        else:
            test_pretrain(args.name, args.data_src, TrainOptions(arg_dict))
    if args.train:
        if args.ckt:
            train_ckt_predictor_and_split_data(args.pretrained_name, args.name, args.data_src, TrainOptions(arg_dict))
        else:
            train_predictor_and_split_data(args.pretrained_name, args.name, args.data_src, TrainOptions(arg_dict))
    if args.test_predictor:
        if args.ckt:
            test_ckt_predictor(args.pretrained_name, args.name, args.data_src, TrainOptions(arg_dict))
        else:
            test_predictor(args.name, args.data_src, TrainOptions(arg_dict))
    if args.full_pipeline:
        full_pipeline(args.pretrained_name, args.name, args.ckt, TrainOptions(arg_dict))
    if args.ppl:
        get_ppl_performance(args.ppl)
    if args.cluster:
        cluster(args.name, args.data_src, TrainOptions(arg_dict))
    if args.irt:
        irt(args.pretrained_name, args.name, args.data_src, args.ckt, TrainOptions(arg_dict))
    if args.test_irt:
        test_irt(args.name, args.data_src, args.ckt, TrainOptions(arg_dict))
