import argparse
import json
import os
from data_processing import save_type_mappings, convert_raw_data_to_json, convert_raw_labels_to_json, gen_score_label, gen_per_q_stat_label, analyze_processed_data
from training import (
    pretrain_and_split_data, train_predictor_and_split_data, train_ckt_encoder_and_split_data, train_ckt_predictor_and_split_data,
    test_pretrain, test_predictor, test_ckt_encoder, test_ckt_predictor
)
from analysis import cluster, cluster_irt, visualize_irt, block_scores
from experiments import full_pipeline, get_ppl_performance
from utils import initialize_seeds, device
from constants import TrainOptions, PredictionState, Direction
from irt.irt_training import irt

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
    # Data processing
    parser.add_argument("--process_data", nargs="+", help="Process raw data files into json format")
    parser.add_argument("--block", choices=["A", "B"], help="If given, only use events from this block while processing data")
    parser.add_argument("--data_classes", nargs="+", help="Assign data_class of each sequence in corresponding data file")
    parser.add_argument("--trim_after", nargs="+", type=float, help="Stop processing streams after they exceed this many seconds for each data file")
    parser.add_argument("--out", help="File to output processed data to")
    parser.add_argument("--labels", help="Process raw labels into json format")
    parser.add_argument("--types", help="Calculate event types")
    parser.add_argument("--analyze_data", help="Print properties of processed data file")
    # Training and testing
    parser.add_argument("--name", help="Filename to save/load model to/from", default="models/model")
    parser.add_argument("--pretrained_name", help="File that contains pretrained model", default="models/model")
    parser.add_argument("--data_src", help="File to use as data source")
    parser.add_argument("--pretrain", help="Pre-train the process model", action="store_true")
    parser.add_argument("--test_pretrain", help="Validate pretrained model", action="store_true")
    parser.add_argument("--train", help="Train the per-student predictor model", action="store_true")
    parser.add_argument("--task", help="The per-student label to use", choices=["comp", "score", "q_stats"])
    parser.add_argument("--test_predictor", help="Validate predictive model", action="store_true")
    parser.add_argument("--ckt", help="Use CKT models for operations", action="store_true")
    parser.add_argument("--concat_visits", help="Concatenate visits by QID for per-question encoding. Determines if transfer function uses FFNN or RNN.", type=bool_type)
    parser.add_argument("--full_pipeline", help="Perform cross-validation on pretraining/fine-tuning pipeline", action="store_true")
    parser.add_argument("--ppl", help="Given output of Playpower Labs prediction file, calculate AUC")
    parser.add_argument("--irt", help="Run cross-validation experiment on IRT model", action="store_true")
    parser.add_argument("--lr", help="Custom learning rate", type=float)
    parser.add_argument("--weight_decay", help="Custom weight decay", type=float)
    parser.add_argument("--epochs", help="Custom max epochs", type=int)
    parser.add_argument("--batch_size", help="Cusom batch size", type=int)
    parser.add_argument("--mixed_time", help="Indicate that provided data file contains 10/20/30 splits", action="store_true")
    parser.add_argument("--random_trim", help="During pre-training, randomly trim sequences at 5 minute intervals", action="store_true")
    parser.add_argument("--lstm_dir", help="LSTM direction", choices=list(LSTM_DIRS.keys()))
    parser.add_argument("--pretrained_model", help="Use pre-trained process model parameters", type=bool_type)
    parser.add_argument("--pretrained_emb", help="Use pre-trained embeddings", type=bool_type)
    parser.add_argument("--pretrained_head", help="Use pre-trained prediction head", type=bool_type)
    parser.add_argument("--freeze_model", help="Freeze process model", type=bool_type)
    parser.add_argument("--freeze_emb", help="Freeze embeddings", type=bool_type)
    parser.add_argument("--pred_state", help="Mechanism for creating sequence vector from process model outputs", choices=list(PRED_STATES.keys()))
    parser.add_argument("--dropout", help="Custom dropout on sequence vector prior to prediction head", type=float)
    parser.add_argument("--hidden_ff_layer", help="Use a hidden feed-forward layer in the prediction head", type=bool_type)
    parser.add_argument("--eng_feat", help="Append engineered features to sequence vector prior to prediction head", type=bool_type)
    parser.add_argument("--multi_head", help="Use a separate prediction head for each 10/20/30 data class", type=bool_type)
    parser.add_argument("--use_correctness", help="If correctness will be encoded in inputs and if it can be predicted during pre-training", type=bool_type)
    parser.add_argument("--use_visit_pt_objs", help="During pre-training (for full sequence model) predict event type and correctness at per-visit level", type=bool_type)
    parser.add_argument("--do_pretraining", help="If pre-training should be performed in cross-validation experiments", type=bool_type)
    parser.add_argument("--do_fine_tuning", help="If process model weights should be fine-tuned after prediction head training in cross-validation experiments", type=bool_type)
    parser.add_argument("--from_scratch", help="If the process model and prediction head should be trained directly on target lables in cross-validation experiments", type=bool_type)
    parser.add_argument("--predict_event_type", help="If per-event event type should be predicted during pre-training", type=bool_type)
    parser.add_argument("--predict_time", help="If per-event timestamp should be predicted during pre-training", type=bool_type)
    parser.add_argument("--predict_correctness", help="If per-event correctness should be predicted during pre-training", type=bool_type)
    parser.add_argument("--predict_qid", help="If per-event question ID should be predicted during pre-training", type=bool_type)
    parser.add_argument("--per_q_arch", help="If true, split sequence into questions and train NN on combination. Otherwise, use full sequence.", type=bool_type)
    parser.add_argument("--use_behavior_model", help="For IRT, use behavior-enhanced formulation", action="store_true")
    # Visualizations
    parser.add_argument("--cluster", help="Produce student-level visualizations", action="store_true")
    parser.add_argument("--cluster_irt", help="Produce question-level visualizations from IRT model", action="store_true")
    parser.add_argument("--viz_irt", help="For IRT model, plot student ability to score, followed by question difficulty to average score", action="store_true")
    parser.add_argument("--block_scores", help="Plot block A against block B scores per student", action="store_true")

    args = parser.parse_args()
    if os.path.isfile("default.json"):
        with open("default.json") as default_param_file:
            arg_dict: dict = json.load(default_param_file)
    else:
        arg_dict = {}

    arg_dict.update({arg: val for arg, val in vars(args).items() if val is not None})
    arg_dict["lstm_dir"] = LSTM_DIRS.get(arg_dict.get("lstm_dir"), Direction.BI)
    arg_dict["pred_state"] = PRED_STATES.get(arg_dict.get("pred_state"), PredictionState.ATTN)
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
            test_ckt_predictor(args.name, args.data_src, TrainOptions(arg_dict))
        else:
            test_predictor(args.name, args.data_src, TrainOptions(arg_dict))
    if args.full_pipeline:
        full_pipeline(args.pretrained_name, args.name, args.ckt, args.data_classes, TrainOptions(arg_dict))
    if args.ppl:
        get_ppl_performance(args.ppl)
    if args.irt:
        irt(args.pretrained_name, args.name, args.data_src, args.use_behavior_model, args.ckt, TrainOptions(arg_dict))
    if args.cluster:
        cluster(args.name, args.data_src, TrainOptions(arg_dict))
    if args.cluster_irt:
        cluster_irt(args.name, args.data_src, TrainOptions(arg_dict))
    if args.viz_irt:
        visualize_irt(args.name, args.data_src, args.use_behavior_model, TrainOptions(arg_dict))
    if args.block_scores:
        block_scores(args.data_src)
