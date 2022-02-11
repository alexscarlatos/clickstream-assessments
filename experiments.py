import json
import numpy as np
import pandas
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score

from training import pretrain, train_predictor, test_predictor_with_data, get_data, get_labels, train_ckt_encoder, train_ckt_predictor, test_ckt_predictor_with_data
from utils import initialize_seeds
from constants import TrainOptions

def full_pipeline(pretrained_name: str, model_name: str, ckt: bool, options: TrainOptions):
    test_data_from_train_data = False

    # Get train and test data
    if options.mixed_time:
        train_data_full = get_data("data/train_data_mixed.json")
        train_data_full.sort(key=lambda seq: (seq["data_class"], seq["student_id"]))
        train_data_full = np.array(train_data_full)
        train_data_for_split = train_data_full[:len(train_data_full) // 3]
        test_data_file = "data/test_data_full.json"
    else:
        train_data_full = np.array(get_data("data/train_data_full.json"))
        train_data_for_split = train_data_full
        test_data_file = "data/test_data_30.json"
    if not test_data_from_train_data:
        test_data = get_data(test_data_file)

    # Get train labels
    train_data_len = len(train_data_for_split)
    train_label_dict = get_labels(options.task, True)
    if options.task == "q_stats":
        # Can't stratify on per-question label, so stratify on score label
        train_labels_for_skf = get_labels("score", True)
        train_labels_for_split = [train_labels_for_skf[str(seq["student_id"])] for seq in train_data_for_split]
    else:
        train_labels_for_split = [train_label_dict[str(seq["student_id"])] for seq in train_data_for_split]

    all_frozen_stats = []
    all_ft_stats = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=221)
    for k, (train_data_idx, val_data_idx) in enumerate(skf.split(train_data_for_split, train_labels_for_split)):
        # Split train and validation (and possibly test) data on current partition
        if options.mixed_time:
            train_data = np.concatenate([
                train_data_full[chunk_start : chunk_start + train_data_len][train_data_idx]
                for chunk_start in range(0, train_data_len * 3, train_data_len)
            ], axis=0)
            val_data = np.concatenate([
                train_data_full[chunk_start : chunk_start + train_data_len][val_data_idx]
                for chunk_start in range(0, train_data_len * 3, train_data_len)
            ], axis=0)
        else:
            train_data = train_data_full[train_data_idx]
            if test_data_from_train_data:
                val_data = train_data_full[val_data_idx][:int(len(val_data_idx) / 2)]
                test_data = train_data_full[val_data_idx][int(len(val_data_idx) / 2):]
            else:
                val_data = train_data_full[val_data_idx]
        # Ensure no overlap between partitions
        assert not any(vd["student_id"] == td["student_id"] for vd in val_data for td in train_data)

        initialize_seeds(221)

        print("\n------ Iteration", k + 1, "------")
        cur_pretrained_name = f"{pretrained_name}_{k + 1}"
        cur_model_name = f"{model_name}_{k + 1}"
        cur_model_ft_name = f"{model_name}_ft_{k + 1}"

        print("\nPretraining Phase")
        options.lr = 1e-3
        options.epochs = 150 if ckt else 100
        # Just pretrain on 30-minute data if using mixed training data
        train_data_for_pt = train_data[len(train_data_idx) * 2:] if options.mixed_time else train_data
        val_data_for_pt = val_data[len(val_data_idx) * 2:] if options.mixed_time else val_data
        if False:
            if ckt:
                train_ckt_encoder(cur_pretrained_name, train_data_for_pt, val_data_for_pt, options)
            else:
                pretrain(cur_pretrained_name, train_data_for_pt, val_data_for_pt, options)

        print("\nTrain classifier on frozen LSTM")
        options.lr = 5e-4 if ckt else 5e-3
        options.epochs = 100
        options.use_pretrained_embeddings = True
        options.use_pretrained_weights = True
        options.use_pretrained_head = False
        options.freeze_embeddings = True
        options.freeze_model = True
        if ckt:
            val_stats = train_ckt_predictor(cur_pretrained_name, cur_model_name, train_data, val_data, train_label_dict, options)
            test_stats = test_ckt_predictor_with_data(cur_pretrained_name, cur_model_name, test_data, options)
        else:
            val_stats = train_predictor(cur_pretrained_name, cur_model_name, train_data, val_data, train_label_dict, options)
            test_stats = test_predictor_with_data(cur_model_name, test_data, options)
        all_frozen_stats.append(val_stats + test_stats)

        # Not fine-tuning CKT models for now
        if True:
            all_ft_stats.append([0] * (len(val_stats) * 2))
            continue

        print("\nFine-tune LSTM for classifier")
        options.lr = 5e-5
        options.epochs = 50
        options.use_pretrained_head = True
        options.freeze_embeddings = False
        options.freeze_model = False
        val_stats = train_predictor(cur_model_name, cur_model_ft_name, train_data, val_data, train_label_dict, options)
        test_stats = test_predictor(cur_model_ft_name, test_data, options)
        all_ft_stats.append(val_stats + test_stats)

    stat_template = "Epoch: {:.3f}, Val Loss: {:.3f}, Acc: {:.3f}, AUC: {:.3f}, Kap: {:.3f}, Agg: {:.3f}, Test Loss: {:.3f}, Acc: {:.3f}, AUC: {:.3f}, Kap: {:.3f}, Agg: {:.3f}"

    all_frozen_stats_np = np.array(all_frozen_stats)
    print("\nFrozen Average:")
    print(stat_template.format(*all_frozen_stats_np.mean(axis=0)))
    print("Frozen Std:")
    print(stat_template.format(*all_frozen_stats_np.std(axis=0)))

    all_ft_stats_np = np.array(all_ft_stats)
    print("FT Average:")
    print(stat_template.format(*all_ft_stats_np.mean(axis=0)))
    print("FT Std:")
    print(stat_template.format(*all_ft_stats_np.std(axis=0)))

    # Report validation and test AUC for latex table
    all_stats_np = np.concatenate([all_frozen_stats_np[:, [3, 8]], all_ft_stats_np[:, [3, 8]]], axis=1)
    stat_template = "& {:.3f} " * all_stats_np.shape[1]
    for idx, stats in enumerate(all_stats_np):
        print(idx + 1, stat_template.format(*stats))
    print("Average", stat_template.format(*all_stats_np.mean(axis=0)))
    print("Std.", stat_template.format(*all_stats_np.std(axis=0)))

def get_ppl_performance(output_filename: str):
    """
    Get performance metrics from output of PPL submission to NAEP Competition
    """
    data = pandas.read_csv(output_filename)
    for test_filename in ["data/test_data_10.json", "data/test_data_20.json", "data/test_data_30.json"]:
        with open(test_filename) as test_file:
            student_ids = {seq["student_id"] for seq in json.load(test_file)}
        idx = data["STUDENTID"].isin(student_ids)
        labels = data["EfficientlyCompletedBlockB"][idx]
        preds = data["prob"][idx]
        auc = roc_auc_score(labels, preds)
        adj_auc = 2 * (auc - .5)
        preds[preds < .5] = 0
        preds[preds >= .5] = 1
        accuracy = accuracy_score(labels, preds)
        kappa = cohen_kappa_score(labels, preds)
        agg = adj_auc + kappa
        print(f"Accuracy: {accuracy:.3f}, Adj. AUC: {adj_auc:.3f}, Kappa: {kappa:.3f}, Agg: {agg:.3f}")
