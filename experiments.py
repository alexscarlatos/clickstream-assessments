import numpy as np
import pandas
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score

from model import TrainOptions
from training import pretrain, train_predictor, test_predictor, get_data, get_labels
from utils import initialize_seeds

def full_pipeline(pretrained_name: str, model_name: str, options: TrainOptions):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=221)
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

    train_data_len = len(train_data_for_split)
    train_label_dict = get_labels(options.task, True)
    train_labels_for_split = [train_label_dict[str(seq["student_id"])] for seq in train_data_for_split]

    all_frozen_stats = []
    all_ft_stats = []
    for k, (train_data_idx, val_data_idx) in enumerate(skf.split(train_data_for_split, train_labels_for_split)):
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
        options.epochs = 100
        if options.mixed_time:
            # Just pretrain on 30-minute data if using mixed training data
            pretrain(cur_pretrained_name, train_data[len(train_data_idx) * 2:], val_data[len(val_data_idx) * 2:], options)
        else:
            pretrain(cur_pretrained_name, train_data, val_data, options)

        print("\nTrain classifier on frozen LSTM")
        options.lr = 1e-3
        options.epochs = 50
        options.use_pretrained_embeddings = True
        options.use_pretrained_weights = True
        options.use_pretrained_head = False
        options.freeze_embeddings = True
        options.freeze_model = True
        val_stats = train_predictor(cur_pretrained_name, cur_model_name, train_data, val_data, train_label_dict, options)
        test_stats = test_predictor(cur_model_name, test_data_file, options)
        all_frozen_stats.append(val_stats + test_stats)

        print("\nFine-tune LSTM for classifier")
        options.lr = 5e-5
        options.epochs = 50
        options.use_pretrained_head = True
        options.freeze_embeddings = False
        options.freeze_model = False
        val_stats = train_predictor(cur_model_name, cur_model_ft_name, train_data, val_data, train_label_dict, options)
        test_stats = test_predictor(cur_model_ft_name, test_data_file, options)
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
    labels = data["EfficientlyCompletedBlockB"]
    preds = data["prob"]
    auc = roc_auc_score(labels, preds)
    adj_auc = 2 * (auc - .5)
    preds[preds < .5] = 0
    preds[preds >= .5] = 1
    accuracy = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    agg = adj_auc + kappa
    print(f"Accuracy: {accuracy:.3f}, Adj. AUC: {adj_auc:.3f}, Kappa: {kappa:.3f}, Agg: {agg:.3f}")
