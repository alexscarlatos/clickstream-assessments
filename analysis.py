import random
import math
from typing import List
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import manifold, decomposition
from scipy.stats import pearsonr

from model import LSTMModel
from joint_model import JointModel
from irt.irt_training import get_model_for_testing, get_processed_dataset, get_all_problem_qids
from training import get_data, get_labels
from data_processing import load_type_mappings, get_problem_qids
from data_loading import Dataset, Collator
from per_question_data_loading import PerQuestionDataset, PerQuestionCollator
from constants import TrainOptions, Mode, Correctness
from utils import device


def reduce_latent_states(latent_states: np.array, sample_rate: float, perplexity: int = 15):
    # Map latent states to 2D space
    print("Performing Dimension Reduction")
    algo = "tsne"
    if algo == "pca":
        transformer = decomposition.PCA(2)
    elif algo == "mds":
        transformer = manifold.MDS(2)
    elif algo == "tsne":
        transformer = manifold.TSNE(2, perplexity=perplexity, learning_rate="auto", n_iter=1000, init="pca", random_state=221)
    reduced_states = transformer.fit_transform(latent_states)

    # Randomly downsample rendered points to reduce clutter
    print("Downsampling")
    sample_idxs = random.sample(range(len(reduced_states)), int(sample_rate * len(reduced_states)))
    reduced_states = reduced_states[sample_idxs]
    return reduced_states, sample_idxs

def render_scatter_plots(data: np.array, reduced_states: np.array, plot_configs: list, labels: List[int] = None):
    """
    Register the plots based on their definitions
    labels - will render check and x marks for 1 and 0 values repsectively. note that this causes the click callback results to be incorrect.
    """

    num_cols = 2 if len(plot_configs) > 1 else 1
    num_rows = math.ceil(len(plot_configs) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols)
    fig.suptitle("Dimension-Reduced Latent State Vectors")
    if labels is not None:
        correct_states = reduced_states[labels == 1]
        incorrect_states = reduced_states[labels == 0]
    for plot_idx, (title, cmap, c_vec, legend_map) in enumerate(plot_configs):
        ax = axes if len(plot_configs) == 1 else axes[plot_idx] if num_rows == 1 else axes[plot_idx // num_cols, plot_idx % num_cols]
        add_args = {} if isinstance(cmap, str) else {"vmin": 0, "vmax": len(cmap.colors)}
        if labels is not None:
            scatter = ax.scatter(correct_states[:,0], correct_states[:,1], s=100, c=c_vec[labels == 1], cmap=cmap, picker=True, pickradius=5, marker="$\u2713$", **add_args)
            scatter = ax.scatter(incorrect_states[:,0], incorrect_states[:,1], s=100, c=c_vec[labels == 0], cmap=cmap, picker=True, pickradius=5, marker="x", **add_args)
        else:
            scatter = ax.scatter(reduced_states[:,0], reduced_states[:,1], s=15, c=c_vec, cmap=cmap, picker=True, pickradius=5, **add_args)
        artists, legend_labels = scatter.legend_elements()
        if legend_map:
            legend_labels = [legend_map[label] for label in legend_labels]
        ax.legend(artists, legend_labels, loc='lower left', title=title)#, bbox_to_anchor=(1, 1))

    # Define click handler - print information associated with clicked point
    def onpick(event):
        ind = event.ind
        print(data[ind[0]]['student_id'])
    fig.canvas.mpl_connect('pick_event', onpick)

    # Render the plots
    plt.show()

def cluster(model_name: str, data_file: str, options: TrainOptions):
    type_mappings = load_type_mappings()

    # Load data
    data = get_data(data_file or "data/train_data.json")
    data = [seq for seq in data if max(seq["time_deltas"]) < 2000] # Remove time outliers
    labels = get_labels(options.task, True)

    if options.per_q_arch:
        block_a_qids = {qid for _, qid in get_problem_qids("A", type_mappings)}
        data_loader = torch.utils.data.DataLoader(
            PerQuestionDataset(data, labels, block_a_qids, False, True, options.concat_visits, options.use_correctness),
            collate_fn=PerQuestionCollator(block_a_qids),
            batch_size=len(data)
        )
        num_labels = len(get_problem_qids("B", type_mappings)) if options.task == "q_stats" else 1
        model = JointModel(Mode.CLUSTER, type_mappings, options, num_labels=num_labels, num_input_qids=len(block_a_qids)).to(device)
        model.load_params(torch.load(f"{model_name}.pt", map_location=device))
    else:
        data_loader = torch.utils.data.DataLoader(
            Dataset(data, type_mappings, labels=labels),
            collate_fn=Collator(),
            batch_size=len(data)
        )
        model = LSTMModel(Mode.CLUSTER, load_type_mappings(), options).to(device)
        model.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))

    # Extract latent state with label and prediction for each sequence in the dataset
    print("Extracting latent states")
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            latent_states, predictions = model(batch)
            latent_states = latent_states.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()
            predictions[predictions > 0] = 1
            predictions[predictions < 0] = 0
            labels = batch["labels"].detach().cpu().numpy()

    # Get dimension-reduced latent states
    reduced_states, sample_idxs = reduce_latent_states(latent_states, 1, perplexity=30)
    data = np.array(data)[sample_idxs]
    labels = labels[sample_idxs]
    predictions = predictions[sample_idxs]

    # Define the plots to be shown
    bin_cmap = ListedColormap(["red", "blue"])
    bin_label_map = {
        "$\\mathdefault{0}$": "Below Average",
        "$\\mathdefault{1}$": "Above Average"
    }
    num_visited_questions = [len([qs for qs in seq["q_stats"].values() if qs["visits"]]) for seq in data]
    plots = [
        ("Label", bin_cmap, labels, bin_label_map), # Good
        # ("Prediction", bin_cmap, predictions, None), # Good
        # ("Block A Score", "viridis", [seq["block_a_score"] for seq in data], None), # Good
        # ("Num Events", "viridis", [math.log10(len(seq["event_types"])) for seq in data], None), # Good
        # ("Questions Attempted", "viridis", [sum(qs["correct"] != Correctness.INCOMPLETE.value for qs in seq["q_stats"].values()) for seq in data], None), # Good
        # ("Questions Visited", "viridis", num_visited_questions, None), # Very similar to questions attempted
        # ("Total Time", "viridis", [min(max(seq["time_deltas"]), 1800) for seq in data], None), # Good
        # ("Avg. Visits", "viridis", [sum(qs["visits"] for qs in seq["q_stats"].values()) / len([qs for qs in seq["q_stats"].values() if qs["visits"]]) for seq in data], None), # Not Great
        # ("Num Visits", "viridis", [sum(qs["visits"] for qs in seq["q_stats"].values()) for seq in data], None), # Good
        # ("Avg. Time Spent", "viridis", [
        #     min(
        #         sum(qs["time"] for qs in seq["q_stats"].values()) / num_visited_questions[seq_idx],
        #         1800 / num_visited_questions[seq_idx] # Min since some sequences have messed up timestamps and we don't want outliers
        #     )
        #     for seq_idx, seq in enumerate(data)
        # ], None), # Not Great
        # ("Std. Time Spent", "viridis", [np.array([qs["time"] for qs in seq["q_stats"].values()]).std() for seq in data], None), # Not Great
    ]

    # Render the plots
    render_scatter_plots(data, reduced_states, plots)

def cluster_irt(model_name: str, data_file: str, options: TrainOptions):
    # Get dataset
    type_mappings = load_type_mappings()
    src_data = get_data(data_file)
    problem_qids = {2} # MCSS
    # problem_qids = {3} # MCSS
    # problem_qids = {4} # MatchMS
    # problem_qids = {7} # MultipleFillInBlank
    # problem_qids = {13} # CompositeCR - FillInBlank
    # problem_qids = {14} # FillInBlank
    # problem_qids = {27} # GridMS
    # problem_qids = {30} # MCSS
    # problem_qids = {36} # ZonesMS
    # problem_qids = {37} # CompositeCR - ZonesMS, MultipleFillInBlank
    # problem_qids = {qid for _, qid in get_problem_qids("A", type_mappings) + get_problem_qids("B", type_mappings)}
    data = get_processed_dataset(src_data, type_mappings, False, problem_qids, options.concat_visits)
    data_loader = torch.utils.data.DataLoader(
        data,
        collate_fn=Collator(),
        batch_size=options.batch_size
    )

    # Get model
    model = get_model_for_testing(Mode.CLUSTER, model_name, type_mappings, True, False, None, None, options)

    # Extract latent state with label, behavior scalar and prediction for each entry in the dataset
    print("Extracting latent states")
    ls_batches = []
    bv_batches = []
    pred_batches = []
    label_batches = []
    with torch.no_grad():
        for batch in data_loader:
            latent_states, behavior, predictions = model(batch)
            ls_batches.append(latent_states.detach().cpu().numpy())
            bv_batches.append(behavior.detach().cpu().numpy())
            pred_batches.append(predictions.detach().cpu().numpy())
            label_batches.append(batch["labels"].detach().cpu().numpy())
    latent_states = np.concatenate(ls_batches, axis=0)
    behavior = np.concatenate(bv_batches, axis=0)
    predictions = np.concatenate(pred_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    neg_bv = sorted(behavior[behavior <= 0])
    pos_bv = sorted(behavior[behavior > 0])
    neg_bv_cutoff = neg_bv[len(neg_bv) // 2] if neg_bv else 0
    pos_bv_cutoff = pos_bv[len(pos_bv) // 2] if pos_bv else 0
    predictions[predictions < 0] = 0
    predictions[predictions > 0] = 1

    # Get dimension-reduced latent states
    reduced_states, sample_idxs = reduce_latent_states(latent_states, 0.1, perplexity=30)
    data = torch.utils.data.Subset(data, sample_idxs)
    labels = labels[sample_idxs]
    predictions = predictions[sample_idxs]
    behavior = behavior[sample_idxs]

    # Define the plots to be shown
    bin_cmap = ListedColormap(["red", "blue"])
    quad_cmap = ListedColormap([(1, 0, 0), (1, .5, .5), (.5, .5, 1), (0, 0, 1)])
    quad_label_map = {
        "$\\mathdefault{0}$": "< 50% Neg.",
        "$\\mathdefault{1}$": "> 50% Neg.",
        "$\\mathdefault{2}$": "< 50% Pos.",
        "$\\mathdefault{3}$": "> 50% Pos."
    }
    plots = [
        ("Label", bin_cmap, labels, None),
        # ("Prediction", bin_cmap, predictions, None),
        ("Behavior Scalar", quad_cmap, np.array([0 if bv < neg_bv_cutoff else 1 if bv < 0 else 2 if bv < pos_bv_cutoff else 3 for bv in behavior]), quad_label_map),
        # ("Behavior Scalar", "viridis", behavior, None),
        ("Visits", "viridis", [entry["num_visits"] for entry in data], None),
        ("Time Spent", "viridis", [entry["total_time"] for entry in data], None),
        ("Num Events", "viridis", [len(entry["event_types"]) for entry in data], None),
        ("Max Time Gap", "viridis", [entry["max_gap"] for entry in data], None)
    ]

    # Render the plots
    render_scatter_plots(data, reduced_states, plots)#, labels=labels)

def visualize_irt(model_name: str, data_file: str, use_behavior_model: bool, options: TrainOptions):
    # Get dataset
    type_mappings = load_type_mappings()
    src_data = get_data(data_file)
    problem_qids = get_all_problem_qids(type_mappings)
    data = get_processed_dataset(src_data, type_mappings, False, problem_qids, options.concat_visits)
    data_loader = torch.utils.data.DataLoader(
        data,
        collate_fn=Collator(),
        batch_size=1000
    )

    # Get model
    model = get_model_for_testing(Mode.CLUSTER, model_name, type_mappings, use_behavior_model, False, None, None, options)

    # Get abilities and behavior scalars for each student
    include_behavior = True
    include_behavior = include_behavior and use_behavior_model
    if include_behavior:
        bv_batches = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                print(batch_idx)
                _, behavior, _ = model(batch)
                bv_batches.append(behavior.detach().cpu().numpy())
        behavioral_values = np.concatenate(bv_batches, axis=0)
    abilities = torch.nn.Softplus()(model.ability).detach().cpu().numpy()
    student_to_param = {}
    for idx, entry in enumerate(data):
        student_to_param.setdefault(entry["student_id"], 0 if include_behavior else abilities[entry["sid"]])
        if include_behavior:
            student_to_param[entry["student_id"]] += abilities[entry["sid"]] + behavioral_values[idx]

    student_params = [param for param in student_to_param.values()]
    student_to_score = {seq["student_id"]: seq["block_a_score"] + seq["block_b_score"] for seq in src_data}
    scores = [student_to_score[student_id] for student_id in student_to_param.keys()]

    # Equivalent calculation for non-behavior model for verification
    # student_idxs = [type_mappings["student_ids"][str(seq["student_id"])] for seq in src_data]
    # student_params = torch.nn.Softplus()(model.ability).detach().cpu().numpy()[student_idxs]
    # scores = [seq["block_a_score"] + seq["block_b_score"] for seq in src_data]

    print("Student to score correlation", pearsonr(student_params, scores))

    # Plot learned ability vs. student score over both blocks
    plt.plot(student_params, scores, "bo")
    plt.xlabel("Learned Ability")
    plt.ylabel("Total Score")
    plt.show()

    # Plot question difficulty vs. avg correctness
    qid_to_score = {}
    qid_to_num = {}
    for entry in data:
        qid = entry["question_id"]
        qid_to_score.setdefault(qid, 0)
        qid_to_score[qid] += 1 if entry["correct"] else 0
        qid_to_num.setdefault(qid, 0)
        qid_to_num[qid] += 1
    difficulties = torch.nn.Softplus()(model.difficulty).detach().cpu().numpy()[list(qid_to_score.keys())]
    avg_score = [qid_to_score[qid] / qid_to_num[qid] for qid in qid_to_score]
    print("Difficulty to score correlation", pearsonr(difficulties, avg_score))
    plt.plot(difficulties, avg_score, "ro")
    plt.xlabel("Learned Difficulty")
    plt.ylabel("Average Score")
    plt.show()

def block_scores(data_file: str):
    src_data = get_data(data_file)

    score_to_num = {}
    block_a_score = []
    block_b_score = []
    for seq in src_data:
        bas = seq["block_a_score"]
        bbs = seq["block_b_score"]
        score_to_num.setdefault((bas, bbs), 0)
        score_to_num[(bas, bbs)] += 1
        block_a_score.append(bas)
        block_b_score.append(bbs)
    print("Block A/B score correlation", pearsonr(block_a_score, block_b_score))

    block_a_score = [bas for bas, _ in score_to_num.keys()]
    block_b_score = [bbs for _, bbs in score_to_num.keys()]
    sizes = [num for num in score_to_num.values()]
    plt.scatter(block_a_score, block_b_score, s=sizes, c="blue")
    plt.xlabel("Block A Score")
    plt.ylabel("Block B Score")
    plt.show()
