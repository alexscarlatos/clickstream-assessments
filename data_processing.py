import json
import re
from typing import Dict, List
import pandas
from constants import ASSISTIVE_EVENT_IDS, Correctness
from collections import Counter

def save_type_mappings(data_file: str):
    sep = "," if data_file.endswith(".csv") else "\t"
    src_data = pandas.read_csv(data_file, sep=sep, encoding="latin-1")
    unique_students = {int(student): idx for idx, student in enumerate(src_data["STUDENTID"].unique())}
    unique_questions = {question: idx for idx, question in enumerate(src_data["AccessionNumber"].unique())}
    unique_question_types = {question_type: idx for idx, question_type in enumerate(src_data["ItemType"].unique())}
    unique_event_types = {event_type: idx for idx, event_type in enumerate(src_data["Observable"].unique())}

    with open("ref_data/types.json", "w") as types_file:
        json.dump({
            "student_ids": unique_students,
            "question_ids": unique_questions,
            "question_types": unique_question_types,
            "event_types": unique_event_types
        }, types_file, indent=4)

    event_types_per_question = {}
    for qid_str, qid in unique_questions.items():
        event_types_per_question[qid] = [
            unique_event_types[event_type] for event_type in src_data[src_data["AccessionNumber"] == qid_str]["Observable"].unique()
        ]

    with open("ref_data/event_types_per_question.json", "w") as types_file:
        json.dump(event_types_per_question, types_file, indent=4)

def load_type_mappings() -> Dict[str, Dict[str, int]]:
    with open("ref_data/types.json") as type_mapping_file:
        return json.load(type_mapping_file)

def load_event_types_per_question() -> Dict[str, List[int]]:
    with open("ref_data/event_types_per_question.json") as types_file:
        return json.load(types_file)

def load_question_info() -> Dict[str, dict]:
    with open("ref_data/question_info.json") as qa_file:
        return json.load(qa_file)

def get_problem_qids(block: str, type_mappings: dict):
    q_info_dict = load_question_info()
    return [
        (qid_str, type_mappings["question_ids"][qid_str])
        for qid_str, q_info in q_info_dict.items() if q_info["block"] == block and q_info["answer"] != "na"
    ]

mcss_re = re.compile("^VH.{6}_(\d+)")
gridms_re = re.compile("^VH.{6}-(\d+)-(\d+):(checked|unchecked)")
zonesms_re = re.compile("^VH.{6}-(\d+):(checked|unchecked)")
compcr_cc_re = re.compile("^VH.{6}(.)-(\d+)?:(checked|unchecked)")
non_num_re = re.compile("[^\d\./]")

def code_to_char(code: str):
    if code.startswith("Digit"):
        return code[5:]
    if code.startswith("Key"):
        return code[3:]
    if code == "Period":
        return "."
    if code == "Slash":
        return "/"
    # There are many more codes, but none are included in any of the correct answers
    return ""

def fill_in_blank_correct(answer, response):
    if isinstance(answer, list):
        return response in answer or non_num_re.sub('', response) in answer
    else:
        return response == answer or non_num_re.sub('', response) == answer

class QuestionAnswerState:
    def __init__(self, qid, answer_key):
        self.qid = qid
        self.type = answer_key[qid]["type"]
        self.answer = answer_key[qid]["answer"]
        self.state = None

    def process_event(self, event: Dict[str, str]):
        if event["Observable"] == "Clear Answer":
            self.state = None
            return

        if self.type == "MCSS":
            if event["Observable"] == "Click Choice":
                self.state = mcss_re.match(event["ExtendedInfo"]).group(1)

        elif self.type == "MatchMS":
            if event["Observable"] == "DropChoice":
                choices = json.loads(event["ExtendedInfo"])
                self.state = {str(choice["target"]): choice["source"] for choice in choices}

        elif self.type == "MultipleFillInBlank":
            if self.state is None:
                self.state = {}

            if event["Observable"] == "Math Keypress":
                entry = json.loads(event["ExtendedInfo"])
                base_entry = entry["contentLaTeX"][1:-1] # Strip $ from first and last pos
                if "code" not in entry:
                    print("no code found", entry)
                self.state[entry["numericIdentifier"]] = base_entry + code_to_char(entry["code"])

        elif self.type == "FillInBlank":
            if event["Observable"] == "Math Keypress":
                entry = json.loads(event["ExtendedInfo"])
                base_entry = entry["contentLaTeX"][1:-1] # Strip $ from first and last pos
                if "code" not in entry:
                    print("no code found", entry)
                self.state = base_entry + code_to_char(entry["code"])

        elif self.type == "CompositeCR":
            if self.state is None:
                self.state = {}

            if event["Observable"] == "Click Choice":
                part, choice, action = compcr_cc_re.match(event["ExtendedInfo"]).groups()
                if self.answer[part]["type"] == "MCSS":
                    self.state[part] = choice
                elif self.answer[part]["type"] == "ZonesMS":
                    if part not in self.state:
                        self.state[part] = set()
                    if action == "checked":
                        self.state[part].add(choice)
                    else:
                        self.state[part].remove(choice)

            elif event["Observable"] == "Math Keypress":
                entry = json.loads(event["ExtendedInfo"])
                part = entry["partId"]
                if part not in self.answer: # some fields (explanations) don't need to be checked in the final answer
                    return

                base_entry = entry["contentLaTeX"][1:-1] # Strip $ from first and last pos
                if "code" not in entry:
                    print("no code found", entry)
                if self.answer[part]["type"] == "FillInBlank":
                    self.state[part] = base_entry + code_to_char(entry["code"])
                elif self.answer[part]["type"] == "MultipleFillInBlank":
                    if part not in self.state:
                        self.state[part] = {}
                    self.state[part][entry["numericIdentifier"]] = base_entry + code_to_char(entry["code"])

        elif self.type == "GridMS":
            if event["Observable"] == "Click Choice":
                if self.state is None:
                    self.state = {}

                part, choice, action = gridms_re.match(event["ExtendedInfo"]).groups()
                if action == "checked":
                    self.state[part] = choice
                else:
                    if part in self.state:
                        del self.state[part] # del instead of set to None so we can use len to check for completeness

        elif self.type == "ZonesMS":
            if event["Observable"] == "Click Choice":
                if self.state is None:
                    self.state = set()

                choice, action = zonesms_re.match(event["ExtendedInfo"]).groups()
                if action == "checked":
                    self.state.add(choice)
                else:
                    self.state.remove(choice)

    def get_state_string(self) -> str:
        if isinstance(self.state, dict):
            sorted_state = []
            for part, substate in sorted(self.state.items()):
                if isinstance(substate, dict):
                    substate_sorted = sorted(substate.items())
                elif isinstance(substate, set):
                    substate_sorted = sorted(substate)
                else:
                    substate_sorted = substate
                sorted_state.append((part, substate_sorted))
            state = str(sorted_state)
        elif isinstance(self.state, set):
            state = str(sorted(self.state))
        else:
            state = str(self.state)

        if self.get_correctness_label() == Correctness.CORRECT:
            state = "CORRECT: " + state
        return state

    def get_correctness_label(self) -> Correctness:
        if not self.state:
            return Correctness.INCOMPLETE

        if self.type == "MCSS":
            return Correctness.CORRECT if self.answer == self.state else Correctness.INCORRECT

        if self.type == "MatchMS":
            if len(self.state) != len(self.answer[0]):
                return Correctness.INCOMPLETE
            for possible_answer in self.answer:
                if self.state == possible_answer:
                    return Correctness.CORRECT
            return Correctness.INCORRECT

        if self.type == "MultipleFillInBlank":
            if len(self.state) != len(self.answer):
                return Correctness.INCOMPLETE
            correct = True
            for part, answer in self.answer.items():
                correct = correct and fill_in_blank_correct(answer, self.state[part])
            return Correctness.CORRECT if correct else Correctness.INCORRECT

        if self.type == "FillInBlank":
            return Correctness.CORRECT if fill_in_blank_correct(self.answer, self.state) else Correctness.INCORRECT

        if self.type == "CompositeCR":
            if len(self.state) != len(self.answer):
                return Correctness.INCOMPLETE

            correct = True
            for part in self.answer:
                response, answer, part_type = self.state[part], self.answer[part]["answer"], self.answer[part]["type"]
                if part_type == "FillInBlank":
                    correct = correct and fill_in_blank_correct(answer, response)

                elif part_type == "MultipleFillInBlank":
                    if len(response) != len(answer):
                        return Correctness.INCOMPLETE

                    for subpart, subpart_answer in answer.items():
                        correct = correct and fill_in_blank_correct(subpart_answer, response[subpart])

                elif part_type == "MCSS":
                    correct = correct and response == answer

                elif part_type == "ZonesMS":
                    correct = correct and sorted(response) == answer

            return Correctness.CORRECT if correct else Correctness.INCORRECT

        if self.type == "GridMS":
            if len(self.state) != len(self.answer):
                return Correctness.INCOMPLETE
            return Correctness.CORRECT if self.answer == self.state else Correctness.INCORRECT

        if self.type == "ZonesMS":
            return Correctness.CORRECT if self.answer == sorted(self.state) else Correctness.INCORRECT

        # Always return incomplete for all other (non-question) types
        return Correctness.INCOMPLETE

def convert_raw_data_to_json(data_filenames: List[str], output_filename: str, block: str = None, trim_after: List[float] = None, data_classes: List[str] = None):
    if trim_after:
        assert len(trim_after) == len(data_filenames)
    if data_classes:
        assert len(data_classes) == len(data_filenames)

    type_mappings = load_type_mappings()
    qa_key = load_question_info()
    final_data = []

    # Process data set - each sequence is list of events per student
    for file_idx, data_file in enumerate(data_filenames):
        print("Processing", data_file)
        student_to_sequences: Dict[int, dict] = {}
        student_to_qa_states: Dict[int, Dict[str, QuestionAnswerState]] = {}
        student_to_q_visits: Dict[int, Dict[str, List[List[float]]]] = {}

        sep = "," if data_file.endswith(".csv") else "\t"
        src_data = pandas.read_csv(data_file, parse_dates=["EventTime"], sep=sep, encoding="latin-1").sort_values(["STUDENTID", "EventTime"])
        for _, event in src_data.iterrows():
            if block and event["Block"] != block:
                continue

            # This event messes with visit continuity, and also allows QID prediction to cheat
            if event["Observable"] == "Click Progress Navigator":
                continue

            # Skip entries with no timestamp
            if pandas.isnull(event["EventTime"]):
                print("Skipping event, no timestamp", event)
                continue

            sequence: Dict[str, list] = student_to_sequences.setdefault(event["STUDENTID"], {
                "data_class": data_classes[file_idx] if data_classes else None,
                "student_id": event["STUDENTID"],
                "question_ids": [],
                "question_types": [],
                "event_types": [],
                "time_deltas": [],
                "correctness": [],
                "assistive_uses": {eid: 0 for eid in ASSISTIVE_EVENT_IDS},
                "q_stats": {},
                "block_a_score": 0,
                "block_b_score": 0
            })

            if not sequence["event_types"]:
                start_time = event["EventTime"]
            time_delta = (event["EventTime"] - start_time).total_seconds()
            if trim_after and time_delta > trim_after[file_idx]:
                continue

            qid = event["AccessionNumber"]
            eid = type_mappings["event_types"][event["Observable"]]

            # Update number of times assistives were used
            if eid in ASSISTIVE_EVENT_IDS:
                sequence["assistive_uses"][eid] += 1

            # Update state of student's answer to the current question
            qa_states = student_to_qa_states.setdefault(event["STUDENTID"], {
                qid: QuestionAnswerState(qid, qa_key) for qid in type_mappings["question_ids"]
            })
            qa_state = qa_states[qid]
            qa_state.process_event(event)

            # Keep track of visits to each question
            if event["STUDENTID"] not in student_to_q_visits:
                cur_question_id = None
            qid_to_visits = student_to_q_visits.setdefault(event["STUDENTID"], {
                qid: [] for qid in type_mappings["question_ids"]
            })
            q_visits = qid_to_visits[qid]
            if qid != cur_question_id: # If we went to a new question, start a new visit
                q_visits.append([time_delta, time_delta])
                cur_question_id = qid
            else: # Update current visit
                q_visits[-1][1] = time_delta # Update current end time

            # Append per-event data
            sequence["question_ids"].append(type_mappings["question_ids"][qid])
            sequence["question_types"].append(type_mappings["question_types"][event["ItemType"]])
            sequence["event_types"].append(eid)
            sequence["time_deltas"].append(time_delta)
            sequence["correctness"].append(qa_state.get_correctness_label().value)

        # Final processing based on per-student question data
        qids_to_track = [qid for qid, q_info in qa_key.items() if (not block or q_info["block"] == block) and q_info["answer"] != "na"]
        for student_id, seq in student_to_sequences.items():
            q_stats = seq["q_stats"]
            for qid in qids_to_track:
                q_visits = student_to_q_visits[student_id][qid]
                qa_state = student_to_qa_states[student_id][qid]
                q_correctness = qa_state.get_correctness_label()
                q_stats[qid] = {
                    "time": sum(end_time - start_time for start_time, end_time in q_visits),
                    "visits": len(q_visits),
                    "correct": q_correctness.value,
                    "final_state": qa_state.get_state_string()
                }

                if q_correctness == Correctness.CORRECT:
                    block = qa_key[qid]["block"]
                    if block == "A":
                        seq["block_a_score"] += 1
                    elif block == "B":
                        seq["block_b_score"] += 1

        final_data += list(student_to_sequences.values())

    with open(output_filename, "w") as output_file:
        json.dump(final_data, output_file, indent=2)

def analyze_processed_data(data_filename: str):
    type_mappings = load_type_mappings()

    with open(data_filename) as data_file:
        data = json.load(data_file)

    # Sanity check
    for seq in data[:10]:
        print(seq["student_id"])
        print(json.dumps(seq["q_stats"], indent=4))
        print("")

    # Look at answers for top students
    qid_to_answers: Dict[str, Counter] = {qid: Counter() for qid in type_mappings["question_ids"]}
    qid_to_num_correct: Dict[str, int] = {qid: 0 for qid in type_mappings["question_ids"]}
    top_students = sorted(data, key=lambda seq: -seq["block_a_score"])[:100]
    for seq in top_students:
        for qid, q_stats in seq["q_stats"].items():
            qid_to_answers[qid][q_stats["final_state"]] += 1
            if q_stats["correct"] == Correctness.CORRECT.value:
                qid_to_num_correct[qid] += 1

    # Show top responses and correctness for each question
    for qid, qa_counter in qid_to_answers.items():
        print(f"---- {qid} ({100 * qid_to_num_correct[qid] / len(top_students):.0f}% correct) ----")
        for count in qa_counter.most_common():
            print(count)
        print("")

    # Show average scores across both blocks
    total_block_a_score = sum(seq["block_a_score"] for seq in data)
    total_block_b_score = sum(seq["block_b_score"] for seq in data)
    avg_block_a_score, avg_block_b_score = total_block_a_score / len(data), total_block_b_score / len(data)
    print(f"Avg block A score: {avg_block_a_score:.3f}, Avg block B score: {avg_block_b_score:.3f}")

def convert_raw_labels_to_json(data_filename: str, output_filename: str):
    src_data = pandas.read_csv(data_filename)

    student_to_label = {}
    for _, event in src_data.iterrows():
        student_to_label[event["STUDENTID"]] = event["EfficientlyCompletedBlockB"]

    with open(output_filename, "w") as output_file:
        json.dump(student_to_label, output_file)

def gen_score_label(data_filename: str, out_filename: str):
    with open(data_filename) as data_file:
        data = json.load(data_file)

    # Label is if student had higher or lower than average score on block B
    total_block_b_score = sum(seq["block_b_score"] for seq in data)
    avg_block_b_score = total_block_b_score / len(data)
    student_to_label = {seq["student_id"]: seq["block_b_score"] > avg_block_b_score for seq in data}
    with open(out_filename, "w") as out_file:
        json.dump(student_to_label, out_file, indent=2)

    # Write labels in same format as original label files, for compatibility with PPL code
    # labels = pandas.read_csv("src_data/data_train_label.csv")
    # labels["EfficientlyCompletedBlockB"] = [student_to_label[student["STUDENTID"]] for _, student in labels.iterrows()]
    # labels.to_csv("data_train_label_alt.csv", index=False)
    # labels = pandas.read_csv("src_data/hidden_label.csv")
    # labels["EfficientlyCompletedBlockB"] = [student_to_label[student["STUDENTID"]] for _, student in labels.iterrows()]
    # labels.to_csv("hidden_label_alt.csv", index=False)

def gen_per_q_stat_label(data_filename: str, out_filename: str):
    with open(data_filename) as data_file:
        data = json.load(data_file)

    type_mappings = load_type_mappings()
    block_b_qids = [qid_str for qid_str, _ in get_problem_qids("B", type_mappings)]

    # Label is correctness and time spent for each question in block B
    student_to_label = {
        seq["student_id"]: [seq["q_stats"][qid]["correct"] == Correctness.CORRECT.value for qid in block_b_qids]
        for seq in data
    }
    with open(out_filename, "w") as out_file:
        json.dump(student_to_label, out_file, indent=2)
