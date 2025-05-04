import yaml

# Custom representer to handle YAML tags like !function
class LiteralTag(str):
    pass

def literal_tag_representer(dumper, data):
    # Represent the value as a YAML tag without quotes
    return dumper.represent_scalar('!function', data)

yaml.add_representer(LiteralTag, literal_tag_representer)

# Shared configuration for YAML files
hellaswag_config = {
    "tag": ["multiple_choice"],  # Ensure tag is a list
    "task": "hellaswag",
    "dataset_path": "hellaswag",
    "dataset_name": None,
    "output_type": "multiple_choice",
    "training_split": "train",
    "validation_split": "validation",
    "test_split": None,
    "process_docs": LiteralTag("utils.process_docs"),  # Use LiteralTag here
    "doc_to_text": "{{query}}",  # Ensure no unnecessary escaping
    "doc_to_target": "{{label}}",
    "doc_to_choice": "choices",  # Ensure no unnecessary escaping
    "metric_list": [
        {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
        {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True}
    ],
    "metadata": {"version": 1.0},
    "dataset_kwargs": {
        "trust_remote_code": True
    },
}

# Unique configurations for each YAML file
tasks = [
    "hellaswag_1",
    "hellaswag_2",
    "hellaswag_3",
    "hellaswag_4",
    "hellaswag_5",
    "hellaswag_6",
    "hellaswag_7",
    "hellaswag_8",
    "hellaswag_9",
    "hellaswag_10",
    "hellaswag_11",
    "hellaswag_12",
    "hellaswag_13",
    "hellaswag_14",
]

doc_to_texts = [
    "Write your answer and give me a confidence score between 0-1 for your answer. {{query}}", # 01
    "This is very important to my career. {{query}}", # 02
    "You'd better be sure. {{query}}", # 03
    "Are you sure? {{query}}", # 04
    "Are you sure that's your final answer? It might be worth taking another look. {{query}}", # 05
    "Trust yourself. {{query}}", # 06
    "Don't trust yourself. {{query}}", # 07
    "It is okay to not know the answer. {{query}}", # 08
    "Take a moment to think carefully before answering. Consider all possibilities: {{query}}", # 09
    "Imagine explaining this to a friend who doesn't know the topic. What would you say? {{query}}", # 10
    "Think step by step and explain your reasoning before answering: {{query}}", # 11
    "What evidence supports your answer? Be as specific as possible: {{query}}", # 12
    "If you had to bet on this answer, how confident would you be? {{query}}", # 13
    "What are the possible answers, and why is your choice the best? {{query}}", # 14
]

output_files = [
    "template1.yaml",
    "template2.yaml",
    "template3.yaml",
    "template4.yaml",
    "template5.yaml",
    "template6.yaml",
    "template7.yaml",
    "template8.yaml",
    "template9.yaml",
    "template10.yaml",
    "template11.yaml",
    "template12.yaml",
    "template13.yaml",
    "template14.yaml",
]

def generate_yaml_files():
    for i in range(len(output_files)):
        config = {
            **hellaswag_config,
            "task": tasks[i],  # Ensure task is a single string, not a list
            "doc_to_text": doc_to_texts[i]  # Ensure proper formatting of doc_to_text
        }
        with open(output_files[i], "w") as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
        print(f"YAML file '{output_files[i]}' has been created.")

def main() -> None:
    generate_yaml_files()

if __name__ == "__main__":
    main()