import re

import datasets


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [f"{preprocess(ending)} Wait, let me think again. The answer is {preprocess(ending)}" for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)

def process_docs_2(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [f"{preprocess(ending)} Wait, let me think again. The answer is \"{preprocess(ending)}\"" for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)

def process_docs_3(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Processes HellaSwag documents for a multiple-choice loglikelihood task.
    This function creates an instructional prompt to guide the model's commonsense
    reasoning. The 'choices' are kept clean, as required by lm-harness for
    multiple-choice evaluations.
    """
    def _process_doc(doc: dict) -> dict:
        # Extract HellaSwag specific fields from the input document
        activity_label = doc["activity_label"]  # e.g., "Taking a Shower"
        context_part_a = doc["ctx_a"]           # The first part of the context
        context_part_b = doc["ctx_b"]           # The second part of the context
        
        candidate_endings = doc["endings"]      # List of possible endings for the scenario
        correct_ending_index = int(doc["label"]) # Index of the true ending

        # Construct the full context sentence, capitalizing the first letter of context_part_b
        # HellaSwag contexts often form a sentence when ctx_a and ctx_b are combined.
        full_initial_context = context_part_a + " " + context_part_b.capitalize()

        # This is the core scenario text, which is effective for HellaSwag
        scenario_text = f"{activity_label}: {full_initial_context}"

        # --- Instructional Part of the Prompt ---
        # This instruction aims to guide the model's internal reasoning process
        # for commonsense NLI, which is what HellaSwag tests.
        instruction = (
            "The following is a commonsense reasoning task. "
            "You are given an initial scenario and several possible endings. "
            "Your goal is to choose the ending that most logically and naturally "
            "completes the scenario. Consider the typical sequence of events and "
            "what would make the most sense in a real-world situation."
        )

        # --- Question Part of the Prompt ---
        # Clearly ask the model what to do.
        question = "Which of the following options is the most plausible completion for the scenario?"
        
        # --- Construct the Final Query ---
        # This structure provides clear guidance before presenting the core scenario.
        # The actual choices (A, B, C, D) are not listed in the query here.
        # lm-harness's multiple_choice output_type will take this `final_query`
        # and append each choice from the `choices` list to it for scoring.
        final_query = (
            f"{instruction}\n\n"
            f"Scenario:\n{scenario_text}\n\n"
            f"{question}"
        )
        
        # The output dictionary.
        # "query" is the prompt for the model.
        # "choices" are the clean endings for lm-harness to use.
        # "label" is the index of the correct choice, matching doc_to_target in YAML.
        processed_doc = {
            "query": preprocess(final_query),
            "choices": [preprocess(ending) for ending in candidate_endings],
            "label": correct_ending_index, 
        }
        return processed_doc

    return dataset.map(_process_doc)

def process_docs_4(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Processes HellaSwag documents for a multiple-choice loglikelihood task.
    This function prepends a very short, subtle guiding phrase to the core scenario.
    The aim is to maintain accuracy while potentially improving calibration (ACE)
    by gently framing the task. Choices are kept clean.
    """
    def _process_doc(doc: dict) -> dict:
        # Extract HellaSwag specific fields
        activity_label = doc["activity_label"]
        context_part_a = doc["ctx_a"]
        context_part_b = doc["ctx_b"]
        candidate_endings = doc["endings"]
        correct_ending_index = int(doc["label"])

        # Construct the core scenario text (this was likely the best performing "raw" prompt)
        full_initial_context = context_part_a + " " + context_part_b.capitalize()
        core_scenario_prompt = f"{activity_label}: {full_initial_context}"

        # Prepend a very short, gentle guiding phrase.
        # This phrase aims to frame the task without being overly instructive.
        # "Consider the commonsense completion for:"
        # "Which is the most natural continuation of:"
        # "From the options, the best commonsense fit for the scenario:"
        # Let's try a very neutral one:
        guiding_prefix = "Determine the most logical continuation for the following scenario:"

        final_query = f"{guiding_prefix}\n{preprocess(core_scenario_prompt)}"
        
        processed_doc = {
            "query": final_query, # Preprocessing is already applied to core_scenario_prompt if needed
            "choices": [preprocess(ending) for ending in candidate_endings],
            "label": correct_ending_index, 
        }
        return processed_doc

    return dataset.map(_process_doc)

def process_docs_5(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Processes HellaSwag documents for a multiple-choice loglikelihood task.
    This template uses a concise instruction focusing on "commonsense choice"
    and a direct question, hoping to improve accuracy for smaller models.
    """
    def _process_doc(doc: dict) -> dict:
        # Extract HellaSwag specific fields
        activity_label = doc["activity_label"]
        context_part_a = doc["ctx_a"]
        context_part_b = doc["ctx_b"]
        candidate_endings = doc["endings"]
        correct_ending_index = int(doc["label"])

        # Construct the core scenario text
        full_initial_context = context_part_a + " " + context_part_b.capitalize()
        core_scenario_text = f"{activity_label}: {full_initial_context}"

        # Concise instruction and question
        instructional_prefix = "Use commonsense to choose the most plausible continuation for the scenario below."
        question_suffix = "Which is the best ending?"
        
        # Combine elements. Preprocess the core scenario text.
        final_query = (
            f"{instructional_prefix}\n\n"
            f"Scenario: {preprocess(core_scenario_text)}\n\n"
            f"{question_suffix}"
        )
        
        processed_doc = {
            "query": final_query, # The entire prompt
            "choices": [preprocess(ending) for ending in candidate_endings], # Clean choices
            "label": correct_ending_index, # Correct index
        }
        return processed_doc

    return dataset.map(_process_doc)