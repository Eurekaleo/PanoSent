import numpy as np
import openai


def calculate_precision_recall_f1(correct, predicted, gold):
    precision = correct / predicted if predicted > 0 else 0
    recall = correct / gold if gold > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f1


def evaluate_element_wise(predictions, gold_labels):
    correct_terms = sum(
        1 for pred, gold in zip(predictions, gold_labels) if pred == gold
    )
    predicted_terms = len(predictions)
    gold_terms = len(gold_labels)
    ep, er, exact_f1 = calculate_precision_recall_f1(
        correct_terms, predicted_terms, gold_terms
    )
    return ep, er, exact_f1


def semantic_evaluation_function(
    predicted_term, gold_term, dialogue_context, openai_api_key
):
    openai.api_key = openai_api_key
    prompt = f"Given the context of the dialogue: '{dialogue_context}', do '{predicted_term}' and '{gold_term}' have similar meanings?"
    try:
        response = openai.Completion.create(
            engine="gpt-4-turbo-preview",
            prompt=prompt,
            max_tokens=16,
            temperature=0,
            n=1,
            stop=None,
        )
        response_text = response.get("choices")[0].get("text").strip().lower()
        if "yes" in response_text:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0


def evaluate_implicit_elements(predictions, gold_labels):
    semantically_correct = sum(
        1
        for pred, gold in zip(predictions, gold_labels)
        if semantic_evaluation_function(pred, gold)
    )
    bp, br, binary_f1 = calculate_precision_recall_f1(
        semantically_correct, len(predictions), len(gold_labels)
    )
    return bp, br, binary_f1


def evaluate_explicit_rationale(predictions, gold_labels):
    proportional_correct = sum(
        min(len(pred), len(gold)) / max(len(pred), len(gold))
        for pred, gold in zip(predictions, gold_labels)
    )
    pp, pr, proportional_f1 = calculate_precision_recall_f1(
        proportional_correct, len(predictions), len(gold_labels)
    )
    return pp, pr, proportional_f1


def evaluate_pairwise(predictions, gold_labels):
    correct_pairs = sum(
        1 for pred, gold in zip(predictions, gold_labels) if pred == gold
    )
    pp, pr, pairwise_f1 = calculate_precision_recall_f1(
        correct_pairs, len(predictions), len(gold_labels)
    )
    return pp, pr, pairwise_f1


def evaluate_sextuples(predictions, gold_labels):
    correct_sextuples = sum(
        1 for pred, gold in zip(predictions, gold_labels) if pred == gold
    )
    op, or_, micro_f1 = calculate_precision_recall_f1(
        correct_sextuples, len(predictions), len(gold_labels)
    )
    return op, or_, micro_f1


def evaluate_sentiment_classification(predictions, gold_labels):
    classes = ["positive", "negative", "neutral"]
    class_f1_scores = []
    for cls in classes:
        class_preds = [1 if p == cls else 0 for p in predictions]
        class_gold = [1 if g == cls else 0 for g in gold_labels]
        _, _, class_f1 = calculate_precision_recall_f1(
            sum(class_preds), len(class_preds), len(class_gold)
        )
        class_f1_scores.append(class_f1)
    macro_f1 = np.mean(class_f1_scores)
    return macro_f1
