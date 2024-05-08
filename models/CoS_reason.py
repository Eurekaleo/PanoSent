import torch
from transformers import FlanT5ForConditionalGeneration, FlanT5Tokenizer


class CoSReasoningFramework:
    def __init__(self, model_ckpt_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = FlanT5Tokenizer.from_pretrained(model_ckpt_path)
        self.model = FlanT5ForConditionalGeneration.from_pretrained(model_ckpt_path)
        self.model.to(self.device).eval()

    def generate_response(self, prompt):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        outputs = self.model.generate(**inputs, max_length=512, num_beams=5)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def execute_stage(self, prompt):
        return self.generate_response(prompt)

    def sequential_reasoning(self, dialogue, multimodal_data):
        prompt1 = f"""Based on the multi-party dialogue and its accompanying multimodal data, please identify all possible targets and their specific aspects mentioned in the dialogue. 
    Extract each target and aspect explicitly from the utterance text spans, or infer them implicitly via your understanding of the input data. 
    Ensure each identified target is paired with its aspect(s), forming target-aspect pairs."""
        stage1_result = self.execute_stage(prompt1)

        prompt2 = f"""Based on the dialogue and the target-aspect pairs identified: {stage1_result}, please identify the holder (the person who expresses an opinion, normally should be a speaker of certain dialogue utterance) and the opinion. 
    Formulate your output into holder-target-aspect-opinion quadruples, ensuring each element is clearly identified."""
        stage2_result = self.execute_stage(prompt2)

        prompt3 = f"""Considering the dialogue and the holder-target-aspect-opinion quadruples identified: {stage2_result}, please assess the sentiment polarity associated with each opinion and analyze the causal rationale behind it. 
    The sentiment should be classified as positive, neutral, or negative. The rationale should be extracted explicitly from the text or inferred implicitly via your understanding of the input data.
    Formulate your output into holder-target-aspect-opinion-sentiment-rationale sextuples."""
        stage3_result = self.execute_stage(prompt3)

        prompt4 = f"""Review the dialogue and each holder-target-aspect-opinion-sentiment-rationale sextuple: {stage3_result}. Please identify any instances where a sentiment flip occurs for the same holder regarding the specific target-aspect pair. 
    Determine the trigger type for these flips from predefined categories: introduction of new information, logical argumentation, participant feedback and interaction, personal experience and self-reflection.
    Formulate your output to include the holder, target, aspect, initial sentiment, flipped sentiment, and the trigger type, or state 'None' if no flips are identified."""
        stage4_result = self.execute_stage(prompt4)

        return stage1_result, stage2_result, stage3_result, stage4_result


cos_framework = CoSReasoningFramework("path_to_flan_t5_checkpoint")
dialogue = "xxxxxx"
multimodal_data = "xxxxxx"
results = cos_framework.sequential_reasoning(dialogue, multimodal_data)
