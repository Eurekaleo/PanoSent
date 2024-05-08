import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from transformers import T5ForConditionalGeneration, T5Tokenizer


class MultimodalDialogueProcessor:
    def __init__(self, imagebind_ckpt_path, flan_t5_ckpt_path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.imagebind_model = imagebind_model.imagebind_huge(
            pretrained=True, store_path=imagebind_ckpt_path
        )
        self.imagebind_model.eval()
        self.imagebind_model.to(self.device)

        self.llm_model = T5ForConditionalGeneration.from_pretrained(flan_t5_ckpt_path)
        self.llm_tokenizer = T5Tokenizer.from_pretrained(flan_t5_ckpt_path)
        self.llm_model.eval()
        self.llm_model.to(self.device)

        self.projection_layer = torch.nn.Linear(1024, self.llm_model.config.d_model)
        self.projection_layer.to(self.device)

    def process_modalities(self, text_list, image_paths, audio_paths):
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(text_list, self.device),
            ModalityType.VISION: data.load_and_transform_vision_data(
                image_paths, self.device
            ),
            ModalityType.AUDIO: data.load_and_transform_audio_data(
                audio_paths, self.device
            ),
        }
        return inputs

    def compute_embeddings(self, inputs):
        with torch.no_grad():
            embeddings = self.imagebind_model(inputs)
            text_inputs = self.llm_tokenizer(
                text_list, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            text_outputs = self.llm_model(**text_inputs, return_dict=True)
            text_embeddings = text_outputs.last_hidden_state

            projected_vision_embeddings = self.projection_layer(
                embeddings[ModalityType.VISION]
            )
            projected_audio_embeddings = self.projection_layer(
                embeddings[ModalityType.AUDIO]
            )

        combined_embeddings = torch.cat(
            (projected_vision_embeddings, projected_audio_embeddings, text_embeddings),
            dim=0,
        )
        normalized_embeddings = torch.nn.functional.normalize(
            combined_embeddings, p=2, dim=1
        )
        return normalized_embeddings
