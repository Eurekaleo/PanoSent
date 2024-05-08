import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer


class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-xxl"
        )
        self.image_model = ImageBindModel(pretrained=True)
        self.fc = nn.Linear(
            self.text_model.config.d_model + self.image_model.output_features, 512
        )
        self.output_layer = nn.Linear(512, 256)

    def forward(self, input_ids, attention_mask, image_tensors):
        text_output = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        image_features = self.image_model(image_tensors)
        combined_features = torch.cat((text_output, image_features), dim=1)
        x = self.fc(combined_features)
        output = self.output_layer(x)
        return output


class MultimodalDataset(Dataset):
    def __init__(self, dataset_name):
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.data = self.load_dataset(dataset_name)

    def load_dataset(self, dataset_name):
        return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        image = item["image"]
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).input_ids.squeeze()
        attention_mask = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).attention_mask.squeeze()
        image_tensor = self.transform(image)
        return input_ids, attention_mask, image_tensor


multimodal_loader = DataLoader(MultimodalDataset("xxxx"), batch_size=32, shuffle=True)
multimodal_model = MultimodalModel()
nll_loss = nn.NLLLoss()
optimizer_multimodal = optim.Adam(multimodal_model.parameters(), lr=0.001)


def train_multimodal_model(model, data_loader, optimizer, loss_function):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, image_tensor in data_loader:
        optimizer.zero_grad()
        output = model(input_ids, attention_mask, image_tensor)
        loss = loss_function(output, data["target"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


total_loss_multimodal = train_multimodal_model(
    multimodal_model, multimodal_loader, optimizer_multimodal, nll_loss
)
