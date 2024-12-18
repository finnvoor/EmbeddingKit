from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import coremltools as ct
from utils import log


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("avsolatorio/GIST-all-MiniLM-L6-v2")

    def forward(self, input_ids, attention_mask):
        model_output = self.model(input_ids, attention_mask)
        token_embeddings = model_output[0]

        # Mean pooling
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

        # Normalize
        return F.normalize(embeddings, p=2, dim=1)


log("Loading model…")
tokenizer = AutoTokenizer.from_pretrained("avsolatorio/GIST-all-MiniLM-L6-v2")
model = Model()

log("Exporting…")
example_input = (
    torch.randint(0, tokenizer.vocab_size, (1, 512), dtype=torch.int32),  # input_ids
    torch.ones((1, 512), dtype=torch.int32),  # attention_mask
)
exported_model = torch.export.export(model, example_input)

log("Converting…")
mlmodel = ct.convert(
    exported_model,
    outputs=[ct.TensorType(name="embeddings")],
    minimum_deployment_target=ct.target.macOS15,
)

log("Saving…")
mlmodel.author = "avsolatorio"
mlmodel.license = "MIT"
mlmodel.short_description = "GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning\n\nhttps://huggingface.co/avsolatorio/GIST-all-MiniLM-L6-v2"
mlmodel.save("models/GIST-all-MiniLM-L6-v2.mlpackage")

log("Done!")
