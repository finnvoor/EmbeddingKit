from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import coremltools as ct
from utils import log
import argparse

parser = argparse.ArgumentParser(
    prog="AllMiniLML6v2", description="Convert All-MiniLM-L6-v2 to CoreML"
)
parser.add_argument("--palettize", action="store_true", help="Palettize weights")
args = parser.parse_args()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

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
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
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

if args.palettize:
    log("Palettizing…")
    op_config = ct.optimize.coreml.OpPalettizerConfig(nbits=4, weight_threshold=512)
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config)

log("Saving…")
mlmodel.author = "sentence-transformers"
mlmodel.license = "Apache License 2.0"
mlmodel.short_description = "This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.\n\nhttps://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
mlmodel.save("models/all-MiniLM-L6-v2.mlpackage")

log("Done!")
