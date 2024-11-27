import torch

# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("recursionpharma/OpenPhenom", trust_remote_code=True)

model.eval()

def test_model_predict(huggingface_model, C, return_channelwise_embeddings):
    example_input_array = torch.randint(
        low=0,
        high=255,
        size=(2, C, 256, 256),
        dtype=torch.uint8,
        device=huggingface_model.device,
    )
    huggingface_model.return_channelwise_embeddings = return_channelwise_embeddings
    embeddings = huggingface_model.predict(example_input_array)
    expected_output_dim = 384 * C if return_channelwise_embeddings else 384
    assert embeddings.shape == (2, expected_output_dim)
    print("Success!")
    print(embeddings.shape)

test_model_predict(model, 3, False)