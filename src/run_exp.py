import fire
from utils.utils import load_model, read_json
from utils.vision_utils import create_html
from config import IMG_TOKEN_ID
import torch
from PIL import Image


def run(model_id: str, im_path: str):
    model, processor, tokenizer = load_model(model_id)
    
    raw_image = Image.open(im_path)
    mapping = read_json("data/mapping.json")
    im_name = im_path.split('/')[-1]

    prompt = mapping[im_name]
    conversation = [
        {
    
          "role": "user",
          "content": [
              {"type": "text", "text": prompt},
              {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.forward(
            **inputs.to(model.device),
            output_hidden_states=True, 
            return_dict=True
            )

    token_labels = []
    image_token_counter = 0
    for token_id in inputs['input_ids'][0]:
        if token_id.item() == IMG_TOKEN_ID:
            # One indexed because the HTML logic wants it that way
            token_labels.append(f"<IMG{(image_token_counter+1):03d}>")
            image_token_counter += 1
        else:
            token_labels.append(tokenizer.decode([token_id]))

    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states)
    sequence_length = hidden_states[0].size(1)

    norm = model.language_model.model.norm
    lm_head = model.language_model.lm_head
    
    all_top_tokens = []
    for layer in range(num_layers):
        layer_hidden_states = hidden_states[layer]
        
        with torch.no_grad():
            logits = lm_head(norm(layer_hidden_states))
            probs = torch.softmax(logits, dim=-1)
            top_5_values, top_5_indices = torch.topk(probs, k=5, dim=-1)
        
        layer_top_tokens = []
        for pos in range(sequence_length):
            top_5_tokens = [tokenizer.decode(idx.item()) for idx in top_5_indices[0, pos]]
            top_5_probs = [f"{prob.item():.4f}" for prob in top_5_values[0, pos]]
            layer_top_tokens.append(list(zip(top_5_tokens, top_5_probs)))
        
        all_top_tokens.append(layer_top_tokens)

    create_html(raw_image, im_name, model_id, all_top_tokens, token_labels, prompt)
    


if __name__ == "__main__":
    fire.Fire(run)