import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch
import os

from huggingface_hub import login
login(token=os.environ["HF_TOKEN"]) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
llm_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_caption(image):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    outputs = clip_model.get_image_features(**inputs)
    return outputs[0].detach().cpu()

def recommend_styles(image, user_request, body_info=""):
    caption = generate_caption(image)
    prompt = f"""
You are a fashion stylist AI.
Given this description: "{caption}"
{f"User body info: {body_info}" if body_info else ""}
And this user request: "{user_request}"
Generate a list of 5 suitable fashion styles (short phrase each).
List:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return f"Caption: {caption}\n\n{response}"

# Gradio-Huggingface Deploy
with gr.Blocks() as demo:
    gr.Markdown("## AutoStylish - AI Fashion Stylist ")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an image")
        text_input = gr.Textbox(label="What event or style do you want?", placeholder="E.g. wedding dinner")
    
    body_input = gr.Textbox(label="(Optional) Body Info", placeholder="E.g. height: 175cm, chest: 95cm, etc.")
    output = gr.Textbox(label="Recommended Styles")

    run_button = gr.Button("Get Style Suggestions")

    run_button.click(fn=recommend_styles, 
                     inputs=[image_input, text_input, body_input],
                     outputs=output)

demo.launch()
