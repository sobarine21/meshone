import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import google.generativeai as genai
from threading import Thread
import trimesh
import numpy as np
import tempfile
import os

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Set an environment variable for Hugging Face token
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

# Load the LLaMA-Mesh model and tokenizer
model_path = "Zhengyi/LLaMA-Mesh"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", low_cpu_mem_usage=True)
terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

def generate_mesh(prompt, temperature=0.9, max_new_tokens=4096):
    conversation = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=terminators,
    )

    if temperature == 0:
        generate_kwargs['do_sample'] = False

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
    return "".join(outputs)

def apply_gradient_color(mesh_text):
    temp_file = tempfile.NamedTemporaryFile(suffix="", delete=False).name
    with open(temp_file + ".obj", "w") as f:
        f.write(mesh_text)
    mesh = trimesh.load_mesh(temp_file + ".obj", file_type='obj')

    vertices = mesh.vertices
    y_values = vertices[:, 1]

    y_normalized = (y_values - y_values.min()) / (y_values.max() - y_values.min())

    colors = np.zeros((len(vertices), 4))
    colors[:, 0] = y_normalized
    colors[:, 2] = 1 - y_normalized
    colors[:, 3] = 1.0

    mesh.visual.vertex_colors = colors

    glb_path = temp_file + ".glb"
    with open(glb_path, "wb") as f:
        f.write(trimesh.exchange.gltf.export_glb(mesh))
    return glb_path

# Streamlit App UI
st.title("Ever AI - 3D CAD Model Generator")
st.write("Use generative AI to create 3D CAD models based on your prompt.")

prompt = st.text_input("Enter your prompt:", "Create a 3D model of a house.")

if st.button("Generate CAD Model"):
    try:
        response = generate_mesh(prompt)
        
        cad_file_path = "generated_model.obj"
        with open(cad_file_path, "w") as f:
            f.write(response)
        
        st.write("CAD Model Generated:")
        st.code(response, language='plaintext')

        glb_path = apply_gradient_color(response)
        with open(glb_path, "rb") as f:
            btn = st.download_button(
                label="Download GLB File",
                data=f,
                file_name="generated_model.glb",
                mime="application/octet-stream"
            )
    except Exception as e:
        st.error(f"Error: {e}")
