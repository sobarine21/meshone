import streamlit as st
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from stl import mesh  # Importing mesh handling for STL output

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI Setup
st.title("Ever AI: Text-to-CAD Generator")
st.write("Use generative AI to create CAD models (STL/OBJ) from text prompts.")

# Input field for entering prompt
prompt = st.text_input("Enter your prompt:", "Design a simple cube with dimensions 10x10x10 cm.")

# Button to generate response
if st.button("Generate CAD Model"):
    try:
        # Step 1: Use Google Generative AI to interpret the text prompt for model generation
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        st.write("Response from AI:")
        st.write(response.text)

        # Step 2: Generate mesh (CAD model) based on the AI response (this part needs to be custom tailored)
        # Assuming the model provides textual data describing the mesh (e.g., cube size)
        cad_model = generate_mesh_from_text(response.text)  # This is your custom function to generate the mesh

        # Step 3: Provide download link for the generated CAD file (STL format as an example)
        cad_file_path = "generated_model.stl"
        cad_model.save(cad_file_path)  # Save the mesh to an STL file

        # Display a download link to the user
        st.download_button(label="Download STL File", data=cad_file_path, file_name="model.stl", mime="application/stl")
        
    except Exception as e:
        st.error(f"Error: {e}")


def generate_mesh_from_text(description: str):
    """
    A custom function to generate a mesh based on a textual description.
    This function should parse the description and create corresponding 3D mesh data.
    For simplicity, we'll assume a cube or a simple shape in this example.
    """

    # Example: Parse the description for a simple cube
    if "cube" in description:
        dimensions = extract_dimensions_from_text(description)
        if dimensions:
            return create_cube_mesh(dimensions)
    else:
        raise ValueError("Could not interpret the description for mesh generation.")
    
    # If we can’t interpret, we return a default cube
    return create_cube_mesh([10, 10, 10])  # Default to 10x10x10 cm cube


def extract_dimensions_from_text(description: str):
    """
    A helper function to extract dimensions from a description (e.g., "10x10x10 cm").
    This can be expanded for more complex shapes.
    """
    try:
        # Extract numbers (dimensions) from the description, e.g., "10x10x10"
        parts = [int(i) for i in description.split() if i.isdigit()]
        if len(parts) == 3:
            return parts
        else:
            return None
    except:
        return None


def create_cube_mesh(dimensions: list):
    """
    A function to create a simple cube mesh. This is just for demonstration.
    It uses the `numpy-stl` library to generate the mesh and save as an STL file.
    """
    # Cube dimensions: [width, height, depth]
    width, height, depth = dimensions

    # Generate 8 vertices of the cube
    vertices = np.array([
        [-width / 2, -height / 2, -depth / 2],
        [ width / 2, -height / 2, -depth / 2],
        [ width / 2,  height / 2, -depth / 2],
        [-width / 2,  height / 2, -depth / 2],
        [-width / 2, -height / 2,  depth / 2],
        [ width / 2, -height / 2,  depth / 2],
        [ width / 2,  height / 2,  depth / 2],
        [-width / 2,  height / 2,  depth / 2],
    ])

    # Define the 12 triangles composing the cube faces
    faces = np.array([
        [0, 3, 1], [1, 3, 2],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 4], [1, 5, 4],  # Front face
        [2, 3, 7], [2, 7, 6],  # Back face
        [0, 4, 3], [3, 4, 7],  # Left face
        [1, 2, 5], [5, 2, 6],  # Right face
    ])

    # Create the mesh
    cube_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            cube_mesh.vectors[i][j] = vertices[face[j]]

    return cube_mesh
