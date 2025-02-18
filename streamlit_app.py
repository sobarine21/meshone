import streamlit as st
import google.generativeai as genai
import trimesh

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Define the function to generate mesh from text description
def generate_mesh_from_text(description: str):
    """
    This function generates a 3D mesh based on any description.
    It doesn't assume anything about the object, and no fallback shapes are created.
    """
    shape_info = parse_shape_info_from_text(description)
    
    if shape_info:
        return create_dynamic_shape_mesh(shape_info)
    else:
        return None  # No fallback, return None if description can't be parsed correctly.

def parse_shape_info_from_text(description: str):
    """
    Parses the description dynamically based on natural language input.
    This function doesn't assume specific shapes like "toy car" or "cube".
    It extracts attributes like size, shape, and components based on the description.
    """
    description = description.lower()

    # Extracting size and shape based on keywords, this is highly flexible
    shape_info = {}

    words = description.split()

    # Check if the description includes any geometric indicators (radius, size, etc.)
    shape_info["dimensions"] = []
    for word in words:
        if word.isdigit():
            shape_info["dimensions"].append(int(word))
        elif word in ['radius', 'height', 'width', 'length']:
            shape_info["dimensions"].append(word)

    # If no dimensions or shapes are found, return None (i.e., no mesh)
    if not shape_info["dimensions"]:
        return None

    return shape_info

def create_dynamic_shape_mesh(shape_info):
    """
    This function creates meshes based on the description from the AI.
    There are no assumptions or fallbacks for undefined shapes.
    """
    # Get the dimensions, and if no valid dimensions are given, return None
    dimensions = shape_info["dimensions"]

    # If no meaningful dimensions, return None (no fallback box)
    if "unspecified" in dimensions:
        return None  # Do not create any fallback shape if dimensions are unclear.

    # Dynamically interpret the dimensions and create objects only if possible
    if len(dimensions) == 1:
        # If there's just one dimension, assume it's a sphere (radius)
        return trimesh.creation.icosphere(radius=dimensions[0])

    elif len(dimensions) == 3:
        # If three dimensions are given, create a box
        return trimesh.creation.box(extents=(dimensions[0], dimensions[1], dimensions[2]))

    elif "radius" in dimensions and "height" in dimensions:
        # Create a cylinder if "radius" and "height" are mentioned
        return trimesh.creation.cylinder(radius=dimensions[0], height=dimensions[1])

    else:
        # If no recognizable pattern, return None (no default shapes)
        return None


# Streamlit UI Setup
st.title("Ever AI: Text-to-CAD Generator with Trimesh")
st.write("Use generative AI to create CAD models (STL/OBJ) from text prompts.")

# Input field for entering prompt
prompt = st.text_input("Enter your prompt:", "Create a toy car.")

# Button to generate response
if st.button("Generate CAD Model"):
    try:
        # Step 1: Use Google Generative AI to interpret the text prompt for model generation
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        st.write("Response from AI:")
        st.write(response.text)

        # Step 2: Generate mesh (CAD model) dynamically based on the AI response
        cad_model = generate_mesh_from_text(response.text)  # Dynamic mesh generation using Trimesh

        if cad_model:
            # Step 3: Provide download link for the generated CAD file (STL format as an example)
            cad_file_path = "generated_model.stl"
            cad_model.export(cad_file_path)  # Save the mesh to an STL file

            # Display a download link to the user
            st.download_button(label="Download STL File", data=cad_file_path, file_name="model.stl", mime="application/stl")
        else:
            st.error("The description could not be parsed to create a valid 3D model.")

    except Exception as e:
        st.error(f"Error: {e}")
