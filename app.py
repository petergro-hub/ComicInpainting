#!/usr/bin/env python3
import base64
import json
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path
import cv2   

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path

#TODO add iterative inpainting

import argparse
import io
import multiprocessing
from typing import Union

import torch

try:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
except:
    pass

from helper import (
    download_model,
    load_img,
    norm_img,
    numpy_to_bytes,
    pad_img_to_modulo,
    resize_max_size,
)

NUM_THREADS = str(multiprocessing.cpu_count())

os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
if os.environ.get("CACHE_DIR"):
    os.environ["TORCH_HOME"] = os.environ["CACHE_DIR"]

#BUILD_DIR = os.environ.get("LAMA_CLEANER_BUILD_DIR", "./lama_cleaner/app/build")

model = None
device = "cpu"


def run(image, mask):
    """
    image: [C, H, W]
    mask: [1, H, W]
    return: BGR IMAGE
    """
    origin_height, origin_width = image.shape[1:]
    image = pad_img_to_modulo(image, mod=8)
    mask = pad_img_to_modulo(mask, mod=8)

    mask = (mask > 0) * 1
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        inpainted_image = model(image, mask)

    print(f"process time: {(time.time() - start)*1000}ms")
    cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
    cur_res = cur_res[0:origin_height, 0:origin_width, :]
    cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
    cur_res = cv2.cvtColor(cur_res, cv2.COLOR_BGR2RGB)
    return cur_res


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


#mask = 255-image[:,:,3]
#rgbImage = cv2.cvtColor(npimage, cv2.COLOR_RGBA2RGB)
#np_img = cv2.cvtColor(np_img, cv2.COLOR_BGRA2RGB)
#np_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

def process(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    original_shape = image.shape
    interpolation = cv2.INTER_CUBIC

    #size_limit: Union[int, str] = request.form.get("sizeLimit", "1080")
    #if size_limit == "Original":
    size_limit = max(image.shape)
    #else:
    #    size_limit = int(size_limit)

    print(f"Origin image shape: {original_shape}")
    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)
    print(f"Resized image shape: {image.shape}")
    image = norm_img(image)

    mask = 255-mask[:,:,3]
    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)
    mask = norm_img(mask)

    res_np_img = run(image, mask)

    # resize to original size
    # res_np_img = cv2.resize(
    #     res_np_img,
    #     dsize=(original_shape[1], original_shape[0]),
    #     interpolation=interpolation,
    # )

    return cv2.cvtColor(res_np_img, cv2.COLOR_BGR2RGB)
    #send_file(
    #    io.BytesIO(numpy_to_bytes(res_np_img)),
    #    mimetype="image/jpeg",
    #    as_attachment=True,
    #    attachment_filename="result.jpeg",
    #)




def main():
    global model
    global device
    #args = get_args_parser()
    device = torch.device("cpu")#args.device)

    if os.environ.get("LAMA_MODEL"):
        model_path = os.environ.get("LAMA_MODEL")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"lama torchscript model not found: {model_path}")
    else:
        model_path = download_model()

    model = torch.jit.load(model_path, map_location="cpu")
    model = model.to(device)
    model.eval()


    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        #"About": about,
        #"Basic example": full_app,
        #"Get center coords of circles": center_circle_app,
        #"Color-based image annotation": color_annotation_app,
        #"Download Base64 encoded PNG": png_export,
        "Image Inpainting": image_inpainting,
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page]()

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://www.linkedin.com/in/petergro/">Peter Grönquist</a></h6>',
            unsafe_allow_html=True,
        )







# def about():
#     st.markdown(
#         """
#     Welcome to the demo of [Streamlit Drawable Canvas](https://github.com/andfanilo/streamlit-drawable-canvas).
    
#     On this site, you will find a full use case for this Streamlit component, and answers to some frequently asked questions.
    
#     :pencil: [Demo source code](https://github.com/andfanilo/streamlit-drawable-canvas-demo/)    
#     """
#     )
#     st.image("img/demo.gif")
#     st.markdown(
#         """
#     What you can do with Drawable Canvas:

#     * Draw freely, lines, circles and boxes on the canvas, with options on stroke & fill
#     * Rotate, skew, scale, move any object of the canvas on demand
#     * Select a background color or image to draw on
#     * Get image data and every drawn object properties back to Streamlit !
#     * Choose to fetch back data in realtime or on demand with a button
#     * Undo, Redo or Drop canvas
#     * Save canvas data as JSON to reuse for another session
#     """
#     )


# def full_app():
#     st.sidebar.header("Configuration")
#     st.markdown(
#         """
#     Draw on the canvas, get the drawings back to Streamlit!
#     * Configure canvas in the sidebar
#     * In transform mode, double-click an object to remove it
#     * In polygon mode, left-click to add a point, right-click to close the polygon, double-click to remove the latest point
#     """
#     )

#     with st.echo("below"):
#         # Specify canvas parameters in application
#         stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
#         stroke_color = st.sidebar.color_picker("Stroke color hex: ")
#         bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
#         bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
#         drawing_mode = st.sidebar.selectbox(
#             "Drawing tool:",
#             ("freedraw", "line", "rect", "circle", "transform", "polygon"),
#         )
#         realtime_update = st.sidebar.checkbox("Update in realtime", True)

#         # Create a canvas component
#         canvas_result = st_canvas(
#             fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
#             stroke_width=stroke_width,
#             stroke_color=stroke_color,
#             background_color=bg_color,
#             background_image=Image.open(bg_image) if bg_image else None,
#             update_streamlit=realtime_update,
#             height=150,
#             drawing_mode=drawing_mode,
#             display_toolbar=st.sidebar.checkbox("Display toolbar", True),
#             key="full_app",
#         )

#         # Do something interesting with the image data and paths
#         if canvas_result.image_data is not None:
#             st.image(canvas_result.image_data)
#         if canvas_result.json_data is not None:
#             objects = pd.json_normalize(canvas_result.json_data["objects"])
#             for col in objects.select_dtypes(include=["object"]).columns:
#                 objects[col] = objects[col].astype("str")
#             st.dataframe(objects)


# def center_circle_app():
#     st.markdown(
#         """
#     Computation of center coordinates for circle drawings some understanding of Fabric.js coordinate system
#     and play with some trigonometry.

#     Coordinates are canvas-related to top-left of image, increasing x going down and y going right.

#     ```
#     center_x = left + radius * cos(angle * pi / 180)
#     center_y = top + radius * sin(angle * pi / 180)
#     ```
#     """
#     )
#     bg_image = Image.open("img/tennis-balls.jpg")

#     with open("saved_state.json", "r") as f:
#         saved_state = json.load(f)

#     canvas_result = st_canvas(
#         fill_color="rgba(255, 165, 0, 0.2)",  # Fixed fill color with some opacity
#         stroke_width=5,
#         stroke_color="black",
#         background_image=bg_image,
#         initial_drawing=saved_state
#         if st.sidebar.checkbox("Initialize with saved state", False)
#         else None,
#         height=400,
#         width=600,
#         drawing_mode="circle",
#         key="center_circle_app",
#     )
#     with st.echo("below"):
#         if canvas_result.json_data is not None:
#             df = pd.json_normalize(canvas_result.json_data["objects"])
#             if len(df) == 0:
#                 return
#             df["center_x"] = df["left"] + df["radius"] * np.cos(
#                 df["angle"] * np.pi / 180
#             )
#             df["center_y"] = df["top"] + df["radius"] * np.sin(
#                 df["angle"] * np.pi / 180
#             )

#             st.subheader("List of circle drawings")
#             for _, row in df.iterrows():
#                 st.markdown(
#                     f'Center coords: ({row["center_x"]:.2f}, {row["center_y"]:.2f}). Radius: {row["radius"]:.2f}'
#                 )


# def color_annotation_app():
#     st.markdown(
#         """
#     Drawable Canvas doesn't provided out-of-the-box image annotation capabilities, but we can hack something with session state,
#     by mapping a drawing fill color to a label.

#     Annotate pedestrians, cars and traffic lights with this one, with any color/label you want 
#     (though in a real app you should rather provide your own label and fills :smile:).

#     If you really want advanced image annotation capabilities, you'd better check [Streamlit Label Studio](https://discuss.streamlit.io/t/new-component-streamlit-labelstudio-allows-you-to-embed-the-label-studio-annotation-frontend-into-your-application/9524)
#     """
#     )
#     with st.echo("below"):
#         bg_image = Image.open("img/annotation.jpeg")
#         label_color = (
#             st.sidebar.color_picker("Annotation color: ", "#EA1010") + "77"
#         )  # for alpha from 00 to FF
#         label = st.sidebar.text_input("Label", "Default")
#         mode = "transform" if st.sidebar.checkbox("Move ROIs", False) else "rect"

#         canvas_result = st_canvas(
#             fill_color=label_color,
#             stroke_width=3,
#             background_image=bg_image,
#             height=320,
#             width=512,
#             drawing_mode=mode,
#             key="color_annotation_app",
#         )
#         if canvas_result.json_data is not None:
#             df = pd.json_normalize(canvas_result.json_data["objects"])
#             if len(df) == 0:
#                 return
#             st.session_state["color_to_label"][label_color] = label
#             df["label"] = df["fill"].map(st.session_state["color_to_label"])
#             st.dataframe(df[["top", "left", "width", "height", "fill", "label"]])

#         with st.expander("Color to label mapping"):
#             st.json(st.session_state["color_to_label"])


# def png_export():
#     st.markdown(
#         """
#     Realtime update is disabled for this demo. 
#     Press the 'Download' button at the bottom of canvas to update exported image.
#     """
#     )
#     try:
#         Path("tmp/").mkdir()
#     except FileExistsError:
#         pass

#     # Regular deletion of tmp files
#     # Hopefully callback makes this better
#     now = time.time()
#     N_HOURS_BEFORE_DELETION = 1
#     for f in Path("tmp/").glob("*.png"):
#         st.write(f, os.stat(f).st_mtime, now)
#         if os.stat(f).st_mtime < now - N_HOURS_BEFORE_DELETION * 3600:
#             Path.unlink(f)

#     if st.session_state["button_id"] == "":
#         st.session_state["button_id"] = re.sub(
#             "\d+", "", str(uuid.uuid4()).replace("-", "")
#         )

#     button_id = st.session_state["button_id"]
#     file_path = f"tmp/{button_id}.png"

#     custom_css = f""" 
#         <style>
#             #{button_id} {{
#                 display: inline-flex;
#                 align-items: center;
#                 justify-content: center;
#                 background-color: rgb(255, 255, 255);
#                 color: rgb(38, 39, 48);
#                 padding: .25rem .75rem;
#                 position: relative;
#                 text-decoration: none;
#                 border-radius: 4px;
#                 border-width: 1px;
#                 border-style: solid;
#                 border-color: rgb(230, 234, 241);
#                 border-image: initial;
#             }} 
#             #{button_id}:hover {{
#                 border-color: rgb(246, 51, 102);
#                 color: rgb(246, 51, 102);
#             }}
#             #{button_id}:active {{
#                 box-shadow: none;
#                 background-color: rgb(246, 51, 102);
#                 color: white;
#                 }}
#         </style> """

#     data = st_canvas(update_streamlit=False, key="png_export")
#     if data is not None and data.image_data is not None:
#         img_data = data.image_data
#         im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
#         im.save(file_path, "PNG")

#         buffered = BytesIO()
#         im.save(buffered, format="PNG")
#         img_data = buffered.getvalue()
#         try:
#             # some strings <-> bytes conversions necessary here
#             b64 = base64.b64encode(img_data.encode()).decode()
#         except AttributeError:
#             b64 = base64.b64encode(img_data).decode()

#         dl_link = (
#             custom_css
#             + f'<a download="{file_path}" id="{button_id}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
#         )
#         st.markdown(dl_link, unsafe_allow_html=True)


def image_inpainting():
    st.title("Image Inpainting Tool")
    st.markdown(
        """
    Draw over the parts of the image you want removed and this tool will try to remove it using a DeepLearning model. 
    Should it not work perfectly the first time, you can keep redrawing until it fits your needs.
    """
    )
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:

        #st.write(bytes_data)
        #string_data = stringio.read()
        #st.write(string_data)
        bytes_data = uploaded_file.getvalue()
        up_image = Image.open(BytesIO(bytes_data)).convert("RGBA")
        width, height = up_image.size
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 100, 20)
        show_mask = st.sidebar.checkbox('Show mask')

        canvas_result = st_canvas(
            stroke_color="rgba(255, 0, 255, 0.8)",
            stroke_width=stroke_width,
            background_image=up_image,
            height=height,#320,
            width=width,#512,
            drawing_mode="freedraw",
            key="compute_arc_length",
        )
        if canvas_result.image_data is not None:
            
            im = canvas_result.image_data
            background = np.where(
                (im[:, :, 0] == 0) & 
                (im[:, :, 1] == 0) & 
                (im[:, :, 2] == 0)
            )
            drawing = np.where(
                (im[:, :, 0] == 255) & 
                (im[:, :, 1] == 0) & 
                (im[:, :, 2] == 255)
            )
            im[background]=[0,0,0,255]
            im[drawing]=[0,0,0,0] #RGBA
            if show_mask:
                st.write("Image mask:")
                st.image(im)
            if st.button('Run'):
                st.write("Image with inpainting:")
                inpainted_img = process(np.array(up_image), np.array(im)) #TODO Put button here
                st.image(inpainted_img)
        #if (
        #    canvas_result.json_data is not None
        #    and len(canvas_result.json_data["objects"]) != 0
        #):
        #    df = pd.json_normalize(canvas_result.json_data["objects"])
        #    paths = df["path"].tolist()
        #    for ind, path in enumerate(paths):
        #        path = parse_path(" ".join([str(e) for line in path for e in line]))
        #        st.write(f"Path {ind} has length {path.length():.3f} pixels")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Comic ML Tools", page_icon=":pencil2:"
    )
    #st.title("Drawable Canvas Demo")
    st.sidebar.subheader("Configuration")
    main()


















#@app.route("/")
#def index():
#    return send_file(os.path.join(BUILD_DIR, "index.html"))






# def main():
#     global model
#     global device
#     args = get_args_parser()
#     device = torch.device(args.device)

#     if os.environ.get("LAMA_MODEL"):
#         model_path = os.environ.get("LAMA_MODEL")
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"lama torchscript model not found: {model_path}")
#     else:
#         model_path = download_model()

#     model = torch.jit.load(model_path, map_location="cpu")
#     model = model.to(device)
#     model.eval()
#     app.run(host="0.0.0.0", port=args.port, debug=args.debug)
