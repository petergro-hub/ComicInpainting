# Inpainting and Retargeting Tool

This is a small inpainting and retargeting app I made based on recent papers using streamlit. The goal of this tool is to provide artists with a simple tool to edit their comics, but can also be used on any type of picture. No GPU is needed for this tool.

The inpainting tool is based on _LaMa inpainting with fourier convolutions_ <sup>[1](https://arxiv.org/abs/2109.07161), [2](https://github.com/saic-mdal/lama)</sup>, and the retargeting tool is based on _improved seam-carving_ <sup>[3](http://www.eng.tau.ac.il/~avidan/papers/vidret.pdf), [4](https://github.com/andrewdcampbell/seam-carving)</sup>


To install the requirements using conda use ```conda env create -f environment_app.yml```

To use it simply run ```streamlit run app.py```

