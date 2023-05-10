from bokeh.io import output_notebook, show
from bokeh.plotting import figure
import matplotlib.pyplot as plt

def plot_config(width: int, height: int,title=None,
                x_label: str = None,
                y_label: str = None):
    p = figure(title=title, background_fill_color="#fafafa", width=width, height=height)
    p.output_backend = "svg"    # save plot in HTML as SVG
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.yaxis.axis_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "12pt"
    p.xaxis.axis_label_text_font_size = "12pt"
    return p
