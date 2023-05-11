from get_data import *
from model import *
from bokeh.io import show
from bokeh.layouts import gridplot
from bokeh.plotting import figure


def plot_config(width: int, height: int,title=None, x_label: str = None,
                                                    y_label: str = None):
    p = figure(title=title, background_fill_color="#fafafa", width=width, height=height)
    p.output_backend = "svg"    # save plot in HTML as SVG
    p.title.text_font_size = "15pt"
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.yaxis.axis_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "12pt"
    p.xaxis.axis_label_text_font_size = "12pt"
    return p


# ==========================================Simulation================================================
a_z = imu['a_z'] - 0.33    # compensate the data offset of linear acceleration
w_y = imu['w_y']
U_k = np.append(a_z,w_y,axis=0) # 'input' of simulation system

timestamp = imu['time'] - imu['time'][0][0] # time start from 0s
Ts = np.diff(timestamp)  # time steps of time stamps, number of data will be -1


x0 = [0,0,0,0]  # initial states
x_all = sim(x0, U_k, timestamp)
print(x_all.shape)


# ============================================ Plot =============================================
p_total = []

p = plot_config(width=600, height=600, title='imu curve',  x_label='x [ unknown ]', y_label= 'y [ unknown ]')
p.circle(x_all[0][0], x_all[1][0], fill_color="red",legend_label='start', size=10)
p.line(x=x_all[0], y=x_all[1],legend_label='curve', line_width=1, line_color='black')
p.asterisk(x_all[0][-1], x_all[1][-1], line_color="green",legend_label='end', size=10)
p_total.append(p)


p1 = plot_config(width=600, height=600, title='tf translation curve', x_label='x [ unknown ]', y_label='y [ unknown ]')
p1.circle(tf['x'][0][0], tf['y'][0][0], fill_color="red",legend_label='start', size=10)
p1.line(x=tf['x'][0], y=tf['y'][0],legend_label='curve', line_width=1, line_color='black')
p1.asterisk(tf['x'][0][-1], tf['y'][0][-1], line_color="green",legend_label='end', size=10)
p_total.append(p1)


p2 = plot_config(width=900, height=600, title='system states',  x_label='t [ s ]', y_label= 'states [ unknown ] ')
p2.line(x=timestamp[0], y=x_all[0],legend_label='x', line_width=2, line_color='green')
p2.line(x=timestamp[0], y=x_all[1],legend_label='y', line_width=2, line_color='black')
p2.line(x=timestamp[0], y=x_all[2],legend_label='theta', line_width=2, line_color='royalblue')
p2.line(x=timestamp[0], y=x_all[3],legend_label='v', line_width=2, line_color='crimson')
p_total.append(p2)

show(gridplot(p_total, ncols=2))

























