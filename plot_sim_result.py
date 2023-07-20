import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from bokeh.io import show
from bokeh.models import DataRange1d
from bokeh.layouts import gridplot
from bokeh.plotting import figure


def plot_config(width: int, height: int,title=None, x_label: str = None, y_label: str = None):
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

# # Simulation Data ====================================================================================================
file_name = 'MHE'
# file_name = 'MHE_nodelay'
# file_name = 'KF'
# file_name = 'KF_nodelay'

df_sim = pd.read_csv('./sim_result/' + file_name + '.csv')
df_sim_hw = pd.read_csv('./sim_result/' + file_name + '_halfway.csv')

df_sim.drop('Unnamed: 0', axis=1, inplace=True)
df_sim_hw.drop('Unnamed: 0', axis=1, inplace=True)

df_tf = df_sim[df_sim['sensor.type'] == 'tf']
df_tf_hw = df_sim_hw[df_sim_hw['sensor.type'] == 'tf']

t0 = df_sim['time'][0]


# # Plot ===============================================================================================================
cir_size = 2
p_total = []

# # Simulation
p0 = plot_config(width=600, height=600, title='Simulation', x_label='x [ m ]', y_label='y [ m ]')
p0.match_aspect=True
p0.aspect_scale=1
# start
p0.triangle(x=df_sim['x'].iloc[0], y=df_sim['y'].iloc[0], legend_label='start_sim', fill_color="crimson", color='black', size=15)
# end
p0.asterisk(x=df_sim['x'].iloc[-1], y=df_sim['y'].iloc[-1], legend_label='end_sim', fill_color="blue", color='crimson', size=15)
# curve
p0.circle(x=df_sim['x'], y=df_sim['y'], legend_label='Sim offline', fill_color="crimson", line_color="crimson", size=cir_size)
p0.circle(x=df_sim['rt_x'], y=df_sim['rt_y'], legend_label='Sim realtime', fill_color="royalblue", line_color="royalblue", size=cir_size)
p0.circle(x=df_sim["tf.x"], y=df_sim["tf.y"], legend_label='TF data', fill_color='black', line_color="black", size=cir_size)
p_total.append(p0)


# # Offline curve vs TF data
p1 = plot_config(width=600, height=600, title='Simulation in the Halfway', x_label='x [ m ]', y_label='y [ m ]')
p1.match_aspect=True
p1.aspect_scale=1
# start
p1.triangle(x=df_sim_hw['x'].iloc[0], y=df_sim_hw['y'].iloc[0], legend_label='start_sim', fill_color="crimson", color='black', size=15)
# end
p1.asterisk(x=df_sim_hw['x'].iloc[-1], y=df_sim_hw['y'].iloc[-1], legend_label='end_sim', fill_color="blue", color='crimson', size=15)
# curve
p1.circle(x=df_sim_hw['x'], y=df_sim_hw['y'], legend_label='Sim offline', fill_color="crimson", line_color="crimson", size=cir_size)
p1.circle(x=df_sim_hw['rt_x'], y=df_sim_hw['rt_y'], legend_label='Sim realtime', fill_color="royalblue", line_color="royalblue", size=cir_size)
p1.circle(x=df_sim_hw["tf.x"], y=df_sim_hw["tf.y"], legend_label='TF data', fill_color='black', line_color="black", size=cir_size)
p_total.append(p1)


# # Euclidean Error based on GPR of TF Data
ti = np.array(df_tf['time'] - t0).reshape(-1,1)
xi = np.array(df_tf['tf.x']).reshape(-1)
yi = np.array(df_tf['tf.y']).reshape(-1)

kernel = 10 * RBF(length_scale=0.4, length_scale_bounds='fixed')
gpr_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
gpr_x.fit(ti, xi)
gpr_y = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
gpr_y.fit(ti, yi)

condi1 = df_sim['time'] - t0 >= 5
condi2 = df_sim['time'] - t0 <= 30
df_window = df_sim[condi1*condi2]

t_gpr = df_window['time'].to_numpy().reshape(-1,1) - t0
x_gpr = gpr_x.predict(t_gpr)
y_gpr = gpr_y.predict(t_gpr)
t_gpr = t_gpr.reshape(-1)

df_win_rt = df_window[-df_window['rt_x'].isna()]

t_rt_gpr = df_win_rt['time'].to_numpy().reshape(-1,1) - t0
x_rt_gpr = gpr_x.predict(t_rt_gpr)
y_rt_gpr = gpr_y.predict(t_rt_gpr)
t_rt_gpr = t_rt_gpr.reshape(-1)

# Sum of Absolute Error (SAE) with GPR
dx_rt = df_win_rt['rt_x'].to_numpy() - x_rt_gpr
dy_rt = df_win_rt['rt_y'].to_numpy() - y_rt_gpr
dx = df_window['x'].to_numpy() - x_gpr
dy = df_window['y'].to_numpy() - y_gpr
mae_offline = np.sum(np.sqrt(dx ** 2 + dy ** 2)) / len(dx)
mae_rt = np.sum(np.sqrt(dx_rt ** 2 + dy_rt ** 2)) / len(dx_rt)

p2 = plot_config(width=600, height=600, title='Euclidean Error of Sim Real Time based on GPR', x_label='x [ m ]', y_label='y [ m ] ')
p2.match_aspect=True
p2.aspect_scale=1
p2.circle(x=df_sim['rt_x'], y=df_sim['rt_y'], legend_label=f'Sim Realtime, MAE= {mae_rt:.4f}', size=cir_size, fill_color='royalblue', line_color='royalblue')
p2.circle(x=x_rt_gpr, y=y_rt_gpr, legend_label='GPR of TF 5~30 s', size=cir_size, fill_color='green',line_color='green')
p2.circle(x=df_tf['tf.x'], y=df_tf['tf.y'], legend_label='TF data', size=cir_size, fill_color='black',line_color='black')
p_total.append(p2)

p3 = plot_config(width=600, height=600, title='Euclidean Error of Sim Offline based on GPR', x_label='x [ m ]', y_label='y [ m ] ')
p3.match_aspect=True
p3.aspect_scale=1
p3.circle(x=df_sim['x'], y=df_sim['y'], legend_label='Sim Offline, MAE= ' + "{:.4f}".format(mae_offline), size=cir_size, fill_color='crimson', line_color='crimson')
p3.circle(x=x_gpr, y=y_gpr, legend_label='GPR of TF 5~30 s', size=cir_size, fill_color='green',line_color='green')
p3.circle(x=df_tf['tf.x'], y=df_tf['tf.y'], legend_label='TF data', size=cir_size, fill_color='black',line_color='black')
p_total.append(p3)
# p3.circle(x=df_sim['time'] - t0, y=df_sim['x'], legend_label='sim offline, SAE= ' + "{:.4f}".format(sum_absolute_error), size=cir_size, fill_color='crimson', line_color='crimson')
# p3.circle(x=t_gpr, y=x_gpr, legend_label='GPR', size=cir_size, fill_color='green',line_color='green')
# p3.circle(x=df_tf['time'] - t0, y=df_tf['tf.x'], legend_label='tf data', size=cir_size, fill_color='black',line_color='black')
# p3.line(x=[t_gpr[0], t_gpr[0]], y=[min(x_gpr), max(x_gpr)], line_width=2, line_dash=[2,2], color='royalblue')
# p3.line(x=[t_gpr[-1], t_gpr[-1]], y=[min(x_gpr), max(x_gpr)]    , line_width=2, line_dash=[2,2], color='royalblue')


# # Simulation States
p01 = plot_config(width=600, height=400, title="state 'x'", x_label='t [ s ]', y_label='x [ m ] ')
p01.circle(x=df_sim['time'] - t0, y=df_sim['x'], legend_label='offline', size=cir_size, fill_color='crimson',line_color='crimson')
p01.circle(x=df_sim['time'] - t0, y=df_sim['rt_x'], legend_label='real time', size=cir_size, fill_color='royalblue',line_color='royalblue')
p01.circle(x=df_sim['time'] - t0, y=df_sim['tf.x'], legend_label='sensor data', size=cir_size, fill_color='black',line_color='black')
p_total.append(p01)
p02 = plot_config(width=600, height=400, title="state 'y'", x_label='t [ s ]', y_label='y [ m ] ')
p02.circle(x=df_sim['time'] - t0, y=df_sim['y'], legend_label='offline', size=cir_size, fill_color='crimson',line_color='crimson')
p02.circle(x=df_sim['time'] - t0, y=df_sim['rt_y'], legend_label='real time', size=cir_size, fill_color='royalblue',line_color='royalblue')
p02.circle(x=df_sim['time'] - t0, y=df_sim['tf.y'], legend_label='sensor data', size=cir_size, fill_color='black',line_color='black')
p_total.append(p02)
p03 = plot_config(width=600, height=400, title="state 'theta'", x_label='t [ s ]', y_label='theta [ rad ] ')
p03.circle(x=df_sim['time'] - t0, y=df_sim['theta'], legend_label='offline', size=cir_size, fill_color='crimson',line_color='crimson')
p03.circle(x=df_sim['time'] - t0, y=df_sim['rt_theta'], legend_label='real time', size=cir_size, fill_color='royalblue',line_color='royalblue')
p03.circle(x=df_sim['time'] - t0, y=df_sim['tf.yaw_z'], legend_label='sensor data', size=cir_size, fill_color='black',line_color='black')
p_total.append(p03)
p04 = plot_config(width=600, height=400, title="state 'v'", x_label='t [ s ]', y_label='v [ m/s ] ')
p04.circle(x=df_sim['time'] - t0, y=df_sim['velocity'], legend_label='sensor data', size=cir_size, fill_color='black',line_color='black')
p04.circle(x=df_sim['time'] - t0, y=df_sim['v'], legend_label='offline', size=cir_size, fill_color='crimson',line_color='crimson')
p04.circle(x=df_sim['time'] - t0, y=df_sim['rt_v'], legend_label='real time', size=cir_size, fill_color='royalblue',line_color='royalblue')
p_total.append(p04)

# # Computational Time
p4 = plot_config(width=600, height=400, title='computational time', x_label='t [ s ]', y_label='computational time [ ms ] ')
p4.circle(x=df_sim['time'] - t0, y=df_sim['t_com']*1000, legend_label='t_com', size=cir_size, fill_color='orange', line_color='orange')
# p4.y_range = DataRange1d(start=0, end=max(df_sim['t_com']*1000))
p_total.append(p4)

# # Velocity of Estimated Curve
dif_t = np.diff(df_window['time'])
dif_x = np.diff(df_window['x'])
dif_y = np.diff(df_window['y'])

dif_t_rt = np.diff(df_win_rt['time'])
dif_x_rt = np.diff(df_win_rt['x'])
dif_y_rt = np.diff(df_win_rt['y'])

v_ec = np.sqrt(dif_x ** 2 + dif_y ** 2) / dif_t
v_ec_rt = np.sqrt(dif_x_rt ** 2 + dif_y_rt ** 2) / dif_t_rt

stdv = np.std(v_ec)
stdv_rt = np.std(v_ec_rt)
maxv = np.max(v_ec)
maxv_rt = np.max(v_ec_rt)

p5 = plot_config(width=600, height=400, title='Velocity of Estimated Curve', x_label='t [ s ]', y_label='|v| [ m/s ] ')
# p5.circle(x=df_window['time'].iloc[:-1] - t0, y=v_ec, legend_label='Sim Offline', size=cir_size, fill_color='crimson', line_color='crimson')
# p5.circle(x=df_win_rt['time'].iloc[:-1] - t0, y=v_ec_rt, legend_label='Sim Realtime', size=cir_size, fill_color='royalblue', line_color='royalblue')

p5.line(x=df_window['time'].iloc[:-1] - t0, y=v_ec, legend_label=f'Sim Offline, std: {stdv:.3f}, max: {maxv:0.0f}', line_width=2, line_color='crimson')
p5.line(x=df_win_rt['time'].iloc[:-1] - t0, y=v_ec_rt, legend_label=f'Sim Realtime, std: {stdv_rt:.3f}, max: {maxv_rt:0.0f}', line_width=2, line_color='royalblue')
p_total.append(p5)

from bokeh.io import output_file
output_file(filename=f"./plot_result/" + file_name + ".html", title=file_name)

show(gridplot(p_total, ncols=2))











