import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from bokeh.models import DataRange1d
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show


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
# file_name = 'MHE'
# file_name = 'MHE_nodelay'
file_name = 'KF'
# file_name = 'KF_nodelay'

suffix = '_10_20s'

df_sim = pd.read_csv('./sim_result/' + file_name + '.csv')
columns = df_sim.columns.to_list()
t0 = df_sim['time'][0]

cond1 = df_sim['time'] - t0 >= 10
cond2 = df_sim['time'] - t0 <= 20
df_sim = df_sim[cond1*cond2]

gpr_t0 = 10
gpr_tn = 20

df_sim_hw = pd.read_csv('./sim_result/' + file_name + '_halfway.csv')

df_sim.drop('Unnamed: 0', axis=1, inplace=True)
df_sim_hw.drop('Unnamed: 0', axis=1, inplace=True)

df_tf = df_sim[df_sim['sensor.type'] == 'tf']
df_tf_hw = df_sim_hw[df_sim_hw['sensor.type'] == 'tf']




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


# # Simulation in the Halfway
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

kernel = 3 * RBF(length_scale=0.7, length_scale_bounds='fixed')
gpr_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
gpr_x.fit(ti, xi)
gpr_y = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
gpr_y.fit(ti, yi)

condi1 = df_sim['time'] - t0 >= gpr_t0
condi2 = df_sim['time'] - t0 <= gpr_tn
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

# Mean of Absolute Error (MAE) with GPR
dx_rt = df_win_rt['rt_x'].to_numpy() - x_rt_gpr
dy_rt = df_win_rt['rt_y'].to_numpy() - y_rt_gpr
dx = df_window['x'].to_numpy() - x_gpr
dy = df_window['y'].to_numpy() - y_gpr
norm2 = np.sqrt(dx ** 2 + dy ** 2)
norm2_rt = np.sqrt(dx_rt ** 2 + dy_rt ** 2)
mae_offline = np.average(norm2)
mae_rt = np.average(norm2_rt)

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
p3.circle(x=df_sim['x'], y=df_sim['y'], legend_label='Sim Offline, MAE= ' + f"{mae_offline:.4f}", size=cir_size, fill_color='crimson', line_color='crimson')
p3.circle(x=x_gpr, y=y_gpr, legend_label='GPR of TF 5~30 s', size=cir_size, fill_color='green',line_color='green')
p3.circle(x=df_tf['tf.x'], y=df_tf['tf.y'], legend_label='TF data', size=cir_size, fill_color='black',line_color='black')
p_total.append(p3)

p20 = plot_config(width=600, height=400, title="Euclidean Error of Sim based on GPR", x_label='t [ s ]', y_label='||SIM - GPR|| [ m ] ')
p20.circle(x=t_rt_gpr, y=norm2_rt, legend_label='Sim Realtime', size=cir_size, fill_color='royalblue', line_color='royalblue')
p20.circle(x=t_gpr, y=norm2, legend_label='Sim Offline', size=cir_size, fill_color='crimson', line_color='crimson')
p20.line(x=[t_rt_gpr[0], t_rt_gpr[-1]], y=[mae_rt]*2, legend_label=f'avg: {mae_rt:.3f}, std: {np.std(norm2_rt):.3f}', line_width=2, line_dash=[4,4], line_color='darkblue')
p20.line(x=[t_gpr[0], t_gpr[-1]], y=[mae_offline]*2, legend_label=f'avg: {mae_offline:.3f}, std: {np.std(norm2):.3f}', line_width=2, line_dash=[4,4], line_color='darkred')
p20.legend.ncols=2
p_total.append(p20)


# # Computational Time
p4 = plot_config(width=600, height=400, title='Computational Time', x_label='t [ s ]', y_label='computational time [ ms ] ')
p4.circle(x=df_sim['time'] - t0, y=df_sim['t_com']*1000, legend_label='t_com', size=cir_size, fill_color='orange', line_color='orange')
if 't_nlp' in columns:
    p4.circle(x=df_sim['time'] - t0, y=df_sim['t_nlp']*1000, legend_label='t_nlp', size=cir_size, fill_color='deeppink', line_color='deeppink')
# p4.y_range = DataRange1d(start=0, end=max(df_sim['t_com']*1000))
p_total.append(p4)

# # Simulation States
p01 = plot_config(width=600, height=400, title="State 'x'", x_label='t [ s ]', y_label='x [ m ] ')
p01.circle(x=df_sim['time'] - t0, y=df_sim['x'], legend_label='offline', size=cir_size, fill_color='crimson',line_color='crimson')
p01.circle(x=df_sim['time'] - t0, y=df_sim['rt_x'], legend_label='real time', size=cir_size, fill_color='royalblue',line_color='royalblue')
p01.circle(x=df_sim['time'] - t0, y=df_sim['tf.x'], legend_label='sensor data', size=cir_size, fill_color='black',line_color='black')
p_total.append(p01)
p02 = plot_config(width=600, height=400, title="State 'y'", x_label='t [ s ]', y_label='y [ m ] ')
p02.circle(x=df_sim['time'] - t0, y=df_sim['y'], legend_label='offline', size=cir_size, fill_color='crimson',line_color='crimson')
p02.circle(x=df_sim['time'] - t0, y=df_sim['rt_y'], legend_label='real time', size=cir_size, fill_color='royalblue',line_color='royalblue')
p02.circle(x=df_sim['time'] - t0, y=df_sim['tf.y'], legend_label='sensor data', size=cir_size, fill_color='black',line_color='black')
p_total.append(p02)
p03 = plot_config(width=600, height=400, title="State 'theta'", x_label='t [ s ]', y_label='theta [ rad ] ')
p03.circle(x=df_sim['time'] - t0, y=df_sim['theta'], legend_label='offline', size=cir_size, fill_color='crimson',line_color='crimson')
p03.circle(x=df_sim['time'] - t0, y=df_sim['rt_theta'], legend_label='real time', size=cir_size, fill_color='royalblue',line_color='royalblue')
p03.circle(x=df_sim['time'] - t0, y=df_sim['tf.yaw_z'], legend_label='sensor data', size=cir_size, fill_color='black',line_color='black')
p_total.append(p03)
p04 = plot_config(width=600, height=400, title="State 'v'", x_label='t [ s ]', y_label='v [ m/s ] ')
p04.circle(x=df_sim['time'] - t0, y=df_sim['velocity'], legend_label='sensor data', size=cir_size, fill_color='black',line_color='black')
p04.circle(x=df_sim['time'] - t0, y=df_sim['v'], legend_label='offline', size=cir_size, fill_color='crimson',line_color='crimson')
p04.circle(x=df_sim['time'] - t0, y=df_sim['rt_v'], legend_label='real time', size=cir_size, fill_color='royalblue',line_color='royalblue')
p_total.append(p04)


# # # Absolute Velocity Between Points (x,y)
# dif_t = np.diff(df_window['time'])
# dif_x = np.diff(df_window['x'])
# dif_y = np.diff(df_window['y'])
#
# dif_t_rt = np.diff(df_win_rt['time'])
# dif_x_rt = np.diff(df_win_rt['rt_x'])
# dif_y_rt = np.diff(df_win_rt['rt_y'])
#
# d_e = np.sqrt(dif_x ** 2 + dif_y ** 2)
# d_e_rt = np.sqrt(dif_x_rt ** 2 + dif_y_rt ** 2)
#
# v_e = d_e / dif_t
# v_e_rt = d_e_rt / dif_t_rt
#
# stdv = np.std(v_e)
# stdv_rt = np.std(v_e_rt)
# maxv = np.max(v_e)
# maxv_rt = np.max(v_e_rt)
#
# p5 = plot_config(width=600, height=400, title='Absolute Velocity Between Points (x,y)', x_label='t [ s ]', y_label='||v|| [ m/s ] ')
# # p5.circle(x=df_window['time'].iloc[:-1] - t0, y=v_e, legend_label='Sim Offline', size=cir_size, fill_color='crimson', line_color='crimson')
# # p5.circle(x=df_win_rt['time'].iloc[:-1] - t0, y=v_e_rt, legend_label='Sim Realtime', size=cir_size, fill_color='royalblue', line_color='royalblue')
#
# p5.line(x=df_window['time'].iloc[:-1] - t0, y=v_e, legend_label=f'Sim Offline, std: {stdv:.3f}, max: {maxv:0.0f}', line_width=2, line_color='crimson')
# p5.line(x=df_win_rt['time'].iloc[:-1] - t0, y=v_e_rt, legend_label=f'Sim Realtime, std: {stdv_rt:.3f}, max: {maxv_rt:0.0f}', line_width=2, line_color='royalblue')
# p_total.append(p5)


# # Euclidian Distance Between Points (x,y)
# bins = np.linspace(0, 0.3, 50)
# hist, edge = np.histogram(d_e, density=True, bins=bins)
# hist_rt, edge_rt = np.histogram(d_e_rt, density=True, bins=bins)
# p6 = plot_config(width=600, height=400, title='Euclidian Distance Between Points (x,y)', x_label='t [ s ]', y_label='distance [ m ] ')
# p6.quad(top=hist, bottom=0, left=edge[:-1], right=edge[1:], fill_color="crimson", line_color="white", legend_label=f'Sim Offline, std: {np.std(d_e):.3f}, max: {max(d_e):.3f}')
# p6.quad(top=hist_rt, bottom=0, left=edge_rt[:-1], right=edge_rt[1:], fill_color="royalblue", line_color="white", legend_label=f'Sim Realtime, std: {np.std(d_e):.3f}, max: {max(d_e):.3f}')
# p6.circle(x=df_window['time'].iloc[:-1] - t0, y=d_e, legend_label=f'Sim Offline, std: {np.std(d_e):.3f}, max: {max(d_e):.3f}', size=cir_size, fill_color='crimson', line_color='crimson')
# p6.circle(x=df_win_rt['time'].iloc[:-1] - t0, y=d_e_rt, legend_label=f'Sim Realtime, std: {np.std(d_e_rt):.3f}, max: {max(d_e_rt):.3f}', size=cir_size, fill_color='royalblue', line_color='royalblue')

# p6.line(x=df_win_rt['time'].iloc[:-1] - t0, y=d_e_rt, legend_label=f'Sim Realtime, std: {np.std(d_e_rt):.3f}, max: {max(d_e_rt):.3f}', line_width=2, line_color='royalblue')
# p6.line(x=df_window['time'].iloc[:-1] - t0, y=d_e, legend_label=f'Sim Offline, std: {np.std(d_e):.3f}, max: {max(d_e):.3f}', line_width=2, line_color='crimson')
#
# p_total.append(p6)

from bokeh.io import output_file, show
output_file(filename=f"./plot_result/" + file_name + suffix + ".html", title=file_name + suffix)

show(gridplot(p_total, ncols=2))











