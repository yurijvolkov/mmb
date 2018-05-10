import pandas as pd

from bokeh.plotting import figure, show, save


results = pd.read_csv('report.csv', index_col=0)

f = figure(width=1000, title='MM algorithms performance')

f.line(x=results.index, y=-1 * results['Straight way'], 
       color='blue', legend='Straigh way')
f.line(x=results.index, y=-1 * results['Strassen'],
       color='red', legend='Strassen')
f.line(x=results.index, y=-1 * results['Winograd'],
       color='green', legend='Winograd')

f.legend.location = 'top_left'
f.legend.click_policy = 'hide'
f.xaxis.axis_label = 'Input matrix dimension (Both matrixes are square)'
f.yaxis.axis_label = 'Time (secs)'

show(f)
save(f, 'main_plot.html')
