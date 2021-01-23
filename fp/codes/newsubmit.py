import pandas as pd
import numpy
from preprocessing import each_day_revenue

revenue = each_day_revenue()
r = [x[0] for x in revenue]
r = numpy.array(r)
a=pd.read_csv('submit.csv')
a['real-revenue']=r
a.to_csv('submit.csv')

