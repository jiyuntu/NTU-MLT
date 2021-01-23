import pandas as pd
import numpy as np
import csv
from datetime import datetime, date
import os

train = pd.read_csv("./newtrain.csv")
train_label = pd.read_csv("./train_label.csv")
test = pd.read_csv("./test.csv")
test_label = pd.read_csv("./test_nolabel.csv")

def what_day_is_that_day(y, m, d):
	d1 = datetime.today()
	d0 = datetime(int(y), int(m), int(d))
	delta = d1 - d0
	week = d1.weekday()
	week = (week + delta.days % 7) % 7
	return week
	
month2number = {
	"January" : "01", "February" : "02", "March" : "03", "April" : "04", "May" : "05", "June" : "06",
	"July" : "07", "August" : "08", "September" : "09", "October" : "10",
	"November" : "11", "December" : "12"
}
dateplus0 = {1: '01', 2: '02', 3: '03',4: '04', 5: '05', 6: '06', 7: '07', 8: '08', 9: '09'}

def each_day_revenue():
	date = pd.DataFrame(train, columns = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]).values.tolist()
	stay = pd.DataFrame(train, columns = ["stays_in_weekend_nights", "stays_in_week_nights"]).values.tolist()
	adr = pd.DataFrame(train, columns = ["adr"]).values.tolist()
	cancel = pd.DataFrame(train, columns = ["is_canceled"]).values.tolist()
	for i, lst in enumerate(date):
		lst[0] = str(lst[0])
		lst[1] = month2number[lst[1]]
		if lst[2] < 10:
			lst[2] = dateplus0[lst[2]]
		else:
			lst[2] = str(lst[2])
		date[i] = ''.join(lst)
	revenue = [] #revenue[i] = [date, revenue]
	for i in range(len(date)):
		if cancel[i][0] == 1:
			continue
		last = len(revenue) - 1
		if last < 0 or revenue[last][0] != date[i]:
			revlist = []
			revlist.append(date[i])
			buf = (stay[i][0] + stay[i][1]) * adr[i][0]
			revlist.append(buf)
			revenue.append(revlist)
		else:
			revenue[last][1] += (stay[i][0] + stay[i][1]) * adr[i][0]
	return revenue

def no_cancel_revenue():
	date = pd.DataFrame(train, columns = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]).values.tolist()
	stay = pd.DataFrame(train, columns = ["stays_in_weekend_nights", "stays_in_week_nights"]).values.tolist()
	adr = pd.DataFrame(train, columns = ["adr"]).values.tolist()
	cancel = pd.DataFrame(train, columns = ["is_canceled"]).values.tolist()
	for i, lst in enumerate(date):
		lst[0] = str(lst[0])
		lst[1] = month2number[lst[1]]
		if lst[2] < 10:
			lst[2] = dateplus0[lst[2]]
		else:
			lst[2] = str(lst[2])
		date[i] = ''.join(lst)
	revenue = [] #revenue[i] = [date, revenue]
	for i in range(len(date)):
		last = len(revenue) - 1
		if last < 0 or revenue[last][0] != date[i]:
			revlist = []
			revlist.append(date[i])
			buf = (stay[i][0] + stay[i][1]) * adr[i][0]
			revlist.append(buf)
			revenue.append(revlist)
		else:
			revenue[last][1] += (stay[i][0] + stay[i][1]) * adr[i][0]
	return revenue

def revenue_on_right_day():
	date = pd.DataFrame(train, columns = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]).values.tolist()
	stay = pd.DataFrame(train, columns = ["stays_in_weekend_nights", "stays_in_week_nights"]).values.tolist()
	adr = pd.DataFrame(train, columns = ["adr"]).values.tolist()
	cancel = pd.DataFrame(train, columns = ["is_canceled"]).values.tolist()
	for i, lst in enumerate(date):
		lst[0] = str(lst[0])
		lst[1] = month2number[lst[1]]
		if lst[2] < 10:
			lst[2] = dateplus0[lst[2]]
		else:
			lst[2] = str(lst[2])
		date[i] = ''.join(lst)
	revenue = [] #revenue[i] = [date, revenue]
	night = []
	for i in range(len(date)):
		last = len(revenue) - 1
		if last < 0 or date[i] != revenue[last][0]:
			revenue.append([date[i], 0])
			night.append([date[i], 0])
	curr = 0
	for i in range(len(date)):
		if cancel[i][0] == 1:
			continue
		while date[i] != revenue[curr][0]:
			curr += 1
		revenue[curr][1] += adr[i][0]
		night[curr][1] += 1
		j = curr + 1
		while j < len(revenue) and j - curr < (stay[i][0] + stay[i][1]):
			revenue[j][1] += adr[i][0]
			night[j][1] += 1
			j += 1
	return revenue, night
	

def make_X():
	date = pd.DataFrame(train, columns = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]).values.tolist()
	stay = pd.DataFrame(train, columns = ["stays_in_weekend_nights", "stays_in_week_nights"]).values.tolist()
	adr = pd.DataFrame(train, columns = ["adr"]).values.tolist()
	cancel = pd.DataFrame(train, columns = ["is_canceled"]).values.tolist()
	weekday = []
	for i in range(len(date)):
		weekday.append(what_day_is_that_day(date[i][0], month2number[date[i][1]], date[i][2]))
# generate revenue and date
	for i, lst in enumerate(date):
		lst[0] = str(lst[0])
		lst[1] = month2number[lst[1]]
		if lst[2] < 10:
			lst[2] = dateplus0[lst[2]]
		else:
			lst[2] = str(lst[2])
		date[i] = ''.join(lst)

# generate X, the data set
	feature = pd.DataFrame(train, columns = ['lead_time', 'total_of_special_requests', 'adults', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'required_car_parking_spaces']).values.tolist()
# hotel_type
	hotel_type = pd.DataFrame(train, columns = ['hotel']).values.tolist()
# how long is the time to stay
	stay = pd.DataFrame(train, columns = ["stays_in_weekend_nights", "stays_in_week_nights"]).values.tolist()
# the number of children
	child_n = pd.DataFrame(train, columns = ['children', 'babies'])
	child_n['children'] =  child_n['children'].fillna(0)
	child_n = child_n.values.tolist()
# market segment
	market = pd.DataFrame(train, columns = ['market_segment']).values.tolist()
# whether room type is the same as required
	room_type = pd.DataFrame(train, columns = ["reserved_room_type", "assigned_room_type"]).values.tolist()
# when the customers enter the hotel
	enter_week = pd.DataFrame(train, columns = ["arrival_date_week_number"]).values.tolist()
# meal_type
	meal = pd.DataFrame(train, columns = ["meal"]).values.tolist()
# order_first
	deposit = pd.DataFrame(train, columns = ["deposit_type"]).values.tolist()
# customer type
	custom = pd.DataFrame(train, columns = ['customer_type']).values.tolist()
# booking from what country
	country = pd.DataFrame(train, columns = ['country']).values.tolist()

	y = pd.DataFrame(train, columns = ["adr"]).values.tolist()
	for i in range(len(feature)):
# what kind of hotel we stay
		feature[i].append(hotel_type[i][0] == "Resort Hotel")
		feature[i].append(hotel_type[i][0] == "City Hotel")
# the number of day we stay
		feature[i].append(stay[i][0] + stay[i][1])
# the number of children, inclusive of babies
		feature[i].append(child_n[i][0] + child_n[i][1])
# the weekday we start staying
		for j in range(7):
			feature[i].append(int(weekday[i] == j))
# all stay weekend, all stay weekday, both weekend and weekday
		if stay[i][0] > 0 and stay[i][1] > 0:
			feature[i].append(1)
		else:
			feature[i].append(0)
		if stay[i][0] > 0 and stay[i][1] == 0:
			feature[i].append(1)
		else:
			feature[i].append(0)
		if stay[i][0] == 0 and stay[i][1] > 0:
			feature[i].append(1)
		else:
			feature[i].append(0)
# the number of children, inclusive of babies
		feature[i].append(child_n[i][0] + child_n[i][1])
# market segment
		feature[i].append(market[i][0] == 'Offline TA/TO')
		feature[i].append(market[i][0] == 'Online TA')
		feature[i].append(market[i][0] == 'Direct')
		feature[i].append(market[i][0] == 'Groups')
		feature[i].append(market[i][0] == 'Corporate')
		feature[i].append(market[i][0] == 'Complementary')
# is the room the same as requested
		feature[i].append(room_type[i][0] == room_type[i][1])
# what kind of room we ask ADEFBGC
		feature[i].append(room_type[i][0] == 'A')
		feature[i].append(room_type[i][0] == 'D')
		feature[i].append(room_type[i][0] == 'E')
		feature[i].append(room_type[i][0] == 'F')
		feature[i].append(room_type[i][0] == 'B')
		feature[i].append(room_type[i][0] == 'G')
		feature[i].append(room_type[i][0] == 'C')
# what kind of room we actually stay
		feature[i].append(room_type[i][1] == 'A')
		feature[i].append(room_type[i][1] == 'D')
		feature[i].append(room_type[i][1] == 'E')
		feature[i].append(room_type[i][1] == 'F')
		feature[i].append(room_type[i][1] == 'B')
		feature[i].append(room_type[i][1] == 'G')
		feature[i].append(room_type[i][1] == 'C')
# Which week we go to the hotel
		for j in range(53):
			feature[i].append(enter_week[i][0] == (j + 1))
# meal
		feature[i].append(meal[i][0] == 'BB')
		feature[i].append(meal[i][0] == 'HB')
		feature[i].append(meal[i][0] == 'SC')
# deposit
		feature[i].append(deposit[i][0] == 'No Deposit')
		feature[i].append(deposit[i][0] == 'Non Refund')
# customer type
		feature[i].append(custom[i][0] == 'Transient')
		feature[i].append(custom[i][0] == 'Transient-Party')
		feature[i].append(custom[i][0] == 'Group')
		feature[i].append(custom[i][0] == 'Contract')
# country
		feature[i].append(country[i][0] == 'PRT')
		feature[i].append(country[i][0] == 'GBR')
		feature[i].append(country[i][0] == 'FRA')
		feature[i].append(country[i][0] == 'ESP')
		feature[i].append(country[i][0] == 'DEU')
		feature[i].append(country[i][0] == 'ITA')
		feature[i].append(country[i][0] == 'IRL')
		feature[i].append(country[i][0] == 'BRA')
		feature[i].append(country[i][0] == 'NLD')
		feature[i].append(country[i][0] == 'BEL')
		feature[i].append(country[i][0] == 'USA')
		feature[i].append(country[i][0] == 'CHE')
		feature[i].append(country[i][0] == 'AUT')
		feature[i].append(country[i][0] == 'CHN')
		feature[i].append(country[i][0] == 'CN')
		feature[i].append(country[i][0] == 'POL')
		feature[i].append(country[i][0] == 'SWE')
		feature[i] = [int(feature[i][j]) for j in range(len(feature[i]))]
	return feature, date

def make_cancel():
	cancel = pd.DataFrame(train, columns = ["is_canceled"]).values.tolist()
	canc = []
	for i in range(len(cancel)):
		if cancel[i][0] == 1:
			canc.append([0, 1])
		else:
			canc.append([1, 0])
	return canc

def make_adr():
	adr = pd.DataFrame(train, columns = ["adr"]).values.tolist()
	return [adr[i][0] for i in range(len(adr))]

def make_test_X():
	date = pd.DataFrame(test, columns = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]).values.tolist()
	adr = pd.DataFrame(test, columns = ["adr"]).values.tolist()
	cancel = pd.DataFrame(test, columns = ["is_canceled"]).values.tolist()
	weekday = []
	for i in range(len(date)):
		weekday.append(what_day_is_that_day(date[i][0], month2number[date[i][1]], date[i][2]))
# generate revenue and date
	for i, lst in enumerate(date):
		lst[0] = str(lst[0])
		lst[1] = month2number[lst[1]]
		if lst[2] < 10:
			lst[2] = dateplus0[lst[2]]
		else:
			lst[2] = str(lst[2])
		date[i] = ''.join(lst)

# generate X, the data set
	feature = pd.DataFrame(test, columns = ['lead_time', 'total_of_special_requests', 'adults', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'required_car_parking_spaces']).values.tolist()
# hotel_type
	hotel_type = pd.DataFrame(test, columns = ['hotel']).values.tolist()
# how long is the time to stay
	stay = pd.DataFrame(test, columns = ["stays_in_weekend_nights", "stays_in_week_nights"]).values.tolist()
# the number of children
	child_n = pd.DataFrame(test, columns = ['children', 'babies'])
	child_n['children'] =  child_n['children'].fillna(0)
	child_n = child_n.values.tolist()
# market segment
	market = pd.DataFrame(test, columns = ['market_segment']).values.tolist()
# whether room type is the same as required
	room_type = pd.DataFrame(test, columns = ["reserved_room_type", "assigned_room_type"]).values.tolist()
# Which week we enter the hotel
	enter_week = pd.DataFrame(test, columns = ['arrival_date_week_number']).values.tolist()
# meal_type
	meal = pd.DataFrame(test, columns = ["meal"]).values.tolist()
# order_first
	deposit = pd.DataFrame(test, columns = ["deposit_type"]).values.tolist()
# customer type
	custom = pd.DataFrame(test, columns = ['customer_type']).values.tolist()
# booking from what country
	country = pd.DataFrame(test, columns = ['country']).values.tolist()

	y = pd.DataFrame(test, columns = ["adr"]).values.tolist()
	for i in range(len(feature)):
# what kind of hotel we stay
		feature[i].append(hotel_type[i][0] == "Resort Hotel")
		feature[i].append(hotel_type[i][0] == "City Hotel")
# the number of day we stay
		feature[i].append(stay[i][0] + stay[i][1])
# the number of children, inclusive of babies
		feature[i].append(child_n[i][0] + child_n[i][1])
# the weekday we start staying
		for j in range(7):
			feature[i].append(int(weekday[i] == j))
# all stay weekend, all stay weekday, both weekend and weekday
		if stay[i][0] > 0 and stay[i][1] > 0:
			feature[i].append(1)
		else:
			feature[i].append(0)
		if stay[i][0] > 0 and stay[i][1] == 0:
			feature[i].append(1)
		else:
			feature[i].append(0)
		if stay[i][0] == 0 and stay[i][1] > 0:
			feature[i].append(1)
		else:
			feature[i].append(0)
# the number of children, inclusive of babies
		feature[i].append(child_n[i][0] + child_n[i][1])
# market segment
		feature[i].append(market[i][0] == 'Offline TA/TO')
		feature[i].append(market[i][0] == 'Online TA')
		feature[i].append(market[i][0] == 'Direct')
		feature[i].append(market[i][0] == 'Groups')
		feature[i].append(market[i][0] == 'Corporate')
		feature[i].append(market[i][0] == 'Complementary')
# is the room the same as requested
		feature[i].append(room_type[i][0] == room_type[i][1])
# what kind of room we ask ADEFBGC
		feature[i].append(room_type[i][0] == 'A')
		feature[i].append(room_type[i][0] == 'D')
		feature[i].append(room_type[i][0] == 'E')
		feature[i].append(room_type[i][0] == 'F')
		feature[i].append(room_type[i][0] == 'B')
		feature[i].append(room_type[i][0] == 'G')
		feature[i].append(room_type[i][0] == 'C')
# what kind of room we actually stay
		feature[i].append(room_type[i][1] == 'A')
		feature[i].append(room_type[i][1] == 'D')
		feature[i].append(room_type[i][1] == 'E')
		feature[i].append(room_type[i][1] == 'F')
		feature[i].append(room_type[i][1] == 'B')
		feature[i].append(room_type[i][1] == 'G')
		feature[i].append(room_type[i][1] == 'C')
# Which week we go to the hotel
		for j in range(53):
			feature[i].append(enter_week[i][0] == (j + 1))
# meal
		feature[i].append(meal[i][0] == 'BB')
		feature[i].append(meal[i][0] == 'HB')
		feature[i].append(meal[i][0] == 'SC')
# deposit
		feature[i].append(deposit[i][0] == 'No Deposit')
		feature[i].append(deposit[i][0] == 'Non Refund')
# customer type
		feature[i].append(custom[i][0] == 'Transient')
		feature[i].append(custom[i][0] == 'Transient-Party')
		feature[i].append(custom[i][0] == 'Group')
		feature[i].append(custom[i][0] == 'Contract')
# country
		feature[i].append(country[i][0] == 'PRT')
		feature[i].append(country[i][0] == 'GBR')
		feature[i].append(country[i][0] == 'FRA')
		feature[i].append(country[i][0] == 'ESP')
		feature[i].append(country[i][0] == 'DEU')
		feature[i].append(country[i][0] == 'ITA')
		feature[i].append(country[i][0] == 'IRL')
		feature[i].append(country[i][0] == 'BRA')
		feature[i].append(country[i][0] == 'NLD')
		feature[i].append(country[i][0] == 'BEL')
		feature[i].append(country[i][0] == 'USA')
		feature[i].append(country[i][0] == 'CHE')
		feature[i].append(country[i][0] == 'AUT')
		feature[i].append(country[i][0] == 'CHN')
		feature[i].append(country[i][0] == 'CN')
		feature[i].append(country[i][0] == 'POL')
		feature[i].append(country[i][0] == 'SWE')
		feature[i] = [int(feature[i][j]) for j in range(len(feature[i]))]
	return feature, date
