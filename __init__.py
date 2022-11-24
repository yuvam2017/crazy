#Question 1
dt = {'Boys': [72,68,70,69,74], 'Girls': [63,65,69,62,61]}
[{"Boys":x,"Girls":y} for x,y in zip(dt["Boys"],dt["Girls"])]
# keyarr = []
# output_arr = []
# for key in dt:
#     keyarr.append(key)
# for i in range(len(dt[keyarr[0]])):
#     dict_sample = {keyarr[0]:dt[keyarr[0]][i],keyarr[1]:dt[keyarr[1]][i]}
#     output_arr.append(dict_sample)
# output_arr


#Question 2 
#Write programs in Python using NumPy library to do the following
# (a) Compute the mean, standard deviation, and variance of a two dimensional random integer array along the second axis.
import numpy as np 
import pandas as pd
d = np.random.randint(1,5,size=(3,3))
print(d)
print(d.mean(axis=1))
print(d.std(axis=1))
print(d.var(axis=1))

# (b) Get the indices of the sorted elements of a
# given array.
# a. 
B = [56, 48, 22, 41, 78, 91, 24, 46, 8, 33]
np.argsort(B)

# (C) Create a 2-dimensional array of size m x n integer elements, also print the shape, type and 
# data type of the array and then reshape it into nx
# m array, n and m are user inputs given at the run
# time
n,m = 5,6
df = np.random.randint(1,100,size=(n,m))
print(df.shape)
print(df.dtype)
print(type(df))
df
df.reshape(m,n)


# d.  Test whether the elements of a given array are
# zero, non-zero and NaN. Record the indices of these
# elements in three separate arrays
arr = [5,2,3,6,0,np.nan]
zero,nzero,nan = [],[],[]
for i in range(len(arr)):
    if arr[i] == 0:
        zero.append(i)
    elif np.isnan(arr[i]):
        nan.append(i)
    else:
        nzero.append(i)
print(zero,nzero,nan)




# Ques3:
# Create a dataframe having at least 3 columns
# and 50 rows to store numeric data generated using a
# random function. Replace 10% of the values by null
# values whose index positions are generated using
# random function.Do the following:

df = pd.DataFrame(np.random.randn(50,3))
for i in range(15):
    m = np.random.randint(0,50)
    n = np.random.randint(0,3)
    df.iloc[m,n] = np.nan
df.head()

# (a) Identify and count missing values in a dataframe.
print(df.isna().sum().sum())


# (b) Drop the column having more than 5 null values.
df.drop(df.columns[df.isna().sum() > 5],axis=1).head(5)


# (c) Identify the row label having maximum of the
# sum of all values in a row and drop that row
s = df.sum(axis=1)
df.drop(s[s==s.max()].index).head()


# (d) Sort the dataframe on the basis of the first
# column.
df.sort_values(by=df.columns[0]).head()


# (e) Remove all duplicates from the first column.
df.drop_duplicates(0,keep="first").head()


# (f). Find the correlation between first and second
# column and covariance between 2nd & 3rd column.
df[0].corr(df[1])
df[1].cov(df[2])


# (g) Detect the outliers and remove the rows having
# outliers.
import seaborn as sns
df[0][1] = 200
for i in range(len(df.columns)):
    sns.boxplot(x=df[i])
    arr_ol=np.where(df[i]>100)
    df.drop(arr_ol[0],inplace=True)
df.head()


# (h) Discretize second column and create 5 bins
temp = pd.cut(df[1],[0,20,40,60,80,100])
print(temp)



# Ques 4.
# Consider two excel files having attendance of a
# workshop’s participants for two days. Each file has
# three fields ‘Name’, ‘Time of joining’, duration
# (in minutes) where names are unique within a file.
# Note that durationa may take one of three values
# (30, 40, 50) only. Import the data into two
# dataframes and do the following:

df1 = pd.read_excel("firstFile.xlsx")
df2 = pd.read_excel("secondFile.xlsx")
print("First Day Workshop")
print(df1)
print("Second Day Workshop")
print(df2)


# a. Perform merging of the two dataframes to find
# the names of students who had attended the workshop
# on both days

pd.merge(df1,df2,how="inner",on="Name",indicator=True,suffixes=("_left","_right"))[["Name","_merge"]]


# b. Find names of all students who have attended workshop on either of the days.
pd.merge(df1,df2,how="outer",on="Name",indicator=True,suffixes=("_left","_right"))["Name"]

# c. Merge two data frames row-wise and find the total number of records in the data frame.
l = pd.concat([df1,df2])
len(l)


# d. Merge two data frames and use two columns names
# and duration as multi-row indexes. Generate
# descriptive statistics for this multi-index.
d = pd.merge(df1,df2,how="outer")
d = d.set_index(["Name","Time of Joining "])
d = d.sort_values(by="Name")
d.describe()


# Ques 5.
# Taking Iris data, plot the following with proper
# legend and axis labels: (Download IRIS data from:
# https://archive.ics.uci.edu/ml/datasets/iris or
# import it from sklearn.datasets, or DOWNLOAD FROM
# HERE)

df3 = pd.read_csv("iris.csv")
df3.head()

# (a) Plot bar chart to show the frequency of each class label in the data.
df3.plot(kind="line",figsize=(20,5))

# (b) Draw a scatter plot for Petal width vs sepal
# width.
df3.plot.scatter(x="petal_width",y="sepal_width")

# (c) Plot density distribution for feature petal length.
df3["petal_length"].plot.kde()

# (d) Use a pair plot to show pairwise bivariate distribution in the Iris Dataset.
se.pairplot(df3)

# Ques 6. Consider any sales training/ weather forecasting dataset
df4 = pd.read_csv("weatherHistory.csv")
df4.head()

# a. Compute mean of a series grouped by another series
df4.groupby(["Summary"]).agg({'Temperature (C)':'mean'})

# b. Fill an intermittent time series to replace all  missing dates with values of previous non-missing date.
df4.ffill().head()

# b. Fill an intermittent time series to replace all  missing dates with values of previous non-missing date.
df4.ffill().head()

# c. Perform appropriate year-month string to dates
# conversion.
df4["Formatted Date"] = pd.to_datetime(df4["Formatted Date"],utc=True).apply(lambda x:x.date())
df4["Formatted Date"]


# d. Split a dataset to group by two columns and then sort the aggregated results within the groups.
g = df4.groupby(["Summary","Precip Type"]).agg({"Humidity":sum})
g['Humidity'].groupby(level=0, group_keys=False).nlargest()


# e  Split a given dataframe into groups with bin counts
edges = [0.0,0.3,0.6,1.0]
result=pd.cut(df4['Humidity'],edges)
result
df5 = df4.iloc[:,2:8]
df5
df5['bin']=result
df5.set_index(["bin","Precip Type"])


# Ques 7.
# Consider a data frame containing data about
# students i.e. name, gender and passing division: 

data = {
     "Name":[
        "Mudit Chauhan","Seema Chopra","Rani Gupta",
        "Aditya Narayan","Sanjeev Sahni","Prakash Kumar",
        "Ritu Agarwal","Akshay Goel","Meeta Kulkarni",
        "Preeti Ahuja","Suni Das Gupta","Sonali Sapre",
        "Rashmi Talwar","Ashish Dubey","Kiran Sharma",
        "Sameer Bansal"
    ],

    "Birth_Month":[
        "December","January","March",
        "October","February","December",
        "September","August","July",
        "November","April","January",
        "June","May","February",
        "October"
    ],
    "Gender":[
        "M","F","F","M","M","M",
        "F","M","F","F","M","F",
        "F","M","F","M"
    ],
     "Pass_Division":[
        "III","II","I","I","II",
        "III","I","I","II","II",
        "III","I","III","II","II",
        "I"
    ]
}
df6 = pd.DataFrame(data)
df6

# a. Perform one hot encoding of the last two columns
# of categorical data using the get_dummies()
# function.

pd.get_dummies(df6["Gender"]).join(pd.get_dummies(df6["Pass_Division"]))

# b. Sort this data frame on the “Birth Month” column
# (i.e. January to December). Hint: Convert Month to
# Categorical.
month_categ = ['January','February','March','April','May','June','July','August','September','October','November','December']
df6["Birth_Month"] = pd.Categorical(df6["Birth_Month"],categories=month_categ,ordered = True)
df6.sort_values("Birth_Month",ignore_index=True)



# Ques 8.
# Consider the following data frame containing a
# family name, gender of the family member and
# her/his monthly income in each record.
data = {
     "Name":[
        "Shah","Vats","Vats",
        "Kumar","Vats","Kumar",
        "Shah","Shah","Kumar",
        "Vats"
    ],

    "Gender":[
        "Male","Male","Female","Female",
        "Female","Male","Male","Female",
        "Female","Male"
    ],
     "MonthlyIncome (Rs.)":[
        1_14_000.00,65_000.00,43_150.00,69_500.00,
        1_55_000.00,1_03_000.00,55_000.00,1_12_400.00,
        81_030.00,71_900.00
    ]
}
df7 = pd.DataFrame(data)
df7

# a. Calculate and display familywise gross monthly
# income.
df7.groupby("Name").agg({"MonthlyIncome (Rs.)":'mean'})

# b. Calculate and display the member with the highest monthly income in a family.
df7.groupby("Name").max()

# c. Calculate and display monthly income of all members with income greater than Rs. 60000.00.
df7[df7["MonthlyIncome (Rs.)"]>60_000].sort_values(by="MonthlyIncome (Rs.)")

# d Calculate and display the average monthly income of the female members in the Shah family.
df[(df["Name"]=="Shah") & (df["Gender"]=="Female")].agg({"MonthlyIncome (Rs.)":"mean"})

