# Remember Basics ====
# Clearing the environment

# Set the working directory
setwd('C:/Users/vinay/Desktop')
getwd()
print('hello')
greeting='hello'
print(greeting)
# 1a.R as calculator - Example ====

# Addition
a=2+1
# Subtraction
b=2-1
# Multiplication
c=2*1
# Division
d=a/b
# Modulo division
e=4%%2
# Integer division
f=10%/%6

# Exponentiation
g=2**2 #or 2^2
# 1b.R as calculator - Activity ====

# 1. Choose any number between 2 - 10
# 2. Multiply the given number by 2
# 3. Add 5 to the answer
# 4. Multiply the number by 50
# 5. If you have already celebrated birthday this year add 1769 elseadd 1768
# 6. Subtract the year you were born in

a=3
b=a*2
c=b+5
d=c*50
e=d+1769
f=e-1994
((((((a*2)+5)*50)+1769))-1994)
# Note: You can have multiple operations in single line in the following way:
# ((a +b)*c)


# 2a. Variable assignment - Example ====
# You can use any of the following ways to assign a variable in R
# '<-', '->', '='

# General convention is using '<-'
# Observe that the variable can now be found in the environment


# 2b. Variable assignment - Activity ====

# 1. Think of a number and assign it in a variable
# 2. Multiply the variable by 3
# 3. Add 6 to it
# 4. Divide it by 3
# 4. Subtract number from step-1 from output of step 4

# Note: Create new variables for each step
a=5
b=(((a*5)+6)/3)/a
# 3a. Atomic / Primary data types - Example ====
# R has the following data types
# 1. Numeric / Integer
# 2. Character
# 3. Logical

# Integer or numeric
class(1)
# Character- always in quotes (single or double)
class('hello')
# Logical
class(T)
# 3b. Atomic / Primary data types - Activity ====

# 1. What is the class of a variable 'pct' that has a value '100' in it?
# 2. What is the class of a variable 'miss' that has a value T in it?
# 3. What is the class of a variable 'days' that has a value 20 in it?
pct=100
class(pct)
miss=T
class(miss)
days=20
class(days)
# 4a. Relational operators - Example ====
#Following are the realational operators that are available in R
# == checking for equality
# >= greater than or equal to
# <= less than or equal to
# < less than
# > greater than
# != not equal to

# Note: The output these operators is Logical i.e. whether the relation is True or False

'Wealth' == 'Happiness'

hours_spent_ofc = 4
hours_spent_family = 5 
hours_spent_family >= hours_spent_ofc

time_spent_personal = 6
time_spent_travel = 8
time_spent_personal <= time_spent_travel

years_gone_by = 25
active_life = 75
bal_time = active_life - years_gone_by
years_gone_by < active_life

amt_spent_primary_education = 12 
amt_spent_graduation = 4
amt_spent_graduation < amt_spent_primary_education


# 4b. Relational operators - Activity ====

# 1. Create a variable srh with value 6 in it
# 2. Create a variable csk with value 8 in it
# 3. Check if the difference between the variables is equal to 0
# 4. Check if csk is greater than srh
# 5. Add 3 to srh and check if it is less than csk
# 6. Add 2 to csk and 3 to srh and check if csk is greater than or equal to srh
srh=6
csk=8
csk-srh==0
csk>srh
srh+3>csk
csk+2>=srh
# Note: Do not reassign while adding, it will change the original value

# 5a. Logical operators - Example ====
# "|" or "||" is "OR" operator "|" is vectorised while "||" is not
# "&" or "&&" is "AND" operator "&" is vectorised while "&&" is not
# "!" is NOT operator

T|F
T||F
T&F
T&&F
!T&F
T&!F
!T&!F




# 5b. Logical operators - Activity ====

# 1. Create a variable b1 with a value F in it
# 2. Create a variable b2 with a value T in it
# 3. Use or operator between not of b1 , b2
# 4. Use and operator between not of b1 and not of b2
# 5. Use and operator between b1 and b2

b1=F
b2=T
!b1|b2
!(b1&b2)
b1&b2

# 6a. Complex constructs using logical and relational operators - Example ====


# 6b. Complex constructs using logical and relational operators - Activity ====

# 1. From the above example check the following two conditions using or operator
#     a. Add 3 to srh and check if it is greater than (csk+1)
#     b. Add 2 to srh and check if it is greater than (csk+1)

(srh+3 > csk+1) | (srh+2 > csk+1)
# Secondary datatypes (Data Structures) in R ====

# Vectors: A vector is an object that consists of elements of same data type.
#          Lets see how to create a vector using a c function. c is a generic 
#          function that combines all its arguments.

# Matrices: These are arrays of two dimensions or more but we focus on 2d.
#           The data type for each of these elements should be same as in the 
#           case of vectors.

# Data Frames: Data in the form of a matrix (rows and columns).
#              The columns can be of different data types and type coersion 
#              doesn't happen.
#              This is important because the data we work on contain several 
#              attributes of different data types and it is  essential to preserve them.

# Lists: A list is datastructure that can store multiple data structures.

# 7a. Vectors - Creation and subset - Example ====

#Creating a numeric vector, identify the class
a=c(1,2,3,4)
class(a)
#Creating a character vector
a=c('a','b','c')
class(a)
#Creating a logical vector
a=c(T,F,T)
class(a)
# If we want to create numbers in sequence we can use a ":" operator
1:100
# A refined way to generating sequences
#seq is a function that takes parameters from, to, and by
seq(1,100,by=2)
# Extracting elements from a vector
a[1]
# Observe we are using square brackets for subset

 #To find the 3rd element of a vector
 #To find the 1st 4 elements of a vector
a=c(1,2,3,4)
a[3]
a[1:4]


x1=c(10,'i','24')
class(x1)
x1
x2=c('i','j','T')
class(x2)
x2
x3=c(T,F,11,12)
class(x3)
x3
x4=c('i',10,T,F)
class(x4)
x4
x1[2]
x3[c(2,4)]
x4[1:3]
# 8a. Vectors - Applying logical and relational operators - Example ====

 #Get elements divisible by 2
 #Get elements divisible by 2 or 3
 #Get elements divisible by 5 and 3
a
a[a%%2==0]
a[(a%%2==0)|(a%%3==0)]
a[(a%%5==0)&(a%%3==0)]
# 8b. Vectors - Applying logical and relational operators - Activity ====

# 1. Create a vector srh with the elements ('L','L','W','W','W','L','W','W')
# 2. Find how many elements are there in srh and also number of 'W's
# 3. Create a vector csk with elements ('W','W','L','L','W','W','L','W')
# 4. Get the index numbers where srh and csk have 'W's and find how many are they
srh=c('L','L','W','W','W','L','W','W')
length(srh)
length(srh[srh=='W'])
csk=c('W','W','L','L','W','W','L','W')
which(csk=='W'& srh=='W')
length(which(csk=='W'& srh=='W',TRUE))
length(srh[srh=='W'])
# 9a. Vectors - Operations - Example ====

# 9b. Vectors - Operations - Activity ====

# 1. Create a vector x1 with sequence of numbers from 10 to 100 with step size 3
# 2. Create a vecor x2 with random numbers between 10 to 100 and the number of 
#    elements should be number of elements in x1
# 3. Add the elements of x1 and x2 and get the elements of x1 and x2 that are 
#    divisible by 5
x1=seq(from=10,to=100,by=3)
x2=sample(seq(from=10,to=100,by=1),length(x1))
x3=x1+x2
x3[x3%%5==0]
# Note: To generate random numbers use sample('range','number_of_samples') function

# 10a. Matrices - Creation and subset - Example ====

# Matrix is a function to create a matrix. It has the following arguments - 
# what are the elements in the matrix
# how many rows in matrix / how many columns
# how should the filling of elements is done

A<-matrix(1:6,nrow=3,byrow = T)
A
nrow(A)
ncol(A)




# To get the 2nd element
#To check how the elements are arranged
A[1,2]
# To get second element of first row
# In the square braces, left of comma belongs to rows and 
# right of comma represents columns

#To extract the second row
A[2,]
#To extract the second column
A[,2]
#We can also label the rows and columns of a matrix using the function dimnames
A<-matrix(1:6,nrow=3,byrow = T)
A
# 10b. Matrices - Creation and subset - Activity ====

# 1. Create a 'm1' matrix with 16 elements from a sequence of randomly generated numbers
#    between 1:100
# 2. Extract all the rows of matrix 'm1' where the 'column 2' values are divisible by '2'
m1=matrix(sample(seq(1,100),16),nrow=4,byrow=T)
m1[,2][m1[,2]%%2==0]
# Note arrange the elements by rows and the matrix should have 4 columns

# 11a. Matrices - Operations - Example ====

#"+" addition
#"-" subtraction
#"*" is element wise multiplication
#"%*%" is matrix multiplication

m1 = matrix(1:6,nrow = 2,byrow = T)
m2 = matrix(1:6,nrow = 2,byrow = F)
m3 = matrix(1:6,nrow = 3,byrow = T)

m1
m2
m1 + m2
m1 - m2
m1 * m2
m1 %*% m3

# 11b. Matrices - Operations - Activity ====

# 1. Create a matrix m1 with elements first 9 odd numbers arranged row wise in 3 rows
# 2. Create a matrix m2 with elements first 9 even numbers arranged row wise in 3 rows
# 3. Add the matrices m1 and m2 and extract the rows where the values of 1st column 
#    are divisible by 5
# 4. Divide the Matrix from above resultant matrix's 3rd column values by 2 and 
#    replace the orig values
# 5. Find the transpose of matrix m2 and multiply with the matrix in step 4
m1=matrix(seq(1,by=2,length=9),nrow=3,byrow=T)
m2=matrix(seq(2,by=2,length=9),nrow=3,byrow=T)
m1
m2
t(m1)
t(m2)
# Note: Use t(matrix) to find the transpose of the matrix

# 12a. Matrices - Frequently used functions - Example ====

#Lets create two vectors and bind them to create matrix

# cbind to bind the matrices/data frames by columns
# rbind to bind the matrices/data frames by rows
a=c(1,2,3,4)
b=c(2,3,4,5)
c=cbind(a,b)
d=rbind(a,b)
# 12b. Matrices - Frequently used functions - Activity ====

# 1. Using cbind create a matrix 'm1' with even and odd numbers from the above experiment
# 2. Using rbind create a matrix 'm2' with even and odd numbers from the above experiment
# 3. From the matrix 'm2' find the column numbers of elements from 2nd row divisible by 5
#   and subset all the rows and resultant columns
m1=cbind(seq(1,by=2,length=9),seq(2,by=2,length=9))
m1
m2=rbind(seq(1,by=2,length=9),seq(2,by=2,length=9))
m2
m2[2,which(m2[2,]%%5==0,arr.ind=TRUE)]
m2
# 13a. Data frames - Creation and subset - Example ====
library(data.table)
install.packages('data.table')

a=data.frame(columna=c(1,2,3,4,5),columnb=c(6,7,8,9,20),columnc=c(1,2,3,4,5),columnd=c(1,2,3,4,5))
a
# To access the elements of 2nd column
a[,2]
# To access the elements of column 'C'
a['columna']

# To access the 2nd and 4th rows of columns 1 & 3
a[c(2,1),c(4,3)]
a
# 13b. Data frames - Creation and subset - Activity ====

# 1. Create a vector age with values (10,14,12,15,13)
# 2. Create a vector gender with values ('M','F','F','M','M')
# 3. Create a vector group with values ('low','mid','mid','high','mid')
# 4. Create a dataframe df1 with columns age, gender and group
# 5. Get the records where gender is 'F'
# 6. Get the details of column 'group' where the age values is greater than 10 and less than 15
age=c(10,14,12,15,13)
gender=c('M','F','F','M','M')
group=c('low','mid','mid','high','mid')
df1=data.frame(age,gender,group)
subset(df1,gender=='F')
subset(df1$group,age>10&age<15)

# 14a. Data frames - Frequently used functions - Example ====

## To understand the structure of dataframe
str(df1)
#To look into the summary of the dataframe
summary(df1)
str(df1)
#To access a column by name we use "$"
df1$age
#The functions str, summary also work on individual columns
str(df1$age)
summary(df1$age)
#We have a dataframe DF and we want to know the dimensions of it
dim(df1)
df1
#To get the number of rows and number of columns of DF-nrow and ncol respectively
nrow(df1)
ncol(df1)
# 14b. Data frames - Frequently used functions - Activity ====

# 1. Get the column values of gender and group where the age is minimum in df1
# 2. Find the number of unique elements in group and gender columns of df1
# 3. Find the number of rows where the gender is F
# 4. Find the ages of records where the gender is M and group is high

which(df1$age==min(df1$age))
length(unique(df1$age))+length(unique(df1$group)
nrow(subset(df1,df1$gender=='F'))
subset(df1,df1$gender=='M'&df1$group=='high')$age

# 15a. Lists - Creation and access - Example ====

val=c(1,7,9)
ch=c("A","X","Z")
A<-list(val,ch,data.frame(val,ch),list(val,ch))

# To get the first element of the list which is val. But we know that val has 3 elements in it
A[[1]]
# To get the first element of the val
A[[1]][1]
# Since the third element of the list is a dataframe we can access the elements 
# of the dataframe using $
A[[3]]$val

# Elements of list can also be accessed using names if they have
B=list(M=val,N=ch,O=data.frame(val,ch),P=list(val,ch))
B$M
#To unlist the elements
unlist(B)
# 15b. Lists - Creation and access - Activity ====

# 1. Create a list l1 with matrix m3, df1 and age
# 2. Find the length of list l1 and type of each element
# Note use class(object) to find the type / class of object
# 3. Get the elements of 2nd row from the data frame in the list
l1=list(m3,df1,age)
str(l1)
l1[[2]][2,]