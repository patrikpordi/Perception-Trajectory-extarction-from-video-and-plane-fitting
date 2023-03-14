import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from math import nan
from IPython.display import Image
from matplotlib import cm

#1.1
##################################################################
# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv.VideoCapture('/home/pordipatrik/Perception/ball.mov')
if (vid_capture.isOpened() == False):
  print("Error opening the video file")
# Read fps and frame count
else:
  # Get frame rate information
  # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
  fps = vid_capture.get(5)
  print('Frames per second : ', fps,'FPS')
 
  # Get frame count
  # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
  frame_count = vid_capture.get(7)
  print('Frame count : ', frame_count)

# Creating two lists to store the path of the ball
x=[]
y=[]
while(vid_capture.isOpened()):
    # vid_capture.read() methods returns a tuple, first element is a bool 
    # and the second is frame
    ret, frame = vid_capture.read()
    if ret == True:
        
        # Bluring, conversion to hsv, creating limits, creating the mask, creating the results using the mask
        blurred = cv.GaussianBlur(frame, (11, 11), 0)
        hsv=cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        lower_red=np.array([0,170,90])
        upper_red=np.array([5,255,255])
        mask=cv.inRange(hsv,lower_red, upper_red)
        res=cv.bitwise_and(frame,frame,mask=mask)
        cv.imshow('mask', mask)
        cv.imshow('frame', frame)
        cv.imshow('res',res)

        # Extracting the points that represents the path of the ball, and calculate their mean
        yis, xis = np.nonzero(mask)
        a=xis.mean()
        b=yis.mean()
        x.append(a)
        y.append(-b)
        
        key = cv.waitKey(30)
        if key == ord('q'):
            break
    else:
        break
 
# Release the video capture object
vid_capture.release()
cv.destroyAllWindows()

#Plotting the path of the ball
plt.scatter(x,y)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
##########################################################################
#1.2
#########################################################################
# First we filter out the useless data
x=[number for number in x if str(number)!="nan"]
y=[number for number in y if str(number)!="nan"]

## Creating the required parts for the system of equations
second=[number**2 for number in x]
third=[number**3 for number in x]
fourth=[number**4 for number in x]
xy=[]
for i in range(0,len(x)):
    xy.append(x[i]*y[i])
xxy=[]
for i in range(0,len(x)):
    xxy.append(second[i]*y[i])

# Creating the matrices for the system of equations
A=np.array([[len(x),np.sum(x),np.sum(second)],
[np.sum(x),np.sum(second),np.sum(third)],
[sum(second),np.sum(third),np.sum(fourth)]])
B=np.array([np.sum(y),np.sum(xy),np.sum(xxy)])
h=np.linalg.solve(A,B)
def f(x):
    return h[0]+h[1]*x+h[2]*x**2
g=np.linspace(0,1200,50)
plt.scatter(x,y, Color="g",s=2)
plt.plot(g,f(g), Color="b")
plt.xlabel("X")
plt.ylabel("Y")
# Plotting the curve
plt.show()
print("Equation of the curve: x^2*" ,h[2], "x*",h[1],"+",h[0])
###################################################################
#1.3
###################################################################
# Finding the solutions of the equation using the given information
x_1=(-h[1]+np.sqrt(h[1]**2-4*(h[2]*(h[0]-y[0]+300))))/(2*h[2])
x_2=(-h[1]-np.sqrt(h[1]**2-4*(h[2]*(h[0]-y[0]+300))))/(2*h[2])
print(max(x_1,x_2),",",y[0]-300)
######################################################################
#2.1
#####################################################################
#The texts have to be inserted and the x,y,z coordinates are stored separately
A=np.loadtxt("pc1.csv",delimiter=",",dtype=float)
B=np.loadtxt("pc2.csv",delimiter=",",dtype=float)
x=A[:,0]
y=A[:,1]
z=A[:,2]


# Means and the deifference between actual values and means are calculated
x_m=np.mean(x, axis=0).reshape(-1,1)
y_m=np.mean(y, axis=0).reshape(-1,1)
z_m=np.mean(z, axis=0).reshape(-1,1)
x_d=x-x_m
y_d=y-y_m
z_d=z-z_m

# Variancies are calculated
var_x=(x_d@x_d.T)/len(x)
var_y=(y_d@y_d.T)/len(x)
var_z=(z_d@z_d.T)/len(x)

# Covariancies are calculated
cov_xy=(x_d@y_d.T)/len(x)
cov_yz=(y_d@z_d.T)/len(x)
cov_xz=(x_d@z_d.T)/len(x)
cov_m=np.array([[var_x[0,0],cov_xy[0,0],cov_xz[0,0]],[cov_xy[0,0],var_y[0,0],cov_yz[0,0]],[cov_xz[0,0],cov_yz[0,0],var_z[0,0]]])
print("The covariance matrix is the following:")
print(cov_m)

# Calculating the eigenvalues and eigenvectors of the covariance matrix and then extracting the 
E_val,E_vec=np.linalg.eig(cov_m)
print("Direction:",E_vec[np.where(min(E_val[0],E_val[1],E_val[2]))[0][0]])
print("Magnitude:",np.linalg.norm(E_vec[np.where(min(E_val[0],E_val[1],E_val[2]))[0][0]]))
#########################################################################################
#2.2a
#########################################################################################
#pcs1
# The texts have to be inserted and the x,y,z coordinates are stored separately
A=np.loadtxt("pc1.csv",delimiter=",",dtype=float)
x=A[:,0]
y=A[:,1]
z=A[:,2]

# System matrix is made and the a,b,c values for the plane equation are calculated
M=np.column_stack((x,y, np.ones(len(A))))
a,b,c= np.linalg.inv(M.T@ M) @ M.T @z

# Creating the meshgrid for ploting the plane and calculating the corresponding z-t value
x_f, y_f=np.meshgrid(np.linspace(np.min(x),np.max(x),1000),np.linspace(np.min(y),np.max(y),1000))
z_f=a*x_f+b*y_f+c

# Ploting the points and corresponding plane
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(x,y,z)
ax.plot_surface(x_f,y_f,z_f,alpha=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

#pcs2
# The texts have to be inserted and the x,y,z coordinates are stored separately
A=np.loadtxt("pc2.csv",delimiter=",",dtype=float)
x=A[:,0]
y=A[:,1]
z=A[:,2]

# System matrix is made and the a,b,c values for the plane equation are calculated
M=np.column_stack((x,y, np.ones(len(A))))
a,b,c= np.linalg.inv(M.T@ M) @ M.T @z

# Creating the meshgrid for ploting the plane and calculating the corresponding z-t value
x_f, y_f=np.meshgrid(np.linspace(np.min(x),np.max(x),1000),np.linspace(np.min(y),np.max(y),1000))
z_f=a*x_f+b*y_f+c

# Ploting the points and corresponding plane
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(x,y,z)
ax.plot_surface(x_f,y_f,z_f,alpha=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

#Total pcs1
# The texts have to be inserted and the x,y,z coordinates are stored separately
A=np.loadtxt("pc1.csv",delimiter=",",dtype=float)
x=A[:,0]
y=A[:,1]
z=A[:,2]

# Calculating U and its eigenvalues and the eigenvector corresponding to the smalest eigenvalue 
cent_of_mass = np.mean(A,axis=0)
Am=A-cent_of_mass
U = Am.T@Am
E_val, E_vec = np.linalg.eig(U)
E_vec_s = E_vec[:,np.argmin(E_val)]

# Creating the meshgrid for ploting the plane and calculating the corresponding z-t value
x_f, y_f=np.meshgrid(np.linspace(np.min(x),np.max(x),1000),np.linspace(np.min(y),np.max(y),1000))
d=E_vec_s[0]*cent_of_mass[0]+E_vec_s[1]*cent_of_mass[1]+E_vec_s[2]*cent_of_mass[2]
z_f = (-E_vec_s[0]*x_f - E_vec_s[1]*y_f + d)/E_vec_s[2]

# Ploting the points and corresponding plane
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.plot_surface(x_f, y_f, z_f, alpha= 0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
print(E_vec_s[0],E_vec_s[1],E_vec_s[2],d)
#Total pcs2
# The texts have to be inserted and the x,y,z coordinates are stored separately
A=np.loadtxt("pc2.csv",delimiter=",",dtype=float)
x=A[:,0]
y=A[:,1]
z=A[:,2]

# Calculating U and its eigenvalues and the eigenvector corresponding to the smalest eigenvalue 
cent_of_mass = np.mean(A,axis=0)
Am=A-cent_of_mass
U = Am.T@Am
E_val, E_vec = np.linalg.eig(U)
E_vec_s = E_vec[:,np.argmin(E_val)]

# Creating the meshgrid for ploting the plane and calculating the corresponding z-t value
x_f, y_f=np.meshgrid(np.linspace(np.min(x),np.max(x),1000),np.linspace(np.min(y),np.max(y),1000))
d=E_vec_s[0]*cent_of_mass[0]+E_vec_s[1]*cent_of_mass[1]+E_vec_s[2]*cent_of_mass[2]
z_f = (-E_vec_s[0]*x_f - E_vec_s[1]*y_f + d)/E_vec_s[2]

# Ploting the points and corresponding plane
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.plot_surface(x_f, y_f, z_f, alpha= 0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
#####################################
#2.2b
###################################
#pcs1
# The texts have to be inserted and the x,y,z coordinates are stored separately
A=np.loadtxt("pc1.csv",delimiter=",",dtype=float)
x=A[:,0]
y=A[:,1]
z=A[:,2]

# Function for constructing the plane
def plane(sample):
    v1 = np.array(sample[1])-np.array(sample[0])
    v2 = np.array(sample[2])-np.array(sample[0])
    a,b,c = np.cross(v1,v2)
    l = -np.dot(np.cross(v1,v2),np.array(sample[0]))
    return a,b,c,l

# Function for calculating the distance of a point from a plane
def distance(a,b,c,l,x,y,z):
    distance=(abs(a*x+b*y+c*z+l))/(math.sqrt(a*a+b*b+c*c))
    return distance

# Function for RANSAC
def ransac (Data):
    #Initializing the variables,3 points,200 iterations, 90%treshold
    n,k,d,l = 3,200,2,len(Data)*0.9  
    t_plane = None
    t_points = []
    for i in range(k):
        #Creating a plane based on random points
        plane_temp = plane(Data[np.random.choice(len(Data), n, replace=False)])
        points_in=[]
        # Checking wheteher the distance is less than the limit, if yes the points are stored
        for points in Data:
            dist = distance(plane_temp[0],plane_temp[1], plane_temp[2], plane_temp[3], points[0],points[1],points[2])
            if(dist<d):
                points_in.append(points)
        # Checking if we are in the first iteration or the current iteration is better than the earlier results and tha treshold
        in_q = len(points_in)
        if(in_q>=l) and (t_plane is None or in_q>len(t_points)):
            t_plane = plane(points_in)
            t_points = points_in
            print(in_q/len(Data))

    # Saving the plane and the points as np.array to be able to return it
    t_plane_np = np.array(t_plane)   
    t_points_np = np.array(t_points)

    return t_plane_np, t_points_np

# Extracting the plane and the inlier points
r_plane, r_points = ransac(A)

# Creating the meshgrid for ploting the plane and calculating the corresponding z-d value
x_f, y_f=np.meshgrid(np.linspace(np.min(x),np.max(x),1000),np.linspace(np.min(y),np.max(y),1000))
z = (-r_plane[0] * x_f - r_plane[1] * y_f - r_plane[3]) / r_plane[2]

# Ploting the points and corresponding plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(A[:,0], A[:,1], A[:,2], c='r', marker='o')
ax.scatter(r_points[:,0], r_points[:,1], r_points[:,2], c='g', marker='o')

ax.plot_surface(x_f, y_f, z,alpha= 0.2, Color='g', cmap=cm.twilight)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

#pcs2
# The texts have to be inserted and the x,y,z coordinates are stored separately
A=np.loadtxt("pc2.csv",delimiter=",",dtype=float)
x=A[:,0]
y=A[:,1]
z=A[:,2]

# Function for constructing the plane
def plane(sample):
    v1 = np.array(sample[1])-np.array(sample[0])
    v2 = np.array(sample[2])-np.array(sample[0])
    a,b,c = np.cross(v1,v2)
    l = -np.dot(np.cross(v1,v2),np.array(sample[0]))
    return a,b,c,l

# Function for calculating the distance of a point from a plane
def distance(a,b,c,l,x,y,z):
    distance=(abs(a*x+b*y+c*z+l))/(math.sqrt(a*a+b*b+c*c))
    return distance

# Function for RANSAC
def ransac (Data):
    #Initializing the variables,3 points,200 iterations, 90%treshold
    n,k,d,l = 3,200,2.4,len(Data)*0.9  
    t_plane = None
    t_points = []
    for i in range(k):
        #Creating a plane based on random points
        plane_temp = plane(Data[np.random.choice(len(Data), n, replace=False)])
        points_in=[]
        # Checking wheteher the distance is less than the limit, if yes the points are stored
        for points in Data:
            dist = distance(plane_temp[0],plane_temp[1], plane_temp[2], plane_temp[3], points[0],points[1],points[2])
            if(dist<d):
                points_in.append(points)
        # Checking if we are in the first iteration or the current iteration is better than the earlier results and tha treshold
        in_q = len(points_in)
        if(in_q>=l) and (t_plane is None or in_q>len(t_points)):
            t_plane = plane(points_in)
            t_points = points_in
            print(in_q/len(Data))

    # Saving the plane and the points as np.array to be able to return it
    t_plane_np = np.array(t_plane)   
    t_points_np = np.array(t_points)

    return t_plane_np, t_points_np

# Extracting the plane and the inlier points
r_plane, r_points = ransac(A)

# Creating the meshgrid for ploting the plane and calculating the corresponding z-d value
x_f, y_f=np.meshgrid(np.linspace(np.min(x),np.max(x),1000),np.linspace(np.min(y),np.max(y),1000))
z = (-r_plane[0] * x_f - r_plane[1] * y_f - r_plane[3]) / r_plane[2]

# Ploting the points and corresponding plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(A[:,0], A[:,1], A[:,2], c='r', marker='o')
ax.scatter(r_points[:,0], r_points[:,1], r_points[:,2], c='g', marker='o')

ax.plot_surface(x_f, y_f, z,alpha= 0.2, Color='g', cmap=cm.twilight)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
