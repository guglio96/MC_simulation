import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random


n_points=10
n_steps= 2000
n_part = 500
K=.5
a=3
dt=1.5
cut_off= np.floor_divide(n_steps,3)

k_bT=3
beta=1/k_bT
epsilon= 1


def initial_positions():
    x_coord = [np.random.random()*L for i in range (n_part)]
    y_coord = [np.random.random()*L for i in range (n_part)]
    return x_coord, y_coord



def taking_step_single(x_coord,y_coord):
    random_ptc_order= random.sample(range(0,n_part),n_part)
    for i in range(len(random_ptc_order)):
        j=random_ptc_order[i]
        x_hypotetical= x_coord[j].copy()+dt*(np.random.random()-0.5)
        y_hypotetical = y_coord[j].copy()+dt*(np.random.random()-0.5)
        delta_e = delta_en(x_coord,y_coord,x_hypotetical,y_hypotetical,j)
        x_coord[j],y_coord[j] = step_decision(delta_e,x_coord,y_coord,x_hypotetical,y_hypotetical,j)
      #  if i%20==0:
       #      press_mini[int(i/10)] = pressure(x_coord,y_coord)
    return x_coord,y_coord

def step_decision(delta_e,x_coord,y_coord,x_hypotetical,y_hypotetical,j):

     if delta_e <= 0:
        coord_x, coord_y= x_hypotetical,y_hypotetical
     if delta_e > 0:
        prob=np.exp(-beta*delta_e)
        prob1=np.random.random()
        if prob1 < prob:
             coord_x, coord_y= x_hypotetical,y_hypotetical
        if prob1 >= prob:
            coord_x, coord_y= x_coord[j], y_coord[j]
     return  coord_x, coord_y


def pressure(x_coord,y_coord):
    pot_en_x=np.zeros(n_part)
    pot_en_y=np.zeros(n_part)
   
    bool_mat1x = (x_coord<0).astype(int)
    bool_mat2x= (x_coord>L).astype(int)
    press_x= bool_mat1x*K*np.abs(x_coord)+bool_mat2x*K*np.abs(x_coord-L)
    
    bool_mat1y = (y_coord<0).astype(int)
    bool_mat2y= (y_coord>L).astype(int)
    press_y= bool_mat1y*K*np.abs(x_coord)+bool_mat2y*K*np.abs(x_coord-L)

    press = sum(press_x) + sum(press_y)
    return press



def delta_wall_en1(x_coord,y_coord, x_hyp,y_hyp):
    pot_en_x=0
    pot_en_y=0
   
    bool_1x1 = (x_coord<0).astype(int)
    bool_2x1= (x_coord>L).astype(int)
    pot_en_x1= bool_1x1*K*np.square(x_coord)+bool_2x1*K*np.square(x_coord-L)

    bool_1x2 = (x_hyp<0).astype(int)
    bool_2x2= (x_hyp>L).astype(int)
    pot_en_x2= bool_1x2*K*np.square(x_hyp)+bool_2x2*K*np.square(x_hyp-L)
    
    bool_1y1 = (y_coord<0).astype(int)
    bool_2y1= (y_coord>L).astype(int)
    pot_en_y1= bool_1y1*K*np.square(y_coord)+bool_2y1*K*np.square(y_coord-L)

    bool_1y2 = (y_hyp<0).astype(int)
    bool_2y2= (y_hyp>L).astype(int)
    pot_en_y2= bool_1y2*K*np.square(y_hyp)+bool_2y2*K*np.square(y_hyp-L)

    delta_pot_en_wall = pot_en_x2-pot_en_x1 + pot_en_y2-pot_en_y1
    return delta_pot_en_wall


def delta_en(x_coord,y_coord,x_hypotetical,y_hypotetical,j):
    delta_pot_en_wall= delta_wall_en1(x_coord[j],y_coord[j],x_hypotetical,y_hypotetical)
    delta_inter_ptc_pot = lennard1(x_coord,y_coord,x_hypotetical,y_hypotetical,j)
    pot_en = delta_pot_en_wall + delta_inter_ptc_pot
    return pot_en


def hardcore1(x_coord,y_coord,x_hypotetical,y_hypotetical,j):
    hardcore_pot_matrix= np.zeros(n_part)
    accepted_dist = one_ptc_distance(x_coord,y_coord,x_coord[j],y_coord[j])
    accepted_dist[j]=accepted_dist[j]
    new_dist = one_ptc_distance(x_coord,y_coord,x_hypotetical,y_hypotetical)
    new_dist[j]=new_dist[j]+10e20
    acc_bool_mat = (accepted_dist<a)
    acc_bool_mat[j] = "False"
    new_bool_mat = (new_dist<a)
    new_bool_mat[j] = "False"

    hardcore_pot_matrix= (new_bool_mat.astype(int))*1000*k_bT - (acc_bool_mat.astype(int))*1000*k_bT
    hardcore_pot = sum(hardcore_pot_matrix)
    return hardcore_pot 

def lennard1(x_coord,y_coord,x_hypotetical,y_hypotetical,j):
    accepted_dist = one_ptc_distance(x_coord,y_coord,x_coord[j],y_coord[j])
    accepted_dist[j]=accepted_dist[j]+10e20
    new_dist = one_ptc_distance(x_coord,y_coord,x_hypotetical,y_hypotetical)
    new_dist[j]=new_dist[j]+10e20
    lennard_matrix = -epsilon*(np.power(a/new_dist,6)-2*np.power(a/new_dist,12))+epsilon*(np.power(a/accepted_dist,6)-2*np.power(a/accepted_dist,12))
    lennard_pot= sum(lennard_matrix)
    return lennard_pot


def ptc_distance(x_coord,y_coord):
   dist = np.sqrt(np.square(x_coord-np.vstack(x_coord))+np.square(y_coord-np.vstack(y_coord)))
   return dist
def one_ptc_distance(x_coord,y_coord,x_hypotetical,y_hypotetical):
   dist = np.sqrt(np.square(x_coord-x_hypotetical*np.ones(len(x_coord)))+np.square(y_coord-y_hypotetical* np.ones(len(x_coord))))
   return dist



       
def animation(i):
    if i%10==0:
        x_val=positions[i,0,:]
        y_val=positions[i,1,:]
       
        ax.cla()
        ax.scatter(x_val,y_val)
        ax.axis([0, L,0, L])
        ax.plot(i,press[i])


press_val = np.zeros([n_points])
press_std = np.zeros([n_points])
press_mini = np.zeros(int(n_part/10))
press = np.zeros([n_steps])

for j in range(n_points):
    #fig,ax= plt.subplots()
    fig1,ax1= plt.subplots()
    L=10*np.floor(np.sqrt(n_part))+2*j
    x_coord,y_coord= np.array(initial_positions())
    positions=np.zeros([n_steps,2,n_part])

    for i in range(n_steps):
        x_coord,y_coord = taking_step_single(x_coord,y_coord)
        press[i]=pressure(x_coord,y_coord)
        positions[i]=np.array([x_coord,y_coord])
    press_val[j]= np.average(press[cut_off:])
    press_std[j] = np.std(press[cut_off:])
    #ani=FuncAnimation(fig, animation, interval=1)
    ax1.plot(np.arange(n_steps),press)
    plt.show()
    print('The pressure value for Vol ' , L*L ,' is ', press_val[j], '+/- ', press_std[j], 'Which represents', press_std[j]/press_val[j]*100, '%' )
        
        

p_info=[press_val,press_std]



#ani=FuncAnimation(plt.gcf(), animation1() , interval=1)
plt.tight_layout()
plt.errorbar(2+2*np.arange(n_points),p_info[0],p_info[1])
plt.show()    

