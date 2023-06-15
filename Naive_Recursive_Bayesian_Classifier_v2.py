#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 22:43:12 2022

@author: yuanyuan
"""
# in our classification problem, the object is either bird or aircraft, so our state has two values.
# the prior: we assume that at the state S0, the probability for bird and aircraft are both 0.5.
# the transition model(transition probability): P(bird|bird) = 0.9, and P(aircraft|aircraft) = 0.9
# the sensor model(emission probability): the pdf file gives the probability of being bird
# and being aircraft for each measured velocity. And the generated v variance pdf gives
# the probability of being bird and being aircraft for each measured velocity variance

import numpy as np
import matplotlib.pyplot as plt

class Naive_Bayesian_Classifier:


    def __init__(self, Pi0, data_velocity, P_transition, pdf, var_pdf, var_pdf_beams, vlow, vhigh):
        self.Pi0 = Pi0 # the prior
        self.data_velocity = data_velocity # 10 sets of velocity data flow
        self.P_transition = P_transition # transition probability
        self.pdf = pdf # probability distritutions for our objects (bird and aircraft)
        self.var_pdf = var_pdf # v variance pdf for bird and aircraft
        self.var_pdf_beams = var_pdf_beams
        self.vlow = vlow # the lowest velocity in the velocity pdf picture 
        self.vhigh =  vhigh # the highest velocity in the velocity pdf picture
        self.velocity_beam = []
        self.create_velocity_beam() # create beam list for velocity pdf


    def create_velocity_beam(self):
        # create beam list based on the value of velocity for the pdf
        # the probability distributions for the objects (aircrafts and birds) are over velocity of 0-200kt,
        # we have 400 discrete pdf data, so I will generate a beam list with size 400 over value 0-200.
        # the beam is like [0.5, 1, 1.5, 0, ......, 200], for example, when velocity is between [0,0.5),
        # it belongs to beam 0, when velocity is [0.5,1) it belongs to beam 1. beam is used to search likelihood
        # based on the measured velocity samples.
        beam_size = len(self.pdf[0])
        # v = self.vlow
        beam_step_size = self.vhigh/beam_size
        # i = 0
        self.velocity_beam = [self.vlow + beam_step_size*(i+1) for i in range(beam_size)]
        # while i < beam_size:
        #     v += beam_step_size
        #     self.velocity_beam.append(v)
        #     i += 1


    def likelihood_velocity(self, velocity):
        # the sensor model for our classification problem
        # based on the measured velocity, we got[ likelihood of bird, likelihood of aircraft]
        idx = np.where((velocity < np.array(self.velocity_beam))==True)[0][0]
        L = np.array([self.pdf[0][idx], self.pdf[1][idx]]) # [likelihood for bird, likelihood for aircraft]
    
        return L


    def likelihood_variance(self, variance):
        # this observatiion is based on the velocity variance
        idx = np.where((variance < np.array(self.var_pdf_beams))==True)[0][0]
        L = np.array([self.var_pdf[0][idx], self.var_pdf[1][idx]]) # [likelihood for bird, likelihood for aircraft]
        
        return L


    def normalize(self, vector):
        # normalize vector values to probability
        return vector/np.sum(vector)


    def recursive_bayesian_estimation(self):
        
        # since we are based on the naive bayesian model, the two observations (velocity and v variance)
        # are assumed to be conditional independent.
        # recursively did prediction from the last state and its history observations, 
        # then update with the likelihood of the current observations
        b0 = self.Pi0 # initialize b0 with our prior estimation
        dataflow_set = len(self.data_velocity) # 10 data sets
        P_T = [] # store the last probability estimation after T steps for 10 sets of data
        
        for i in range(dataflow_set):
            bt_1 = np.zeros(2) # initialize b(t-1)
            # start from state 0, when t = 1 we got the first velocity, when t = 2 we got the first v variance
            t = 0 
            while t < len(self.data_velocity[i]): # step t from 1 to T
                
                t += 1
                # prediction from t-1 to t
                if t == 1:
                    # prediction from step 0 to step 1. the estimation of step 0 matrix-multiplicates with
                    # the transition probability, we got the estimation of step 1
                    Predic = np.matmul(b0, self.P_transition)
     
                else:
                    # prediction from step t-1 to step t, we got the estimation of step t based on all its
                    # previous observation (not including the recent velocity at step t)
                    Predic = np.matmul(bt_1, self.P_transition)
     
                # update prediction through our measured velocity at step t
                Lt = self.likelihood_velocity(self.data_velocity[i][t-1])
                P_update = Lt * Predic
                
                # update prediction through our measured v variance at step t
                if t > 1: # we can observe v variance when we got the second velocity
                    var = abs(self.data_velocity[i][t-1] - self.data_velocity[i][t-2])
                    Lt_var = self.likelihood_variance(var)
                    P_update = Lt_var * P_update
     
                bt = self.normalize(P_update)
                bt_1 = bt
     
            P_T.append(bt)
            
        return P_T



def fill_nan(data, window_size = 3):
    # fill nan value with the average of its 3 previous values
    for i in range(len(data)):
        row = data[i]

        # find first not nan value
        first_valid_index = np.where(np.isnan(row)==False)[0][0]
        # fill nan before with this first valid value
        row[:first_valid_index] = row[first_valid_index]

        # make sure first window_size elements are valid
        # no need to check index = 0, since we make sure it is filled above
        for j in range(1,window_size):
            if np.isnan(row[j]):
                row[j] = row[j-1]

        # for rest elements, if nan, we fill with mean of 3 previous values
        for j in range(window_size,len(row)):
            if np.isnan(row[j]):
                row[j] = np.mean(row[j-window_size:j])

        data[i] = row

    return data



def remove_nan(data):
    # remove nan values
    data_list = []
    for i in range(len(data)):
        new_data = data[i][~np.isnan(data[i])]
        data_list.append(new_data)
    
    return data_list


# generate pdf for velocity variance of bird and aircraft
def v_variance_pdf(data_velocity):
    
    # generate velocity variance for each neighbor steps in data flow
    v_variance = np.zeros((data_velocity.shape[0], data_velocity.shape[1]-1)) # initialize 2D array
    for i, row in enumerate(data_velocity):
        delta = [abs(x) for x in row[1:] - row[:-1]] # each neighbor variance
        v_variance[i] = delta
    
    # the known results (This is an experiment part: if we had the velocity dataset for bird/aircraft, how could we generate 
    # velocity variance pdf for them.)
    bird = [0,2,3,4,9]
    aircraft = [1,5,6,7,8]
    
    bird_delta_v = v_variance[bird].reshape((-1))
    aircraft_delta_v = v_variance[aircraft].reshape((-1))
    
    # generate pdf for v variance of bird
    max_variance = np.max(v_variance)
    bins = range(0,round(max_variance)+1)
    var_pdf_beams = [i+1 for i in range(round(max_variance)) ]
    var_pdf = np.zeros((2, len(bins)-1)) # initialize an empty 2D variance pdf for bird and aircraft
    # n is the number of variance in each bins
    n, _, _ = plt.hist(x=bird_delta_v, bins=bins)
    total_num = n.sum()
    # normalize the histogram to pdf
    for i,num in enumerate(n):
        n[i] = num / total_num
    var_pdf[0] = n    # save pdf of bird 
    
    # generate pdf for v variance of aircraft
    m, _, _ = plt.hist(x=aircraft_delta_v, bins=bins)
    total_m = m.sum()
    # normalize the histogram to pdf
    for i,num in enumerate(m):
        m[i] = num / total_m
    var_pdf[1] = m # save pdf of aircraft
    
    return var_pdf, var_pdf_beams



def main():
    # using numpy to load txt files as 2D numpy array
    # 10 rows for 10 sets of data flow
    data_velocity = np.loadtxt('data.txt', delimiter=',')
    data_velocity = fill_nan(data_velocity)
    # data_velocity = remove_nan(data_velocity)
    
    # generate v variance pdf for bird and aircraft
    var_pdf, var_pdf_beams = v_variance_pdf(data_velocity)
    
    # 2 rows in velocity pdf for two objects, the first row is for bird, and the second row is for aircraft
    pdf = np.loadtxt('pdf.txt', delimiter=',')
    # the probability distribution for our velocity is over 0 to 200kt
    vlow = 0
    vhigh = 200

    # the prior estimation Pi0 is an array with 2 size, the Pi0[0] is the probability for bird,
    # and the Pi0[1] is the probability for aircraft.
    Pi0 = np.array([0.5, 0.5])

    # create the transition matrix:[p(bird to bird), p(bird to aircraft)],[p(aircraft to bird), p(aircraft to aircraft)]
    P_transition = np.array([[ 0.9, 0.1],
                             [ 0.1, 0.9]])

    # generate classifier object
    classifier = Naive_Bayesian_Classifier(Pi0, data_velocity, P_transition, pdf, var_pdf, var_pdf_beams, vlow, vhigh)
    # return probability array [for bird, for aircraft] after T steps update based on our observations
    P_T = classifier.recursive_bayesian_estimation()
    for i in range(len(data_velocity)):
        if P_T[i][0] > 0.5:
            print("O" + str(i+1) + " is bird")
        else:
            print("O" + str(i+1) + " is aircraft")



if __name__ == "__main__":
    main()












    
    
    
    
