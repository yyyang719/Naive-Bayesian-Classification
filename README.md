# Naive-Bayesian-Classification
I was based on the following assumptions to solve this classification problem(bird or aircraft): 

1. since we are based on the naive bayesian model, observations for the given common cause
are assumed to be conditional independent.

2. The algorithm I used recursively did prediction with transition model from the last state, 
then update the prediction based on the likelihood of the current observations.

3. For our classification problem, the object is either bird or aircraft, so our state has two values.

4. The prior: we assume that at the state S0, the probability for bird and aircraft are both 0.5.

5. The transition model(transition probability): P(bird|bird) = 0.9, and P(aircraft|aircraft) = 0.9

6. The sensor model(emission probability): the pdf file gives the probability of being bird and 
being aircraft for each measured velocity. And the generated v variance pdf gives 
the probability of being bird and being aircraft for each measured velocity variance

7. For the "Nan" in the data file, I replaced it with the average value of its three previous values.

8. Since we cannot correctly classify the total 10 data sets only based on the observation of velocity data, 
we need extra observation to help classify. I chose the distribution of velocity variance between each 
velocity neighbor as this extra observation. we have 10x300 velocity data, so we got 10x299 velocity 
variance data, the max variance is around 57, I used bins: [0,1), [1,2), ..., [56,57) to generate histogram 
image of variance for bird and aircraft. The value for each histogram is the number of variance located in 
its corresponding bin, then I normalized it to make sure the sum of the total histogram value under each 
pdf curve (bird, aircraft) is one.
 
