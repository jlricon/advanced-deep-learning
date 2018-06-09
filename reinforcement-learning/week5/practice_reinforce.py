
# coding: utf-8

# # REINFORCE in TensorFlow
# 
# This notebook implements a basic reinforce algorithm a.k.a. policy gradient for CartPole env.
# 
# It has been deliberately written to be as simple and human-readable.
# 

# In[3]:

from IPython.display import FileLink, FileLinks
FileLinks('.') #lists all downloadable files on server


# In[1]:

get_ipython().system('ls')


# The notebook assumes that you have [openai gym](https://github.com/openai/gym) installed.
# 
# In case you're running on a server, [use xvfb](https://github.com/openai/gym#rendering-on-a-server)

# In[1]:

import gym
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

env = gym.make("CartPole-v0")

#gym compatibility: unwrap TimeLimit
if hasattr(env,'env'):
    env=env.env

env.reset()
n_actions = env.action_space.n
state_dim = env.observation_space.shape

plt.imshow(env.render("rgb_array"))


# # Building the policy network

# For REINFORCE algorithm, we'll need a model that predicts action probabilities given states.
# 
# For numerical stability, please __do not include the softmax layer into your network architecture__. 
# 
# We'll use softmax or log-softmax where appropriate.

# In[2]:

import tensorflow as tf

#create input variables. We only need <s,a,R> for REINFORCE
states = tf.placeholder('float32',(None,)+state_dim,name="states")
actions = tf.placeholder('int32',name="action_ids")
cumulative_rewards = tf.placeholder('float32', name="cumulative_returns")


# In[3]:


<define network graph using raw tf or any deep learning library>

logits = <linear outputs (symbolic) of your network>

policy = tf.nn.softmax(logits)
log_policy = tf.nn.log_softmax(logits)


# In[4]:

#utility function to pick action in one given state
get_action_proba = lambda s: policy.eval({states:[s]})[0] 


# #### Loss function and updates
# 
# We now need to define objective and update over policy gradient.
# 
# Our objective function is
# 
# $$ J \approx  { 1 \over N } \sum  _{s_i,a_i} \pi_\theta (a_i | s_i) \cdot G(s_i,a_i) $$
# 
# 
# Following the REINFORCE algorithm, we can define our objective as follows: 
# 
# $$ \hat J \approx { 1 \over N } \sum  _{s_i,a_i} log \pi_\theta (a_i | s_i) \cdot G(s_i,a_i) $$
# 
# When you compute gradient of that function over network weights $ \theta $, it will become exactly the policy gradient.
# 

# In[ ]:

#get probabilities for parti
indices = tf.stack([tf.range(tf.shape(log_policy)[0]),actions],axis=-1)
log_policy_for_actions = tf.gather_nd(log_policy,indices)


# In[ ]:

# policy objective as in the last formula. please use mean, not sum.
# note: you need to use log_policy_for_actions to get log probabilities for actions taken

J = <YOUR CODE


# In[6]:

#regularize with entropy
entropy = <compute entropy. Don't forget the sign!>


# In[7]:

#all network weights
all_weights = <a list of all trainable weights in your network>

#weight updates. maximizing J is same as minimizing -J. Adding negative entropy.
loss = -J -0.1 * entropy

update = tf.train.AdamOptimizer().minimize(loss,var_list=all_weights)


# ### Computing cumulative rewards

# In[8]:

def get_cumulative_rewards(rewards, #rewards at each step
                           gamma = 0.99 #discount for reward
                           ):
    """
    take a list of immediate rewards r(s,a) for the whole session 
    compute cumulative rewards R(s,a) (a.k.a. G(s,a) in Sutton '16)
    R_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    
    The simple way to compute cumulative rewards is to iterate from last to first time tick
    and compute R_t = r_t + gamma*R_{t+1} recurrently
    
    You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
    """
    
    <your code here>
        
    return <array of cumulative rewards>
    
    


# In[9]:

assert len(get_cumulative_rewards(range(100))) == 100
assert np.allclose(get_cumulative_rewards([0,0,1,0,0,1,0],gamma=0.9),[1.40049, 1.5561, 1.729, 0.81, 0.9, 1.0, 0.0])
assert np.allclose(get_cumulative_rewards([0,0,1,-2,3,-4,0],gamma=0.5), [0.0625, 0.125, 0.25, -1.5, 1.0, -4.0, 0.0])
assert np.allclose(get_cumulative_rewards([0,0,1,2,3,4,0],gamma=0), [0, 0, 1, 2, 3, 4, 0])
print("looks good!")


# In[10]:

def train_step(_states,_actions,_rewards):
    """given full session, trains agent with policy gradient"""
    _cumulative_rewards = get_cumulative_rewards(_rewards)
    update.run({states:_states,actions:_actions,cumulative_rewards:_cumulative_rewards})


# ### Playing the game

# In[11]:

def generate_session(t_max=1000):
    """play env with REINFORCE agent and train at the session end"""
    
    #arrays to record session
    states,actions,rewards = [],[],[]
    
    s = env.reset()
    
    for t in range(t_max):
        
        #action probabilities array aka pi(a|s)
        action_probas = get_action_proba(s)
        
        a = <pick random action using action_probas>
        
        new_s,r,done,info = env.step(a)
        
        #record session history to train later
        states.append(s)
        actions.append(a)
        rewards.append(r)
        
        s = new_s
        if done: break
            
    train_step(states,actions,rewards)
            
    return sum(rewards)
        


# In[12]:

s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())

for i in range(100):
    
    rewards = [generate_session() for _ in range(100)] #generate new sessions
    
    print ("mean reward:%.3f"%(np.mean(rewards)))

    if np.mean(rewards) > 300:
        print ("You Win!")
        break
        


# ### Results & video

# In[13]:

#record sessions
import gym.wrappers
env = gym.wrappers.Monitor(gym.make("CartPole-v0"),directory="videos",force=True)
sessions = [generate_session() for _ in range(100)]
env.close()


# In[14]:

#show video
from IPython.display import HTML
import os

video_names = list(filter(lambda s:s.endswith(".mp4"),os.listdir("./videos/")))

HTML("""
<video width="640" height="480" controls>
  <source src="{}" type="video/mp4">
</video>
""".format("./videos/"+video_names[-1])) #this may or may not be _last_ video. Try other indices


# In[ ]:

from submit import submit_cartpole
submit_cartpole(generate_session, <EMAIL>, <TOKEN>)


# In[ ]:

# That's all, thank you for your attention!
# Not having enough? There's an actor-critic waiting for you in the honor section.
# But make sure you've seen the videos first.

