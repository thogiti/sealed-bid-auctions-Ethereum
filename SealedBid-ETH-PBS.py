#!/usr/bin/env python
# coding: utf-8

# ### This document is a notebook of experiments and simulations to study the effect of sealed bid auctions in PBS mechanism. We will study specifically the effects of block bidding strategies by the builders and the block payments distributions to the proposer in the context of the in-protocol (ePBS) and out-protocol (MEV-Boost). We also see if the first price sealed bid (FPSB) auction is always the best strategy to maximize the propoer payments or if there are situations where second price sealed bid (SPSB) auctions are better for maximizing the proposer payments.   

# # First Price Sealed Bid auctions in Ethereum in/out Protocol PBS

# In[1]:


import numpy as np 
import pandas as pd 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy import interpolate

# ensuring this notebook generates the same answers 
np.random.seed(1337)


# ## Revenue Equiavalence Theorem
# 
# From the Revenue Equiavalence Theorem, the expected revenue for the seller (proposer) is the same in either firrst or second-price sealed bid auction. 
# 
# The average payments will be the same in FPSB and SPSB, but we are interested in studying the distribution of payments and the bidding stratgies against the valuations. This can help us show to how to price a bid given a value of the block.

# ## Expectation of the 2nd order statistic
# 
# Our procedure is the following: 
# 
# * Input: $\bar{v}$: the lower truncation value, $R$: the # of draws. 
# 1. Draw $v_{r}$ for draws $r = 1,...,R$. 
# 2. Subset, choosing $\mathcal{R} \equiv \{ r \in \{1,...,R\} \vert v_{r} \le \bar{v} \}$
# 3. Reshape dataset to $v_{i,r}$ for $i = 1,...,N$ and $r = 1,..., \tilde{R}$. 
#     * *Note:* We will have to throw away some of the very last observations to have a square matrix. Specifically, we throw away the last mod$(\vert \mathcal{R} \vert, N)$ simulated draws. 
# 4. For each simulation, $r \in \mathcal{R}$, find the 2nd largest value, $v_{(n-1),r}$
# 5. Return $\hat{\mathbb{E}}(v_{(n-1)}) = R^{-1} \sum_{r=1}^R v_{(n-1),r}$. 

# In[2]:


def Ev_largest(vi, v_sim_untruncated, N, R_used_min=42): 
    '''Ev_largest: compute the expected value of maximum drawing from a truncated distribtion 
                    where v_sim_untruncated are draws from the untruncated and vi is the 
                    truncation point. 
        
        INPUTS: 
            vi: (scalar) upper truncation point 
            v_sim_untruncated: (R-length 1-dim np array) draws of v from the untruncated distribution
            N: (int) number of draws per iteration  
            R_used_min: (int, optional) assert that we have at least this many samples. (Set =0 to disable.)
            
        OUTPUTS
            Ev: (float) expected value of the largest across simulations
            R_used: (int) no. replications used to compute simulated expectation
    '''
    assert v_sim_untruncated.ndim == 1, f'Expected 1-dimensional array'
    
    # perform truncation 
    I = v_sim_untruncated <= vi
    v_sim = np.copy(v_sim_untruncated[I])

    # drop extra rows
    drop_this_many = np.mod(v_sim.size, N)
    if drop_this_many>0: 
        v_sim = v_sim[:-drop_this_many]
    
    # reshape
    R_used = int(v_sim.size / N)
    v_sim = np.reshape(v_sim, (N,R_used))
    assert R_used > R_used_min, f'Too few replications included: only {R_used}. Try increasing original R.'
    
    # find largest value 
    v_sim = np.sort(v_sim, 0) # sorts ascending ... 
    v_largest = v_sim[-1, :]  # ... so the last *row* is the maximum in columns (samples)
    
    # evaluate expectation
    Ev = np.mean(v_largest)
    
    return Ev


# In[3]:


N = 10
R = 100000


# ## Uniform Distribution Simulations

# In[4]:


np.random.seed(1337)
# Generate random valuations for builders
v = np.random.uniform(0,1,(N,R))


# In[5]:


# Bayesian Nash Equilibrium (BNE) in first-price sealed bid
b_star = lambda vi,N: (N-1)/N * vi
b = b_star(v,N)


# ### Get the highest and 2nd highest bid

# In[6]:


# Sorting and indexing
idx = np.argsort(v, axis=0)
v = np.take_along_axis(v, idx, axis=0)
b = np.take_along_axis(b, idx, axis=0)

ii = np.repeat(np.arange(1,N+1)[:,None], R, axis=1)
ii = np.take_along_axis(ii, idx, axis=0)

winning_player = ii[-1,:]

winner_pays_fpsb = b[-1, :] # highest bid 
winner_pays_spsb = v[-2, :] # 2nd-highest valuation


# ### Distribution of payments (Uniform Distribution)
# 
# We know from the **Revenue Equivalence Theorem** that the *average* payment should be identical. However, the distribution of payments may *look* different. That is, the variance, median, etc. of the winning payments may be different. 

# In[9]:


# Distribution of payments
fig = make_subplots(rows=1, cols=1)
for x, lab in zip([winner_pays_fpsb, winner_pays_spsb], ['FPSB', 'SPSB']):
    print(f'Avg. payment {lab} (Uniform Distribution): {x.mean(): 8.5f} (std.dev. = {np.sqrt(x.var()): 5.2f})')
    fig.add_trace(go.Histogram(x=x, histnorm='probability density', name=lab, nbinsx=100), row=1, col=1)

    # Add vertical lines for means
fig.add_vline(x=winner_pays_fpsb.mean(), line_width=2, line_dash="dash", line_color="blue")
fig.add_vline(x=winner_pays_spsb.mean(), line_width=2, line_dash="dash", line_color="red")

fig.update_layout(title_text='Distribution of Payments (Uniform Distribution)', xaxis_title='Bid b_i', yaxis_title='Density', barmode='overlay', title_x=0.5, legend=dict(yanchor="top",y=0,xanchor="left",x=0.9))
fig.update_traces(opacity=0.6)
fig.show()


# Let's now plot the *winning bids* $b_{(n)}$ (i.e. the payments) against valuations, $v_{(n)}$ for FPSB and SPSB respectively. Here, we may note that 
# * FPSB: there is a unique bid corresponding to each valuation, 
# * SPSB: What the winner pays varies even holding fixed the winner's valuation (because it is equal to the valuation of the second-highest type). 

# In[10]:


# Plotting winning bids against valuations
binned = stats.binned_statistic(v[-1, :], v[-2, :], statistic='mean', bins=20)
xx = binned.bin_edges
xx = [(xx[i]+xx[i+1])/2 for i in range(len(xx)-1)]
yy = binned.statistic


# In[11]:


v = v.flatten()
vgrid = np.linspace(0, 1, 10, endpoint=False)[1:]
Ev = np.empty((vgrid.size,))
bds = np.empty((vgrid.size,))
t_v = np.empty((vgrid.size,))

for i,this_v in enumerate(vgrid): 
    Ev[i] = Ev_largest(this_v, v, N-1) 
    bds[i] = Ev[i]/this_v
    t_v[i] = this_v
v = v.reshape((N,R))


# In[15]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=xx, y=yy, mode='lines', name='SPSB avg. payment'))
fig.add_trace(go.Scatter(x=v[-1, :], y=b[-1, :], mode='lines', name='FPSB analytic', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=vgrid, y=Ev, mode='lines', name='FPSB Numerical'))
fig.add_trace(go.Scatter(x=v[-1, :], y=v[-2, :], mode='markers', name='SPSB: actual bids', opacity=0.3, marker=dict(size=3)))
fig.update_layout(title='Winning Bids vs Valuations (Uniform Distribution)', xaxis_title='Valuation, v_i', yaxis_title='Bid, b_i', title_x=0.5, legend=dict(yanchor="top",y=-0.3,xanchor="left",x=0.2,  orientation="h"))
fig.show()


# In[13]:


# Plot Bid shading as a % of value
fig = go.Figure()
fig.add_trace(go.Scatter(x=t_v, y=bds, mode='lines', name='FPSB Bid shading'))
fig.update_layout(title='FPSB optimal bid as a % of valuation (Uniform Distribution)', xaxis_title='Valuation, v_i', yaxis_title='Optimal Bid as a % of valudation', title_x=0.5)
fig.show()


# ## $\chi^2$ distribution Simulations
# 
# Here, it is somewhat harder to solve for the optimal bidding in FPSB because there is no analytical solution. We will approximate it by numerical simulations.

# In[16]:


np.random.seed(1337)
v = np.random.chisquare(df=2, size=(N*R,))


# In[17]:


fig = go.Figure()
fig.add_trace(go.Histogram(x=v, nbinsx=100, name='Values: $v$', opacity=0.6))
fig.update_layout(title='Chi-squared Distributed Values', xaxis_title='Values: v_i', yaxis_title='Count')
fig.show()


# In[18]:


# Generate the values
np.random.seed(1337)
v = np.random.chisquare(df=2, size=(N*R,)) 

# generate a grid
ngrid = 100
pcts = np.linspace(0, 100, ngrid, endpoint=False)[1:]
vgrid = np.percentile(v, q=pcts)


# This plot shows the grid over which we will evaluate our numerical solution. We will then be interpolating linearly between those dots should we require to when we later *evaluate* our solution

# In[19]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=vgrid, y=stats.distributions.chi2.pdf(vgrid + [v.max()], 2), mode='lines', name='True pdf'))
fig.add_trace(go.Scatter(x=vgrid, y=stats.distributions.chi2.pdf(vgrid + [v.max()], 2), mode='markers', name='Grid points'))
fig.update_layout(title='Block Valuations by the Builders', xaxis_title='Valuation, v_i (Chi-Squared distributed)', title_x=0.5)
fig.show()


# In[20]:


# Expected values on the conditional 
Ev = np.empty(vgrid.shape)
for i,this_v in enumerate(vgrid): 
    Ev[i] = Ev_largest(this_v, v, N-1)
    
# by construction / assumption, the lowest-valued bidder always pays zero
Ev    = np.insert(Ev, 0, 0.0)
vgrid = np.insert(vgrid, 0, 0)

# set up interpolation function for the solution
b_star_num = interpolate.interp1d(vgrid, Ev, fill_value='extrapolate')


# In[21]:


# Create a finer grid for interpolation
pcts = np.linspace(0, 100, 1000, endpoint=False)
vgrid_fine = np.percentile(v, q=pcts)


# In[22]:


# Plot the optimal bid in FPSB
fig = go.Figure()
fig.add_trace(go.Scatter(x=vgrid, y=Ev, mode='markers', marker=dict(color='red', symbol='circle'), name='Solution on grid'))
fig.add_trace(go.Scatter(x=vgrid_fine, y=b_star_num(vgrid_fine), mode='lines', name='Interpolated solution'))
fig.update_layout(title='Optimal Bid in a FPSB - Chi-Squared Distribution', xaxis=dict(title='Valuation: v_i'), yaxis=dict(title='Optimal bid in a FPSB: E(v_{(n-1)} | v{(n)} = v)'),legend=dict(x=0.8, y=0.1),title_x = 0.5)
fig.show()


# In[23]:


v = v.reshape((N,R))


# In[24]:


b = b_star_num(v)


# In[25]:


idx = np.argsort(v, axis=0)
v = np.take_along_axis(v, idx, axis=0) # same as np.sort(v, axis=0), except now we retian idx 
b = np.take_along_axis(b, idx, axis=0)

ii = np.repeat(np.arange(1,N+1)[:,None], R, axis=1)
ii = np.take_along_axis(ii, idx, axis=0)

winning_player = ii[-1,:]

winner_pays_fpsb = b[-1, :] # highest bid 
winner_pays_spsb = v[-2, :] # 2nd-highest valuation


# In[27]:


fig = make_subplots(rows=1, cols=1)
for x, lab in zip([winner_pays_fpsb, winner_pays_spsb], ['FPSB', 'SPSB']):
    print(f'Avg. payment {lab} for Chi-Squared distribution: {x.mean(): 8.5f} (std.dev. = {np.sqrt(x.var()): 5.2f})')
    fig.add_trace(go.Histogram(x=x, histnorm='probability density', name=f'{lab} (Chi-Squared distribution)', nbinsx=100), row=1, col=1)

# Add vertical lines for means
fig.add_vline(x=winner_pays_fpsb.mean(), line_width=2, line_dash="dash", line_color="blue", annotation_text="FPSB Mean")
fig.add_vline(x=winner_pays_spsb.mean(), line_width=2, line_dash="dash", line_color="red", annotation_text="SPSB Mean")
fig.update_layout(title_text=f'Distribution of Payments (Chi-Squared distribution)', xaxis_title='Bid', yaxis_title='Density', barmode='overlay', title_x = 0.5, legend=dict(yanchor="top",y=-0.3,xanchor="left",x=0.2,  orientation="h"))
fig.update_traces(opacity=0.6)
fig.show()


# In[28]:


binned = stats.binned_statistic(v[-1, :], v[-2, :], statistic='mean', bins=20)
xx = binned.bin_edges
xx = [(xx[i]+xx[i+1])/2 for i in range(len(xx)-1)]
yy = binned.statistic


# In[30]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=xx, y=yy, mode='lines', name=f'SPSB avg. payment'))
fig.add_trace(go.Scatter(x=v[-1, :], y=b[-1, :], mode='lines', name=f'FPSB: Bids', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=v[-1, :], y=v[-2, :], mode='markers', name=f'SPSB: actual bids', opacity=0.3, marker=dict(size=2)))
fig.update_layout(title=f'Winning Bids vs Valuations (Chi-Squared distribution)', xaxis_title='Valuation, v_i', yaxis_title='Bid, b_i', title_x = 0.5, legend=dict(yanchor="top",y=-0.3,xanchor="left",x=0.2,  orientation="h"))
fig.show()


# In[31]:


# Plot Bid shading as a % of value
xx = np.linspace(0.1, 10, 10)
yy = np.array([b_star_num(v)/v*100.0 for v in vgrid_fine])
fig = go.Figure()
fig.add_trace(go.Scatter(x=vgrid_fine, y=yy, marker=dict(color='red', symbol='circle'), name='Bid Shading'))
fig.update_layout(title='FPSB bid as a % of valuation - Chi-Squared Distribution', xaxis=dict(title='Valuation: v_i'), yaxis=dict(title='Optimal bid as a % of v_i'),legend=dict(x=0.8, y=0.1),title_x = 0.5)
fig.show()


# In[ ]:




