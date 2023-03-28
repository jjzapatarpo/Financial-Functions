# PACKAGES

import math
from math import factorial
import numpy as np
import pandas as pd
from scipy.stats import norm    # CDF = norm.cdf 
import plotly.graph_objs as go
import plotly.offline as po
import plotly.figure_factory as ff

# Updated: March 28, 2023
# News when updated: two new functions to graph option strategies with a given greek 

# EXCEPTIONS
# We create our own exceptions (just for the fun of it), but they all inherit the characteristics of ValueError 
class OptionTypeError(ValueError): pass
class PositionError(ValueError): pass
class GreekTypeError(ValueError): pass

# CLASSES

class Options:
# AS A RULE: After creating an instance of this class (option object), don't change its attributes as an external value assignment (i.e., option_obj.s = new_spot).
# Doing so will not update the respective values of d1 and d2, and every calculation that depends on such values will be wrong.

    def __init__ (self, s, k, vol, t, r, q=0, call_or_put='call', position='long'):
        if call_or_put not in ['call','put']:
            raise OptionTypeError("Choose a valid option type between 'call' and 'put'" )
        if position not in ['long','short']:
            raise PositionError("Choose a valid position for the option between 'long' and 'short'")
        self.s = s
        self.k = k
        self.vol = vol
        self.t = t
        self.r = r
        self.q = q
        self._d1 = (math.log(s/k)+(r+(vol**2)/2)*t)/(vol*math.sqrt(t))
        self._d2 = self._d1 - vol*math.sqrt(t)
        self.call_or_put = call_or_put
        self.position = position
    
    def __eq__(self, other):
        output = (self.s == other.s) and (self.k == other.k) and (self.vol == other.vol) and (self.t == other.t) and (self.r == other.r) and (self.q == other.q) and (self.call_or_put == other.call_or_put) and (self.position == other.position) and (type(self) == type(other))
        return output
    
    def __str__(self):
        output = f"""
        European {self.call_or_put} option
        Position: {self.position}
            Spot price: {self.s}
            Strike price: {self.k}
            Implied volatility: {self.vol*100}%
            Time to maturity (years): {self.t}
            Risk-free rate: {self.r*100}%
            Dividend rate (foreign rate): {self.q*100}%

            Price: {np.round(self.bsm_valuation(), 4)}

            GREEKS
                Delta: {np.round(self.delta(), 4)}
                Gamma: {np.round(self.gamma(), 4)}
                Theta: {np.round(self.theta(), 4)} (Trading days)
                Vega: {np.round(self.vega(), 4)}
                Rho: {np.round(self.rho(), 4)}
        """
        return output

    def __repr__(self):
        output = f"Options(s={self.s}, k={self.k}, vol={self.vol}, t={self.t}, r={self.r}, q={self.q}, call_or_put='{self.call_or_put}', position='{self.position}')"
        return output
    
    @property
    def d1(self):
        return self._d1
    @property
    def d2(self):
        return self._d2
    
    def _n_inv(self, x):
        n = (1/math.sqrt(2*math.pi))*math.exp(-(x**2)/2)
        return n
    
    def bsm_valuation(self):
        call_price = self.s*math.exp(-self.q*self.t)*norm.cdf(self._d1) - self.k*math.exp(-self.r*self.t)*norm.cdf(self._d2)
        put_price = self.k*math.exp(-self.r*self.t)*norm.cdf(-self._d2) - self.s*math.exp(-self.q*self.t)*norm.cdf(-self._d1)
        if self.call_or_put == 'call':
            output = call_price
        elif self.call_or_put == 'put':
            output = put_price
        return output
    
    def delta(self):
        delta_call = math.exp(-self.q*self.t)*(norm.cdf(self._d1))
        delta_put = math.exp(-self.q*self.t)*(norm.cdf(self._d1)-1)
        if self.call_or_put == 'call':
            output = delta_call
        elif self.call_or_put == 'put':
            output = delta_put
        if self.position == 'short':
            output = -output
        return output
    
    def theta(self):
        theta_call = -((self.s*self._n_inv(self._d1)*self.vol*math.exp(-self.q*self.t))/(2*math.sqrt(self.t))) + self.q*self.s*norm.cdf(self._d1)*math.exp(-self.q*self.t) - self.r*self.k*math.exp(-self.r*self.t)*norm.cdf(self._d2)
        theta_put = -((self.s*self._n_inv(self._d1)*self.vol)/(2*math.sqrt(self.t))) - self.q*self.s*norm.cdf(-self._d1)*math.exp(-self.q*self.t) + self.r*self.k*math.exp(-self.r*self.t)*norm.cdf(-self._d2)
        if self.call_or_put == 'call':
            output = theta_call
        elif self.call_or_put == 'put':
            output = theta_put
        if self.position == 'short':
            output = -output
        return output

    def gamma(self):
        gamma = (self._n_inv(self._d1)*math.exp(-self.q*self.t))/(self.s*self.vol*math.sqrt(self.t))
        if self.position == 'short':
            gamma = -gamma
        return gamma

    def vega(self):
        vega = self.s*math.sqrt(self.t)*self._n_inv(self._d1)*math.exp(-self.q*self.t)
        if self.position == 'short':
            vega = -vega
        return vega

    def rho(self):
        rho_call = self.k*self.t*math.exp(-self.r*self.t)*norm.cdf(self._d2)
        rho_put = -self.k*self.t*math.exp(-self.r*self.t)*norm.cdf(-self._d2)
        if self.call_or_put == 'call':
            output = rho_call
        elif self.call_or_put == 'put':
            output = rho_put
        if self.position == 'short':
            output = -output
        return output
    
    def rho_foreign(self):
        rho_call = -self.t*math.exp(-self.q*self.t)*self.s*norm.cdf(self._d1) 
        rho_put = self.t*math.exp(-self.q*self.t)*self.s*norm.cdf(-self._d1)
        if self.call_or_put == 'call':
            output = rho_call
        elif self.call_or_put == 'put':
            output = rho_put
        if self.position == 'short':
            output = -output
        return output

    def greeks(self, decimals=4, foreign=False):
        if foreign==False:
            output = {
                'Delta':np.round(self.delta(), decimals),
                'Theta':np.round(self.theta(), decimals),
                'Gamma':np.round(self.gamma(), decimals),
                'Vega':np.round(self.vega(), decimals),
                'Rho':np.round(self.rho(), decimals)
            }
        else:
            output = {
                'Delta':np.round(self.delta(), decimals),
                'Theta':np.round(self.theta(), decimals),
                'Gamma':np.round(self.gamma(), decimals),
                'Vega':np.round(self.vega(), decimals),
                'Rho_domestic':np.round(self.rho(), decimals),
                'Rho_foreign':np.round(self.rho_foreign(), decimals)
            }
        return output
    
# FUNCTIONS

def option_strategy(list_of_options, range_width=0.3):
    # Range creation
    spot = list_of_options[0].s
    start = spot*(1-range_width)
    stop = spot*(1+range_width)
    step = (stop-start)/10000
    price_range = np.arange(start=start, stop=stop, step=step)
    # DataFrame to be filled
    df = pd.DataFrame(price_range).rename(columns={0:'price_range'})
    for option in list_of_options:
        # Parameters for payoff
        k = option.k
        price = option.bsm_valuation()
        position = option.position
        call_or_put = option.call_or_put
        # Payoff for long positions
        if (position=='long')&(call_or_put=='call'):
            payoff = [np.max([i-k,0])-price for i in price_range]
            title = 'Long Call'
        elif (position=='long')&(call_or_put=='put'):
            payoff = [np.max([k-i,0])-price for i in price_range]
            title = 'Long Put'
        # Payoff for short positions
        elif (position=='short')&(call_or_put=='call'):
            payoff = [price-np.max([i-k,0]) for i in price_range]
            title = 'Short Call'
        elif (position=='short')&(call_or_put=='put'):
            payoff = [price-np.max([k-i,0]) for i in price_range]
            title = 'Short Put'
        df[f'{title}'] = payoff
    df.set_index('price_range', inplace=True)
    df['Strategy'] = df.sum(axis=1)
    return df

def option_strategy_graph(df, plot_title=''):
    colors = [
        'RGB(39, 24, 126)', 
        'RGB(117, 139, 253)', 
        'RGB(174, 184, 254)',  
        'RGB(71, 181, 255)',
        'RGB(6, 40, 61)', 
        'RGB(37, 109, 133)',
    ]
    traces = []
    trace = go.Scatter(
        x = df.index,
        y = np.zeros(len(df)),
        mode = 'lines',
        name = 'Zero line',
        line = dict(width=1, dash='dash', color='black'),
        opacity=0.5
    )
    traces.append(trace)
    for x,i in enumerate(df.columns):
        if i == 'Strategy':
            trace = go.Scatter(
                x = df.index,
                y = df.loc[:,i],
                mode = 'lines',
                name = i,
                line = dict(width=3, color='RGB(255, 134, 0)'),
                fill='tozeroy',
                fillcolor='RGBA(255, 134, 0, 0.1)'
            )
            traces.append(trace)
        else:
            trace = go.Scatter(
                x = df.index,
                y = df.loc[:,i],
                mode = 'lines',
                name = i,
                line = dict(width=1.2, color=colors[x]),
            )
            traces.append(trace)
    layout = go.Layout(showlegend=True,
        legend={'x':1,'y':1},
        width=900,
        height=500,
        margin=dict(l=50,r=50,b=50,t=50),
        template='plotly_white',
        yaxis={'tickfont':{'size':10}, 'title':'Payoff'},
        xaxis={'tickfont':{'size':10}, 'title':'Range of Spot prices'},
        hovermode='x unified',
        title={'text':f'<b>Option Strategy</b> - {plot_title}','xanchor':'left'},
        titlefont={'size':14}
    )
    po.init_notebook_mode(connected=True)
    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(yaxis_tickprefix = '$', yaxis_tickformat = ',.2f')
    fig.update_layout(xaxis_tickprefix = '$', yaxis_tickformat = ',.2f')
    po.iplot(fig)

def option_strategy_withGreek(list_of_options, greek=None, range_width=0.3):
    if greek != None:
        if greek not in ['delta','gamma','theta','vega','rho','rho_foreign']:
            raise GreekTypeError("The greek to graph must be in ['delta','gamma','theta','vega','rho','rho_foreign']")
    # Range creation
    spot = list_of_options[0].s
    start = spot*(1-range_width)
    stop = spot*(1+range_width)
    step = (stop-start)/10000
    price_range = np.arange(start=start, stop=stop, step=step)
    # DataFrame to be filled
    df_options = pd.DataFrame(price_range).rename(columns={0:'price_range'})
    if greek == None:
        for option in list_of_options:    
            # Parameters for payoff
            k = option.k
            price = option.bsm_valuation()
            position = option.position
            call_or_put = option.call_or_put
            # Payoff for long positions
            if (position=='long')&(call_or_put=='call'):
                payoff = [np.max([i-k,0])-price for i in price_range]
                title = 'Long Call'
            elif (position=='long')&(call_or_put=='put'):
                payoff = [np.max([k-i,0])-price for i in price_range]
                title = 'Long Put'
            # Payoff for short positions
            elif (position=='short')&(call_or_put=='call'):
                payoff = [price-np.max([i-k,0]) for i in price_range]
                title = 'Short Call'
            elif (position=='short')&(call_or_put=='put'):
                payoff = [price-np.max([k-i,0]) for i in price_range]
                title = 'Short Put'
            if title in df_options.columns:
                sufx = str(df_options.columns.tolist().count(title)+1)
                title = title+' '+sufx
            df_options[f'{title}'] = payoff
        df = df_options.copy()
        df.set_index('price_range', inplace=True)
        df['Strategy'] = df.sum(axis=1)
        title_gk = None
    else:
        # We create titles for the auxiliar Greeks DataFrame
        if greek=='delta':
            title_gk = 'Delta'
        elif greek=='gamma':
            title_gk = 'Gamma'
        elif greek=='theta':
            title_gk = 'Theta'
        elif greek=='vega':
            title_gk = 'Vega'
        elif greek=='rho':
            title_gk = 'Rho'
        elif greek=='rho_foreign':
            title_gk = 'Rho foreign'
        # We create such DataFrame
        df_greek = df_options.copy()
        for x,option in enumerate(list_of_options):
            # Parameters for payoff and greeks
            k = option.k
            vol = option.vol
            t = option.t
            r = option.r
            q = option.q
            position = option.position
            call_or_put = option.call_or_put
            price = option.bsm_valuation()
            # Payoff for long positions
            if (position=='long')&(call_or_put=='call'):
                payoff = [np.max([i-k,0])-price for i in price_range]
                title = 'Long Call'
            elif (position=='long')&(call_or_put=='put'):
                payoff = [np.max([k-i,0])-price for i in price_range]
                title = 'Long Put'
            # Payoff for short positions
            elif (position=='short')&(call_or_put=='call'):
                payoff = [price-np.max([i-k,0]) for i in price_range]
                title = 'Short Call'
            elif (position=='short')&(call_or_put=='put'):
                payoff = [price-np.max([k-i,0]) for i in price_range]
                title = 'Short Put'
            if title in df_options.columns:
                sufx = str(df_options.columns.tolist().count(title)+1)
                title = title+' '+sufx
            df_options[f'{title}'] = payoff
            # Greek calculation
            greek_value = []
            for i in price_range:   
                opt = Options(s=i, k=k, vol=vol, t=t, r=r, q=q, call_or_put=call_or_put, position=position)
                if greek=='delta':
                    greek_value.append(opt.delta())
                elif greek=='gamma':
                    greek_value.append(opt.gamma())
                elif greek=='theta':
                    greek_value.append(opt.theta())
                elif greek=='vega':
                    greek_value.append(opt.vega())
                elif greek=='rho':
                    greek_value.append(opt.rho())
                elif greek=='rho_foreign':
                    greek_value.append(opt.rho_foreign())
            df_greek[f'{title_gk}_{x}'] = greek_value  
        df_options.set_index('price_range', inplace=True)
        df_options['Strategy'] = df_options.sum(axis=1)
        df_greek.set_index('price_range', inplace=True)
        df_greek[f'{title_gk}'] = df_greek.sum(axis=1)
        greek_final_series = df_greek[f'{title_gk}'].values
        df_options[f'{title_gk}'] = greek_final_series
        df = df_options.copy()
    return [df, greek, title_gk]

def option_strategy_graph_withGreek(df, plot_title=''):
    if df[1] != None:
        greek_title = df[2]
        df = df[0]
        colors = [
            'RGB(39, 24, 126)', 
            'RGB(117, 139, 253)', 
            'RGB(174, 184, 254)',  
            'RGB(71, 181, 255)',
            'RGB(6, 40, 61)', 
            'RGB(37, 109, 133)',
        ]
        traces = []
        trace = go.Scatter(
            x = df.index,
            y = np.zeros(len(df)),
            mode = 'lines',
            name = 'Zero line option',
            line = dict(width=1, dash='dash', color='black'),
            opacity=0.7
        )
        traces.append(trace)
        trace = go.Scatter(
            x = df.index,
            y = np.zeros(len(df)),
            yaxis='y2',
            mode = 'lines',
            name = 'Zero line greek',
            line = dict(width=1, dash='dot', color='black'),
            opacity=0.7
        )
        traces.append(trace)
        for x,i in enumerate(df.columns): 
            if i == 'Strategy':
                trace = go.Scatter(
                    x = df.index,
                    y = df.loc[:,i],
                    yaxis='y1',
                    mode = 'lines',
                    name = i,
                    line = dict(width=3, color='RGB(255, 134, 0)'),
                    fill='tozeroy',
                    fillcolor='RGBA(255, 134, 0, 0.1)'
                )
                traces.append(trace)
            elif i in ['Delta','Gamma','Theta','Vega','Rho','Rho foreign']:
                trace = go.Scatter(
                    x = df.index,
                    y = df.loc[:,i],
                    yaxis='y2',
                    mode = 'lines',
                    name = i,
                    line = dict(width=2, color='RGB(137,49,239)'),
                )
                traces.append(trace)
            else:
                trace = go.Scatter(
                    x = df.index,
                    y = df.loc[:,i],
                    yaxis='y1',
                    mode = 'lines',
                    name = i,
                    line = dict(width=1.2, color=colors[x]),
                )
                traces.append(trace)
        layout = go.Layout(showlegend=True,
            legend={'x':1.05,'y':1},
            width=1000,
            height=500,
            margin=dict(l=50,r=100,b=50,t=90),
            template='plotly_white',
            yaxis={'tickfont':{'size':10}, 'title':'Payoff'},
            yaxis2={'overlaying':'y','side':'right','title':f'{greek_title}','showgrid':False},
            xaxis={'tickfont':{'size':10}, 'title':'Range of Spot prices'},
            hovermode='x unified',
            title={'text':f'<b>Option Strategy</b> - {plot_title}','xanchor':'left'},
            titlefont={'size':18}
        )
    else:
        df = df[0]
        colors = [
            'RGB(39, 24, 126)', 
            'RGB(117, 139, 253)', 
            'RGB(174, 184, 254)',  
            'RGB(71, 181, 255)',
            'RGB(6, 40, 61)', 
            'RGB(37, 109, 133)',
        ]
        traces = []
        trace = go.Scatter(
            x = df.index,
            y = np.zeros(len(df)),
            mode = 'lines',
            name = 'Zero line',
            line = dict(width=1, dash='dash', color='black'),
            opacity=0.7
        )
        traces.append(trace)
        for x,i in enumerate(df.columns):
            if i == 'Strategy':
                trace = go.Scatter(
                    x = df.index,
                    y = df.loc[:,i],
                    mode = 'lines',
                    name = i,
                    line = dict(width=3, color='RGB(255, 134, 0)'),
                    fill='tozeroy',
                    fillcolor='RGBA(255, 134, 0, 0.1)'
                )
                traces.append(trace)
            else:
                trace = go.Scatter(
                    x = df.index,
                    y = df.loc[:,i],
                    mode = 'lines',
                    name = i,
                    line = dict(width=1.2, color=colors[x]),
                )
                traces.append(trace)
        layout = go.Layout(showlegend=True,
            legend={'x':1,'y':1},
            width=1000,
            height=500,
            margin=dict(l=50,r=100,b=50,t=90),
            template='plotly_white',
            yaxis={'tickfont':{'size':10}, 'title':'Payoff'},
            xaxis={'tickfont':{'size':10}, 'title':'Range of Spot prices'},
            hovermode='x unified',
            title={'text':f'<b>Option Strategy</b> - {plot_title}','xanchor':'left'},
            titlefont={'size':18}
        )
    po.init_notebook_mode(connected=True)
    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(yaxis_tickprefix = '$', yaxis_tickformat = ',.2f')
    fig.update_layout(xaxis_tickprefix = '$', yaxis_tickformat = ',.2f')
    po.iplot(fig)

def graph_greeks(greek, option_object, range_width=2.5):
    # We extract the attributes of the option
    s = option_object.s
    k = option_object.k
    vol = option_object.vol
    call_or_put = option_object.call_or_put
    t = option_object.t
    r = option_object.r
    q = option_object.q
    position = option_object.position
    # We create the price range for the x axis of the graph
    start = 1
    stop = s*(1+range_width)
    step = (stop-start)/1000
    price_range = np.arange(start=start, stop=stop, step=step)
    # We simulate and store the different values of the given greek within the calculated price range. These are the values to be graphed
    greek_value = []
    for i in price_range:   
        opt = Options(s=i, k=k, vol=vol, t=t, r=r, q=q, call_or_put=call_or_put, position=position)
        if greek=='delta':
            greek_value.append(opt.delta())
            title = 'Delta'
        elif greek=='gamma':
            greek_value.append(opt.gamma())
            title = 'Gamma'
        elif greek=='theta':
            greek_value.append(opt.theta())
            title = 'Theta'
        elif greek=='vega':
            greek_value.append(opt.vega())
            title = 'Vega'
        elif greek=='rho':
            greek_value.append(opt.rho())
            title = 'Rho'
        elif greek=='rho_foreign':
            greek_value.append(opt.rho_foreign())
            title = 'Rho foreign'        
    # We create and return the corresponding graph
    fig = go.Figure(data=[go.Scatter(name='Zero line', x=price_range, y=np.zeros(len(price_range)), line={'color':'gray', 'dash':'dash'}),
                          go.Scatter(name=f'{title}', x=price_range, y=greek_value, line={'color':'RGB(39, 24, 126)'})],
                    layout=go.Layout(
                        title=f'<b>{title}</b> for {position} {call_or_put} option',
                        width=900,
                        height=500,
                        template='plotly_white',
                        margin=dict(l=100,r=50,b=80,t=100),
                        yaxis={'title':f'{title} values'},
                        xaxis={'title':'Spot prices'}))
    fig.add_vline(x=k, line_color='red')
    fig.update_layout(xaxis_tickprefix = '$', yaxis_tickformat = ',.2f')
    po.iplot(fig)

def binomial_tree(s, k, t, r, q, vol, n_steps, type_opt='european', call_put='call'):
    # Each step considers the time to maturity divided by the number of steps
    delta_t = t/n_steps
    # We calculate the upward and downward probabilities based on the given volatility
    u = math.exp(vol*math.sqrt(delta_t)) 
    d = 1/u
    a = math.exp((r-q)*delta_t)
    p = (a-d)/(u-d)
    # We build the stock's tree. It will later be merged with the option's tree
    tree = {}
    for i in np.arange(n_steps+1):
        inner_tree = {}
        for j in np.arange(start=0, stop=i+1):
            inner_tree[j] = s*(u**(i-j))*(d**(j))
        tree[i] = inner_tree
    # We create dictionaries that will be used to create the option's tree
    opt_tree = {}
    opt_tree_last = {}
    # We calculate the option's value at the last step. This values will be backpropagated to construct the option's tree
    if call_put=='call':
        for ky,v in tree[n_steps].items():
            poff = np.max([v-k,0])
            opt_tree_last[ky] = poff
    elif call_put=='put':
        for ky,v in tree[n_steps].items():
            poff = np.max([k-v,0])
            opt_tree_last[ky] = poff
    opt_tree[n_steps] = opt_tree_last   # Here, we store the option's payoff of the last step. This values will be backpropagated 
    if type_opt=='american':
        # We backpropagate the option and fill up the option's tree
        for i in list(tree.keys())[::-1][:-1]:
            dict_opt = {}
            for n in np.arange(i):
                fu = opt_tree[i][n]
                fd = opt_tree[i][n+1]
                f = math.exp(-r*delta_t)*(p*fu+(1-p)*fd)
                # Since the option is American and can be exercised before the maturity of the option, we have to check for the real value of the option at any given 
                # time. If the option is more valuable when exercising, the difference between spot and strike prices (according to whether it's a call or put option)
                # then its price will be such a difference. If not, it will be the present value of the expected value of the option for the given probabilities
                if call_put=='call':
                    diff = tree[i-1][n]-k
                elif call_put=='put':
                    diff = k-tree[i-1][n]        
                dict_opt[n] = np.max([f, diff])
            opt_tree[i-1] = dict_opt
    elif type_opt=='european':
        # We backpropagate the option and fill up the option's tree
        for i in list(tree.keys())[::-1][:-1]:
            dict_opt = {}
            for n in np.arange(i):
                fu = opt_tree[i][n]
                fd = opt_tree[i][n+1]
                f = math.exp(-r*delta_t)*(p*fu+(1-p)*fd)
                dict_opt[n] = f
            opt_tree[i-1] = dict_opt
    binom_tree = {}
    for ky,v in tree.items():
        level_dict = {}
        for k1, prices in v.items():
            option_price = opt_tree[ky][k1]
            level_dict[k1] = [np.round(prices,4), np.round(option_price,4)]
        binom_tree[ky] = level_dict
    return {'price':np.round(f,4), 'tree':binom_tree}