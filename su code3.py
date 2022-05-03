def format_date(x, pos=None):
    thisind = np.clip(int(startInd + x + 0.5), startInd, startInd + numTimes - 1)
    return num2date(times[thisind]).strftime('%m/%d/%Y')

def validate_model(simulated_cases, cases):
    
    global times, numTimes
    startInd = 0
    times = [date2num(s) for (s) in df.date]
    numTimes = len(simulated_cases)

    fig = plt.figure(figsize=[22, 12], dpi=120)
    fig.subplots_adjust(top=0.85, right=0.92)
    ind = np.arange(numTimes)
    indObs = np.arange(len(simulated_cases))

    ax = fig.add_subplot(111)
    ax.yaxis.grid(True, color='black', linestyle='dashed')
    ax.xaxis.grid(True, color='black', linestyle='dashed')
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    fig.autofmt_xdate()

    (infectedp,) = ax.plot(indObs, simulated_cases, linewidth=3, color='black')
    (si,) = ax.plot(ind, simulated_cases, linewidth=3, color='orange')
    (i,) = ax.plot(ind, cases, linewidth=3, color='blue')
    ax.set_xlim(0, numTimes)
    ax.set_xlabel('Date')
    ax.set_ylabel('Population ratio')

    plt.legend(
        [si, i],
        ['Simulated', 'Actual'],
        loc='upper right',
        bbox_to_anchor=(1, 1.22),
        fancybox=True,
    )





days = len(df.cases)
startInd = 0
cases = results[:days, 1] + results[:days, 2]
validate_model((results[:days, 1] + results[:days, 2]) , (df.cases / N - df.deaths/N))





class OptimizeModelParameters(object):
    '''SEIR'''
    def __init__(self, init_vals, confirmed):
        
        self.init_vals = init_vals
        self.confirmed = confirmed

    def evaluate(self, params):
        
        S_0, E_0, I_0, R_0 = self.init_vals
        S, E, I, R = [S_0], [E_0], [I_0], [R_0]
        epsilon, beta, gamma = params
        dt = 1
        for _ in range(len(self.confirmed) - 1):
            next_S = S[-1] - (beta * S[-1] * I[-1]) * dt
            next_E = E[-1] + (beta * S[-1] * I[-1] - epsilon * E[-1]) * dt
            next_I = I[-1] + (epsilon * E[-1] - gamma * I[-1]) * dt
            next_R = R[-1] + (gamma * I[-1]) * dt
            S.append(next_S)
            E.append(next_E)
            I.append(next_I)
            R.append(next_R)
        return E, I

    def error(self, params):
        '''
    
        params: Epsilon, beta, gamma
        
        
        '''
        yEim, yIim = self.evaluate(params)
        yCim = [sum(i) for i in zip(yEim, yIim)]  
        res = sum(
              np.subtract(yCim, self.confirmed) ** 2
        )
        return res


    def optimize(self, params):
        '''
        
        params: Epsilon, beta, gamma

        
        '''
        res = optimize.minimize(
            self.error,
            params,
            method = 'L-BFGS-B',
            bounds = [(0.01, 20.0), (0.01, 20.0), (0.01, 20.0)],
            options = {'xtol': 1e-8, 'disp': True, 'ftol': 1e-7, 'maxiter': 1e8},
        )
        return res