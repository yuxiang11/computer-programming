# Initial values
N = 5180493
S_0 = (N - 11) / N
E_0 = 10 / N
I_0 = 1 / N
R_0 = 0
init_vals = [S_0, E_0, I_0, R_0]

# Params
epsilon, beta, gamma = [0.2, 1.75, 0.5]
params = epsilon, beta, gamma

# define time interval 
t_max = 1000
dt = 1
t = np.linspace(0, t_max, int(t_max / dt) + 1)

# Run simulation
results = base_seir_model(init_vals, params, t)





def plot_model(
    simulated_susceptible, simulated_exposure, simulated_infectious, simulated_remove
):
    
    global times, numTimes
    startInd = 0
    numTimes = len(simulated_infectious)

    fig = plt.figure(figsize=[22, 12], dpi=120)
    fig.subplots_adjust(top=0.85, right=0.92)
    ind = np.arange(numTimes)
    indObs = np.arange(len(simulated_infectious))

    ax = fig.add_subplot(111)
    ax.yaxis.grid(True, color='black', linestyle='dashed')
    ax.xaxis.grid(True, color='black', linestyle='dashed')
    ax.set_axisbelow(True)
    fig.autofmt_xdate()

    (infectedp,) = ax.plot(indObs, simulated_infectious, linewidth=3, color='black')
    (sp,) = ax.plot(ind, simulated_susceptible, linewidth=3, color='red')
    (ep,) = ax.plot(ind, simulated_exposure, linewidth=3, color='purple')
    (ip,) = ax.plot(ind, simulated_infectious, linewidth=3, color='blue')
    (rp,) = ax.plot(ind, simulated_remove, linewidth=3, color='orange')
    ax.set_xlim(0, numTimes)
    ax.set_xlabel('Days')
    ax.set_ylabel('Population ratio')

    plt.legend(
        [sp, ep, ip, rp],
        [
            'susceptible',
            'Exposure, asymptomagtic',
            'Infected, symptomatic',
            'rehabilitation',
        ],
        loc='upper right',
        bbox_to_anchor=(1, 1.22),
        fancybox=True,
    )
    
plot_model(results[:200, 0], results[:200, 1], results[:200, 2], results[:200, 3])





df = pd.read_csv('./data/data.csv', parse_dates=['dateRep'])
df = df[(df.countriesAndTerritories == 'United_Kingdom')]
df = df.sort_values(by = 'dateRep')
df = df[['dateRep', 'cases', 'deaths', 'countriesAndTerritories']]
df.rename(columns = {'dateRep': 'date', 'countriesAndTerritories': 'county'}, inplace = True)
df.head(5)