# Set up population distribution in initial state
min_ratio = 221000/13816
max_ratio = 442000/13816
infected_cases = df.cases / N - df.deaths / N

# Instantiate the class
min_seir_eval = OptimizeModelParameters(init_vals, infected_cases * min_ratio)
max_seir_eval = OptimizeModelParameters(init_vals, infected_cases * max_ratio)

# Run optimiza function
min_opt_p = min_seir_eval.optimize(params)
max_opt_p = max_seir_eval.optimize(params)





min_results = base_seir_model(init_vals, min_opt_p.x, t)
max_results = base_seir_model(init_vals, max_opt_p.x, t)

min_simulated_cases = (min_results[:days,1] + min_results[:days,2]) * N/min_ratio
min_simulated_cases = [int(x) for x in min_simulated_cases]

max_simulated_cases = (max_results[:days,1] + max_results[:days,2]) * N/max_ratio
max_simulated_cases = [int(x) for x in max_simulated_cases]

avg_simulated_cases = [sum(i)/(2*N) for i in zip(min_simulated_cases, max_simulated_cases)]

validate_model(avg_simulated_cases, df.cases / N - df.deaths / N)





# Run simulation
results = base_seir_model(init_vals, params, t)
print('Forecasted maximum confrimed numbers: %s' % str(int(max(results[:, 2]) * N)))
plot_model(results[:200, 0], results[:200, 1], results[:200, 2], results[:200, 3])