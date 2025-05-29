import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import DS_pseudo2 as ds
    from tests.helpers.synthetic_generators import generate_game_timeseries
    import numpy as np
    return ds, generate_game_timeseries, np


@app.cell
def _(ds, generate_game_timeseries, np):
    metrics = None
    rulebase = np.full(13, 0.5)
    fitness_scores = []
    for cycle_i in generate_game_timeseries(15)[0]:
        metrics = {
            'parameters.attacks_landed.lower': cycle_i[0],
            'parameters.attacks_landed.upper': cycle_i[1],
            'parameters.current_hp': cycle_i[2],
            'parameters.defenses.crouching': cycle_i[3],
            'parameters.defenses.standing': cycle_i[4],
            'parameters.lower_hits': cycle_i[5],
            'parameters.upper_hits': cycle_i[6]
        }
        # print(metrics)
        script = ds.generate_script(rulebase)
        print("Selected rules:", script)
        fitness = ds.calculate_fitness(metrics)
        print(f"Fitness score: {fitness:3f}")
        fitness_scores.append(fitness)
        updated_weights = ds.adjust_weights(script, fitness, rulebase)
        print("Updated weights:", np.round(updated_weights, 3))
        rulebase = updated_weights
    return (fitness_scores,)


@app.cell
def _(generate_game_timeseries, np):
    from keras.saving import load_model
    model = load_model('app/lstm/rule_adjustment_model_0_0_5.keras')
    # model.predict(generate_game_timeseries(20))[0].tolist()
    np.expand_dims(generate_game_timeseries(10)[0, 3:5, 2], axis=0)
    return (model,)


@app.cell
def _():
    return


@app.cell
def _(ds, generate_game_timeseries, model, np):
    GAME_CYCLES = 15

    game_rulebase = np.full(13, 0.5)
    nds_fitness_scores = []
    game_metrics = generate_game_timeseries(GAME_CYCLES)
    for cycle_n in range(GAME_CYCLES):
        cycle_metrics = {
            'parameters.attacks_landed.lower': game_metrics[0, cycle_n, 0],
            'parameters.attacks_landed.upper': game_metrics[0, cycle_n, 1],
            'parameters.current_hp': game_metrics[0, cycle_n, 2],
            'parameters.defenses.crouching': game_metrics[0, cycle_n, 3],
            'parameters.defenses.standing': game_metrics[0, cycle_n, 4],
            'parameters.lower_hits': game_metrics[0, cycle_n, 5],
            'parameters.upper_hits': game_metrics[0, cycle_n, 6]
        }
        game_metrics[0, cycle_n]
        cumulative_metrics = np.expand_dims(game_metrics[0, 0:cycle_n + 1], axis=0)
        cycle_script = ds.generate_script(game_rulebase)
        print("Selected rules:", cycle_script)
        cycle_fitness = ds.calculate_fitness(cycle_metrics)
        print(f"Fitness score: {cycle_fitness:3f}")
        nds_fitness_scores.append(cycle_fitness)
        cycle_weights = ds.adjust_weights(cycle_script, cycle_fitness, game_rulebase)
        print("Cycle weights:", np.round(cycle_weights, 3))
        # print("Cumulative metrics: ", np.round(cumulative_metrics, 3))
        lstm_cycle_adjustment = model.predict(cumulative_metrics)[0]
        print("LSTM Cycle Adjustment", np.round(lstm_cycle_adjustment, 3))
        game_rulebase = lstm_cycle_adjustment
        print("Updated rulebase:", np.round(game_rulebase, 3))
    return (nds_fitness_scores,)


@app.cell
def _(fitness_scores, nds_fitness_scores):
    fitness_scores, nds_fitness_scores
    return


@app.cell
def _(fitness_scores, mo, nds_fitness_scores):
    import matplotlib.pyplot as plt

    plt.figure(facecolor='white')
    plt.plot(fitness_scores, label='ds fitness', color='blue')
    plt.plot(nds_fitness_scores, label='nds fitness', color='green')
    plt.xlabel("cycle", color='black')
    plt.ylabel("fitness score", color='black')
    plt.legend(facecolor='white', edgecolor='black', labelcolor='black')
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    mo.mpl.interactive(ax)
    return


if __name__ == "__main__":
    app.run()
