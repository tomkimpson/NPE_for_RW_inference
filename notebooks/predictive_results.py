import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():

    """
    Import required modules
    """
    import sys
    import os


    # Add src directory to path
    sys.path.append(os.path.join(os.getcwd(), 'src')) # Assumes notebook launched from project root



    import matplotlib.pyplot as plt
    import scienceplots  # Required for the 'science' style to be available
    from predict import load_prediction_results,compute_prediction_intervals,plot_prediction_intervals


    prediction_data_1D = load_prediction_results('results/workflow_20250827_224139/inference_results/predictive_results.pkl')
    prediction_data_2D = load_prediction_results('results/workflow_20250919_083626/inference_results/predictive_results.pkl')



    def generate_posterior_predictive_plot(data,save_path):

        # Extract
        prediction_results = data['prediction_results']
        input_metadata = data['input_metadata']
        observed_data = data.get('observed_data', None)

        # Extract simulation parameters from loaded metadata
        sim_params = input_metadata['simulation_params']
        actual_Lx = sim_params['Lx']
        actual_Ly = sim_params['Ly']
        actual_T = sim_params['T']

        print(f"⚙️  Using simulation settings from loaded data: Lx={actual_Lx}, Ly={actual_Ly}, T={actual_T}")

        #Recompute prediction intervals with a smoothing kernel (optional)
        raw_predictions = prediction_results['predictions'] 
        prediction_results = compute_prediction_intervals(
                        predictions=raw_predictions,
                        percentiles=prediction_results['metadata']['percentiles'],
                        smooth_sigma=1.0)


        # Main prediction plot
        fig2 = plot_prediction_intervals(
            prediction_results=prediction_results,
            observed_data=observed_data,
            Lx=actual_Lx
        )
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
        plt.show()
    return (
        generate_posterior_predictive_plot,
        prediction_data_1D,
        prediction_data_2D,
    )


@app.cell
def _(generate_posterior_predictive_plot, prediction_data_1D):
    generate_posterior_predictive_plot(prediction_data_1D,'notebooks/images/example_predictive_plot_for_paper_1D.png')
    return


@app.cell
def _(generate_posterior_predictive_plot, prediction_data_2D):
    generate_posterior_predictive_plot(prediction_data_2D,'notebooks/images/example_predictive_plot_for_paper_2D.png')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
